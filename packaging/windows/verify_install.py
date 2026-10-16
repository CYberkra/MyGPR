"""Exercise the actual installer in a new, owned test directory on Windows."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time


SMOKE_PAGES = (
    'projectInterface', 'processingInterface', 'interpretationInterface',
    'spatialInterface', 'deliveryInterface', 'jobsInterface', 'settingsInterface',
)
UNINSTALL_KEY = (r'Software\Microsoft\Windows\CurrentVersion\Uninstall'
                 r'\{3B6B778E-9BFA-4D07-B321-3C58D39EAA04}_is1')


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('installer', type=Path)
    parser.add_argument('--hdf5', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--expected-version', required=True,
                        help='Required installed application, report and About version')
    return parser.parse_args(argv)


def ensure_clean_installation(registry=None):
    if registry is None:
        import winreg as registry
    for hive in (registry.HKEY_CURRENT_USER, registry.HKEY_LOCAL_MACHINE):
        for view in (registry.KEY_WOW64_64KEY, registry.KEY_WOW64_32KEY):
            try:
                with registry.OpenKey(hive, UNINSTALL_KEY, 0, registry.KEY_READ | view):
                    raise RuntimeError('Existing MyGPR installer registration: run QA in a clean Windows account/VM')
            except FileNotFoundError:
                pass


def verify_package_report(package, expected_version):
    assert package['frozen'] and package['status'] == 'passed', package
    versions = package['versions']
    assert set(versions) == {'distribution', 'report', 'settings'}, versions
    assert set(versions.values()) == {expected_version}, versions
    assert package.get('hdf5', {}).get('source_unchanged') is True, package.get('hdf5')


def verify_smoke_screenshots(folder):
    from PIL import Image

    expected = {*(f'page_{page}.png' for page in SMOKE_PAGES), 'window_overall.png'}
    actual = {path.name for path in folder.glob('*.png')}
    assert actual == expected, {'missing': sorted(expected - actual), 'unexpected': sorted(actual - expected)}
    for name in sorted(expected):
        try:
            with Image.open(folder / name) as screenshot:
                assert screenshot.format == 'PNG', name
                screenshot.verify()
        except (OSError, SyntaxError) as exc:
            raise AssertionError(f'Invalid smoke screenshot: {name}') from exc
    return sorted(expected)


def file_sha256(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def main():
    args = parse_args()
    # Never replace a user's registered installation, even when /DIR differs.
    ensure_clean_installation()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    shots_dir = output / 'native-smoke-shots'
    if any(shots_dir.glob('*.png')):
        raise RuntimeError('Smoke screenshots already exist: use a fresh QA output directory')
    source = args.hdf5.resolve()
    source_before = file_sha256(source)
    sandbox = Path(tempfile.mkdtemp(prefix='MyGPR-installer-QA-'))
    install = sandbox / 'app'
    sentinel = sandbox / 'user-project.h5'
    sentinel.write_bytes(b'MyGPR QA: uninstall must preserve user data')
    evidence = {'installer_sha256': file_sha256(args.installer),
                'expected_version': args.expected_version,
                'hdf5_source': str(source), 'hdf5_sha256_before': source_before,
                'sandbox': str(sandbox), 'checks': {}, 'status': 'failed'}
    env = os.environ.copy()
    env.pop('PYTHONPATH', None)
    env.update(QT_QPA_PLATFORM='windows:darkmode=0', QT_ACCESSIBILITY='1',
               MYGPR_SETTINGS_FILE=str(sandbox / 'config/settings.json'),
               MYGPR_RUNTIME_ROOT=str(sandbox / 'runtime'),
               MYGPR_SMOKE_SHOTS_DIR=str(shots_dir))
    env['PATH'] = os.pathsep.join([str(Path(os.environ['SystemRoot']) / 'System32'), os.environ['SystemRoot']])

    def run(name, command, *, expected=0, timeout=180):
        result = subprocess.run([str(value) for value in command], cwd=sandbox, env=env,
                                capture_output=True, timeout=timeout)
        (output / f'{name}.stdout.log').write_bytes(result.stdout + result.stderr)
        evidence['checks'][name] = result.returncode
        print(name, result.returncode, flush=True)
        assert result.returncode == expected, (name, result.returncode)

    setup = [args.installer.resolve(), '/VERYSILENT', '/SUPPRESSMSGBOXES', '/NORESTART', '/SP-',
             '/CURRENTUSER', '/NOICONS', f'/DIR={install}']
    try:
        run('install', [*setup, f'/LOG={output / "install.log"}'])
        executable = install / 'MyGPR.exe'
        assert executable.exists()
        run('installed-package', [executable, '--package-check', output / 'installed-package.json',
                                  '--hdf5', source])
        package = json.loads((output / 'installed-package.json').read_text(encoding='utf-8'))
        verify_package_report(package, args.expected_version)
        run('negative-package', [executable, '--package-check', output / 'negative-package.json',
                                 '--hdf5', sandbox / 'missing.h5'], expected=1)
        run('native-smoke', [executable, '--smoke'])
        evidence['checks']['native-smoke-pages'] = verify_smoke_screenshots(shots_dir)
        run('reinstall', setup)
        from pywinauto import Application
        process = subprocess.Popen([str(executable)], cwd=sandbox, env=env)
        try:
            app = Application(backend='uia').connect(process=process.pid, timeout=60)
            window = app.window(title_re='.*MyGPR.*')
            window.wait('visible', timeout=60)
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                buttons = [control for control in window.descendants()
                           if control.window_text() in ('设置', '系统设置') and control.is_visible()]
                if buttons:
                    buttons[0].click_input()
                    break
                time.sleep(0.5)
            else:
                raise AssertionError('Settings navigation not exposed by UIA')
            from pywinauto.mouse import scroll
            rectangle = window.rectangle()
            scroll(coords=(rectangle.right - 300, rectangle.top + rectangle.height() // 2), wheel_dist=-20)
            window.child_window(title=args.expected_version, control_type='Text').wait('visible', timeout=20)
            window.capture_as_image().save(output / 'installed-settings.png')
            evidence['checks']['native-settings-version'] = args.expected_version
            window.close()
            evidence['checks']['native-close'] = process.wait(timeout=30)
            assert evidence['checks']['native-close'] == 0
        finally:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=15)
        # An unknown file in the installation folder must also survive uninstall.
        retained = install / 'user-created.txt'
        retained.write_text('preserve this user-created file', encoding='utf-8')
        run('uninstall', [install / 'unins000.exe', '/VERYSILENT', '/SUPPRESSMSGBOXES', '/NORESTART',
                          f'/LOG={output / "uninstall.log"}'])
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            uninstall_log = (output / 'uninstall.log').read_text(encoding='utf-8-sig', errors='replace')
            if not executable.exists() and 'Uninstallation process succeeded.' in uninstall_log:
                break
            time.sleep(0.2)
        else:
            raise AssertionError('Uninstall did not finish successfully')
        assert not executable.exists()
        assert retained.read_text(encoding='utf-8') == 'preserve this user-created file'
        assert sentinel.read_bytes() == b'MyGPR QA: uninstall must preserve user data'
        evidence['checks']['user-data-preserved'] = True
        assert file_sha256(source) == source_before, 'HDF5 source changed during installer QA'
        evidence['checks']['hdf5-source-unchanged'] = True
        evidence['status'] = 'passed'
    except Exception as exc:
        evidence['error'] = str(exc)
        raise
    finally:
        evidence['hdf5_sha256_after'] = file_sha256(source) if source.exists() else None
        (output / 'installer-verification.json').write_text(json.dumps(evidence, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
