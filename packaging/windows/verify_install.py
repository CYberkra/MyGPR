"""Exercise the actual installer in a new, owned test directory on Windows."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('installer', type=Path)
    parser.add_argument('--hdf5', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    import winreg
    key = r'Software\Microsoft\Windows\CurrentVersion\Uninstall\{3B6B778E-9BFA-4D07-B321-3C58D39EAA04}_is1'
    for hive in (winreg.HKEY_CURRENT_USER, winreg.HKEY_LOCAL_MACHINE):
        for view in (winreg.KEY_WOW64_64KEY, winreg.KEY_WOW64_32KEY):
            try:
                with winreg.OpenKey(hive, key, 0, winreg.KEY_READ | view):
                    raise RuntimeError('Existing MyGPR installer registration: run QA in a clean Windows account/VM')
            except FileNotFoundError:
                pass
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    sandbox = Path(tempfile.mkdtemp(prefix='MyGPR-installer-QA-'))
    install = sandbox / 'app'
    sentinel = sandbox / 'user-project.h5'
    sentinel.write_bytes(b'MyGPR QA: uninstall must preserve user data')
    evidence = {'installer_sha256': hashlib.sha256(args.installer.read_bytes()).hexdigest(),
                'sandbox': str(sandbox), 'checks': {}}
    env = os.environ.copy()
    env.pop('PYTHONPATH', None)
    env.update(QT_QPA_PLATFORM='windows:darkmode=0', QT_ACCESSIBILITY='1',
               MYGPR_SETTINGS_FILE=str(sandbox / 'config/settings.json'),
               MYGPR_RUNTIME_ROOT=str(sandbox / 'runtime'),
               MYGPR_SMOKE_SHOTS_DIR=str(output / 'native-smoke-shots'))
    env['PATH'] = os.pathsep.join([str(Path(os.environ['SystemRoot']) / 'System32'), os.environ['SystemRoot']])

    def run(name, command, *, expected=0, timeout=180):
        result = subprocess.run([str(value) for value in command], cwd=sandbox, env=env,
                                capture_output=True, timeout=timeout)
        (output / f'{name}.log').write_bytes(result.stdout + result.stderr)
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
                                  '--hdf5', args.hdf5.resolve()])
        package = json.loads((output / 'installed-package.json').read_text(encoding='utf-8'))
        assert package['frozen'] and package['status'] == 'passed'
        assert set(package['versions'].values()) == {'0.10.0'}
        run('negative-package', [executable, '--package-check', output / 'negative-package.json',
                                 '--hdf5', sandbox / 'missing.h5'], expected=1)
        run('native-smoke', [executable, '--smoke'])
        shots = list((output / 'native-smoke-shots').glob('*.png'))
        assert len(shots) >= 9, shots
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
            window.child_window(title='0.10.0', control_type='Text').wait('visible', timeout=20)
            window.capture_as_image().save(output / 'installed-settings.png')
            evidence['checks']['native-settings-version'] = '0.10.0'
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
        assert not executable.exists()
        assert retained.read_text(encoding='utf-8') == 'preserve this user-created file'
        assert sentinel.read_bytes() == b'MyGPR QA: uninstall must preserve user data'
        evidence['checks']['user-data-preserved'] = True
        evidence['status'] = 'passed'
    finally:
        (output / 'installer-verification.json').write_text(json.dumps(evidence, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
