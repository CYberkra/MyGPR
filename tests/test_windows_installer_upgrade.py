"""Compile and exercise the real Inno recipe, with tiny package fixtures, on CI."""
import hashlib
import ctypes
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

ROOT = Path(__file__).resolve().parents[1]
from mygpr.infrastructure.updates.configuration import windows_build_version


@pytest.fixture
def compiler():
    location = os.environ.get('MYGPR_TEST_INNO')
    if sys.platform != 'win32' or not location or os.environ.get('GITHUB_ACTIONS') != 'true':
        pytest.skip('Installer integration runs only in the disposable Windows CI runner')
    spec = importlib.util.spec_from_file_location('install_qa', ROOT / 'packaging/windows/verify_install.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.ensure_clean_installation()
    return location


def package(folder, contents):
    folder.mkdir()
    contents = {'MyGPR.exe': b'test exe; not executed', '_internal/LICENSE': b'test license', **contents}
    managed = []
    for relative, data in contents.items():
        target = folder / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        managed.append(f'{hashlib.sha256(data).hexdigest()}|{relative}')
    (folder / 'mygpr-managed-files.txt').write_text('\n'.join(managed) + '\n')


def test_upgrade_removes_only_unchanged_owned_files_and_uninstall_preserves_user_files(compiler, tmp_path):
    from PIL import Image
    icon = tmp_path / 'app.ico'
    Image.new('RGBA', (32, 32), 'blue').save(icon)
    old, new, install = (tmp_path / name for name in ('old', 'new', 'installed'))
    package(old, {'_internal/obsolete.dll': b'old dependency', '_internal/modified.h5': b'packaged test fixture'})
    package(new, {'_internal/new.dll': b'new dependency',
                  'MyGPR.exe': (Path(os.environ['SystemRoot']) / 'System32/whoami.exe').read_bytes()})
    installers = []
    for version, source in (('0.11.4a1', old), ('0.11.4', new)):
        subprocess.run([compiler, f'/DAppVersion={version}', f'/DAppFileVersion={windows_build_version(version)}', f'/DDistDir={source}',
            f'/DOutputDir={tmp_path}', f'/DIconPath={icon}', str(ROOT / 'packaging/windows/MyGPR.iss')],
            check=True, capture_output=True)
        installers.append(tmp_path / f'MyGPR-{version}-Windows-x64-Setup.exe')
    options = ['/VERYSILENT', '/SUPPRESSMSGBOXES', '/NORESTART', '/SP-', '/CURRENTUSER', '/NOICONS', f'/DIR={install}']
    try:
        subprocess.run([str(installers[0]), *options], check=True, capture_output=True, timeout=60)
        user = install / 'original-import.h5'
        user.write_bytes(b'original acquisition must survive')
        modified = install / '_internal/modified.h5'
        modified.write_bytes(b'user changed this file')
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.CreateMutexW.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_wchar_p]
        kernel.CreateMutexW.restype = ctypes.c_void_p
        kernel.CloseHandle.argtypes = [ctypes.c_void_p]
        mutex = kernel.CreateMutexW(None, False, r'Local\MyGPR.Application')
        assert mutex
        log = tmp_path / 'upgrade.log'
        installer = subprocess.Popen([str(installers[1]), *options, '/MYGPRUPDATE=1', f'/LOG={log}'],
                                     stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            deadline = time.monotonic() + 15
            while time.monotonic() < deadline:
                if log.is_file() and 'Waiting for MyGPR to exit before upgrade' in log.read_text(encoding='utf-8-sig', errors='replace'):
                    break
                time.sleep(0.05)
            else:
                pytest.fail('Installer did not enter the application-exit wait')
            assert installer.poll() is None
            assert (install / '_internal/obsolete.dll').exists()
            assert not (install / '_internal/new.dll').exists()
        finally:
            kernel.CloseHandle(mutex)
        installer.communicate(timeout=60)
        assert installer.returncode == 0
        assert not (install / '_internal/obsolete.dll').exists()
        assert (install / '_internal/new.dll').read_bytes() == b'new dependency'
        assert user.read_bytes() == b'original acquisition must survive'
        assert modified.read_bytes() == b'user changed this file'
    finally:
        uninstaller = install / 'unins000.exe'
        if uninstaller.is_file():
            subprocess.run([str(uninstaller), '/VERYSILENT', '/SUPPRESSMSGBOXES', '/NORESTART'],
                           check=True, capture_output=True, timeout=60)
    assert user.read_bytes() == b'original acquisition must survive'
