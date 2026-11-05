"""Compile and exercise the real Inno recipe, with tiny package fixtures, on CI."""
import hashlib
import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]


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
    package(new, {'_internal/new.dll': b'new dependency'})
    installers = []
    for version, source in (('0.11.4', old), ('0.11.5', new)):
        subprocess.run([compiler, f'/DAppVersion={version}', f'/DDistDir={source}',
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
        subprocess.run([str(installers[1]), *options], check=True, capture_output=True, timeout=60)
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
