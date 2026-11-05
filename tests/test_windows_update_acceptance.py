"""Complete frozen applications and real HTTPS updater UI, explicitly dispatched on CI."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import urllib.request

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def acceptance():
    source = os.environ.get('MYGPR_CI_UPDATE_INPUTS')
    if sys.platform != 'win32' or os.environ.get('GITHUB_ACTIONS') != 'true' or not source:
        pytest.skip('Explicit Windows update_installation workflow with complete package artifacts required')
    spec = importlib.util.spec_from_file_location('verify_update', ROOT / 'packaging/windows/verify_update.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, Path(source)


def test_complete_legacy_installer_upgrade(acceptance):
    module, inputs = acceptance
    output = ROOT / 'output/ci-regressions/legacy-upgrade'
    output.mkdir(parents=True, exist_ok=True)
    previous = inputs / 'MyGPR-0.11.0-Windows-x64-Setup.exe'
    with urllib.request.urlopen('https://github.com/CYberkra/MyGPR/releases/download/v0.11.0/' + previous.name,
                                timeout=60) as response, previous.open('wb') as destination:
        while block := response.read(1024 * 1024):
            destination.write(block)
    with previous.open('rb') as stream:
        assert hashlib.file_digest(stream, 'sha256').hexdigest() == '2dd386865850996cd242c01021e4105af21c8483cc5af9e6516d9e731130a72b'
    release = json.loads((inputs / 'release/build-provenance.json').read_text(encoding='utf-8'))
    source = output / 'original.h5'
    module.synthetic_source(source)
    subprocess.run([sys.executable, str(ROOT / 'packaging/windows/verify_install.py'),
        str(inputs / 'release/assets' / release['installer_name']), '--previous-installer', str(previous),
        '--expected-version', release['version'], '--hdf5', str(source), '--output', str(output / 'verification')],
        check=True, timeout=600)


def test_complete_signed_native_update_and_rejections(acceptance):
    module, inputs = acceptance
    module.run_acceptance(inputs, ROOT / 'output/ci-regressions/native-update',
        os.environ['MYGPR_CI_INSTALLER_SIGNATURE'], ROOT / 'output/update-tls/server.pem',
        ROOT / 'output/update-tls/server.key',
        ROOT / 'output/update-tools/WinSparkle-0.9.4/bin/winsparkle-tool.exe')
