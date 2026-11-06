"""Use the pinned official Windows binary for real signature and ABI checks in CI."""
import base64
import ctypes
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from scripts.publish_windows_update import public_key

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def bundle():
    location = os.environ.get('MYGPR_TEST_WINSPARKLE')
    if sys.platform != 'win32' or not location:
        pytest.skip('Pinned native verifier runs in the Windows updates workflow')
    return Path(location) / 'WinSparkle-0.9.4'


def test_real_key_generation_signature_verification_and_tamper_rejection(bundle, tmp_path):
    tool = bundle / 'bin/winsparkle-tool.exe'
    key = tmp_path / 'publisher.key'
    generated = subprocess.run([sys.executable, '-m', 'scripts.publish_windows_update',
        '--tool', str(tool), '--private-key', str(key), '--generate-key'],
        cwd=ROOT, check=True, capture_output=True, text=True)
    pub = public_key(tool, key)
    assert generated.stdout.strip() == pub and len(base64.b64decode(pub)) == 32
    payload = tmp_path / 'MyGPR-0.12.0-Windows-x64-Setup.exe'
    payload.write_bytes(b'synthetic installer payload for signature testing')
    manifest = tmp_path / 'build.json'
    metadata = {'version': '0.12.0', 'update_public_key': pub, 'installer_name': payload.name,
                'installer_sha256': hashlib.sha256(payload.read_bytes()).hexdigest()}
    manifest.write_text(json.dumps(metadata))
    command = [sys.executable, '-m', 'scripts.publish_windows_update', '--tool', str(tool),
        '--private-key', str(key), '--installer', str(payload), '--build-manifest', str(manifest),
        '--version', '0.12.0', '--url', 'https://example.com/' + payload.name,
        '--notes-url', 'https://example.com/notes', '--output', str(tmp_path / 'feed')]
    subprocess.run(command, cwd=ROOT, check=True, capture_output=True)
    assert (tmp_path / 'feed/appcast.xml').is_file()
    signature = subprocess.check_output([str(tool), 'sign', '--private-key-file', str(key), str(payload)], text=True).strip()
    verify = [str(tool), 'verify', '--public-key', pub, '--signature', signature, str(payload)]
    subprocess.run(verify, check=True, capture_output=True)
    payload.write_bytes(b'changed payload')
    assert subprocess.run(verify, capture_output=True).returncode != 0
    assert subprocess.run(command, cwd=ROOT, capture_output=True).returncode != 0
    metadata['update_public_key'] = base64.b64encode(bytes(32)).decode()
    manifest.write_text(json.dumps(metadata))
    assert subprocess.run(command, cwd=ROOT, capture_output=True).returncode != 0


def test_real_dll_exports_and_public_key_abi_without_network_or_init(bundle):
    library = ctypes.CDLL(str((bundle / 'x64/Release/WinSparkle.dll').resolve()))
    for name in ('set_app_details', 'set_app_build_version', 'set_lang', 'set_appcast_url', 'cleanup',
                 'check_update_with_ui', 'check_update_without_ui', 'set_shutdown_request_callback',
                 'set_can_shutdown_callback', 'set_update_dismissed_callback', 'set_update_cancelled_callback'):
        assert getattr(library, 'win_sparkle_' + name)
    setter = library.win_sparkle_set_eddsa_public_key
    setter.argtypes, setter.restype = [ctypes.c_char_p], ctypes.c_int
    assert setter(base64.b64encode(bytes(range(32)))) == 1
    assert setter(b'invalid') == 0
