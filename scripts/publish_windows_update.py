"""Prepare signed appcasts locally. This command never publishes a release."""
from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import xml.etree.ElementTree as ET

from core.storage_primitives import atomic_write_bytes

ROOT = Path(__file__).resolve().parents[1]
from mygpr.infrastructure.updates.configuration import https_url, windows_build_version

NAMESPACE = 'http://www.andymatuschak.org/xml-namespaces/sparkle'
ET.register_namespace('sparkle', NAMESPACE)


def restrict_private_key(path: Path):
    if os.name == 'nt':
        account = subprocess.check_output(['whoami.exe', '/user', '/fo', 'csv', '/nh'], text=True)
        sid = next(csv.reader([account.strip()]))[-1]
        if not re.fullmatch(r'S-1-[0-9-]+', sid):
            raise ValueError('Could not identify the publishing account')
        subprocess.run(['icacls.exe', str(path), '/inheritance:r', '/grant:r',
                        f'*{sid}:F', '*S-1-5-18:F'], check=True, capture_output=True)
    else:
        path.chmod(0o600)


def public_key(tool: Path, private_key: Path) -> str:
    output = subprocess.check_output([str(tool), 'public-key', '--private-key-file', str(private_key)], text=True)
    keys = re.findall(r'(?<![A-Za-z0-9+/])[A-Za-z0-9+/]{43}=(?![A-Za-z0-9+/=])', output)
    if len(set(keys)) != 1 or len(base64.b64decode(keys[0], validate=True)) != 32:
        raise ValueError('Signing tool did not return one Ed25519 public key')
    return keys[0]


def make_appcast(*, version, url, notes_url, signature, length):
    https_url(url)
    https_url(notes_url)
    if not url.endswith('.exe') or len(base64.b64decode(signature, validate=True)) != 64:
        raise ValueError('Invalid signed installer metadata')
    rss = ET.Element('rss', {'version': '2.0'})
    channel = ET.SubElement(rss, 'channel')
    ET.SubElement(channel, 'title').text = 'MyGPR'
    item = ET.SubElement(channel, 'item')
    ET.SubElement(item, 'title').text = f'MyGPR {version}'
    ET.SubElement(item, f'{{{NAMESPACE}}}releaseNotesLink').text = notes_url
    ET.SubElement(item, f'{{{NAMESPACE}}}minimumSystemVersion').text = '10.0.17763'
    ET.SubElement(item, 'enclosure', {
        'url': url, 'length': str(length), 'type': 'application/octet-stream',
        f'{{{NAMESPACE}}}version': windows_build_version(version),
        f'{{{NAMESPACE}}}shortVersionString': version,
        f'{{{NAMESPACE}}}os': 'windows-x64',
        f'{{{NAMESPACE}}}edSignature': signature,
        f'{{{NAMESPACE}}}installerArguments': '/SILENT /SP- /NOICONS /NORESTART /MYGPRUPDATE=1',
    })
    return ET.tostring(rss, encoding='utf-8', xml_declaration=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--private-key', required=True, type=Path)
    parser.add_argument('--tool', type=Path)
    parser.add_argument('--generate-key', action='store_true')
    parser.add_argument('--installer', type=Path)
    parser.add_argument('--build-manifest', type=Path)
    parser.add_argument('--version')
    parser.add_argument('--url')
    parser.add_argument('--notes-url')
    parser.add_argument('--channel', choices=('stable', 'preview'), default='stable')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    key = args.private_key.expanduser().resolve()
    if key.is_relative_to(ROOT):
        parser.error('Keep the private signing key outside the repository')
    tool = args.tool
    if tool is None:
        spec = importlib.util.spec_from_file_location('windows_update_tools', ROOT / 'packaging/windows/update_tools.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        tool = module.prepare_winsparkle(ROOT / 'output/update-tools') / 'bin/winsparkle-tool.exe'
    if args.generate_key:
        if key.exists():
            parser.error('Refusing to overwrite an existing signing key')
        key.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run([str(tool), 'generate-key', '--file', str(key)], check=True, capture_output=True)
        restrict_private_key(key)
        print(public_key(tool, key))  # The private key is never printed.
        return
    if not all((args.installer, args.build_manifest, args.version, args.url, args.notes_url, args.output)):
        parser.error('Signing requires installer, build-manifest, version, URL, notes URL and output')
    if args.channel == 'stable' and re.search(r'(?:a|b|rc)\d+$', args.version):
        parser.error('Pre-release versions cannot enter the stable channel')
    manifest = json.loads(args.build_manifest.read_text(encoding='utf-8'))
    pub = public_key(tool, key)
    if manifest.get('update_public_key') != pub or manifest.get('version') != args.version:
        raise ValueError('Signing key/version does not match the built installation trust root')
    with args.installer.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    if manifest.get('installer_sha256') != digest or manifest.get('installer_name') != args.installer.name:
        raise ValueError('Installer does not match the verified build manifest')
    signature = subprocess.check_output([str(tool), 'sign', '--private-key-file', str(key),
                                         str(args.installer)], text=True).strip()
    subprocess.run([str(tool), 'verify', '--public-key', pub, '--signature', signature,
                    str(args.installer)], check=True, capture_output=True)
    args.output.mkdir(parents=True, exist_ok=True)
    payload = make_appcast(version=args.version, url=args.url, notes_url=args.notes_url,
                           signature=signature, length=args.installer.stat().st_size)
    atomic_write_bytes(args.output / 'appcast.xml', payload)
    atomic_write_bytes(args.output / 'WINDOWS-SHA256SUMS.txt', f'{digest}  {args.installer.name}\n'.encode('utf-8'))
    print(f'Prepared verified {args.channel} appcast for {args.version}; not published')


if __name__ == '__main__':
    main()
