"""Fetch only the pinned official updater archive; never fetch code at app runtime."""
from __future__ import annotations

import hashlib
from pathlib import Path
import shutil
import urllib.request
import zipfile

VERSION = '0.9.4'
ARCHIVE_SHA256 = '6037df37fc263bd1650a1c4949681a9d40ffe991d01f35892a406cb5d103c976'
ARCHIVE_URL = f'https://github.com/vslavik/winsparkle/releases/download/v{VERSION}/WinSparkle-{VERSION}.zip'


def prepare_winsparkle(destination: Path) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    archive = destination / f'WinSparkle-{VERSION}.zip'
    if not archive.is_file():
        temporary = destination / '.winsparkle-download.tmp'
        with urllib.request.urlopen(ARCHIVE_URL, timeout=60) as response, temporary.open('wb') as output:
            total = 0
            while block := response.read(1024 * 1024):
                total += len(block)
                if total > 128 * 1024 * 1024:
                    raise ValueError('WinSparkle archive exceeds the pinned distribution size limit')
                output.write(block)
        temporary.replace(archive)
    with archive.open('rb') as stream:
        if hashlib.file_digest(stream, 'sha256').hexdigest() != ARCHIVE_SHA256:
            raise ValueError('Official WinSparkle archive checksum mismatch')
    bundle = destination / f'WinSparkle-{VERSION}'
    members = ('x64/Release/WinSparkle.dll', 'bin/winsparkle-tool.exe', 'COPYING', 'COPYING.expat')
    with zipfile.ZipFile(archive) as zipped:
        for member in members:
            target = bundle / member
            target.parent.mkdir(parents=True, exist_ok=True)
            with zipped.open(f'WinSparkle-{VERSION}/{member}') as source, target.open('wb') as output:
                shutil.copyfileobj(source, output, 1024 * 1024)
    return bundle
