"""Build a Windows installer from an immutable source export and this recipe."""
import argparse
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import platform
import shutil
import struct
import subprocess
import sys
import zipfile

RECIPE = Path(__file__).resolve().parent
ROOT = RECIPE.parents[1]


def run(command, **kwargs):
    print(subprocess.list2cmdline([str(item) for item in command]), flush=True)
    subprocess.run([str(item) for item in command], check=True, **kwargs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-ref', default='v0.10.0')
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--inno', required=True, type=Path)
    args = parser.parse_args()
    if sys.platform != 'win32' or struct.calcsize('P') != 8:
        raise RuntimeError('Build requires 64-bit Windows Python')
    if sys.version_info[:2] == (3, 12) and sys.version_info[:3] < (3, 12, 12):
        raise RuntimeError('Use Python 3.12.12+ for freezing; older compiler/runtime patches fail SciPy imports')
    output = args.output.resolve()
    # Only task-owned output below the repository output tree may be replaced.
    allowed = (ROOT / 'output').resolve()
    if not output.is_relative_to(allowed) or output == allowed:
        raise ValueError('Build output must be a subdirectory of repository/output')
    output.mkdir(parents=True, exist_ok=True)
    source = (output / 'source').resolve()
    if not source.is_relative_to(output):
        raise ValueError('Source export must remain inside the build output')
    if source.exists():
        shutil.rmtree(source)
    source.mkdir()
    commit = subprocess.check_output(
        ['git', 'rev-parse', '--verify', f'{args.source_ref}^{{commit}}'], cwd=ROOT, text=True).strip()
    archive = output / 'source.zip'
    run(['git', 'archive', '--format=zip', '-o', archive, commit], cwd=ROOT)
    with zipfile.ZipFile(archive) as handle:
        handle.extractall(source)
    version = (source / 'VERSION').read_text(encoding='utf-8-sig').strip()
    if metadata.version('mygpr') != version:
        raise ValueError('Install the source-ref wheel into the isolated build environment first')
    import tomllib
    from packaging.requirements import Requirement
    project = tomllib.loads((source / 'pyproject.toml').read_text(encoding='utf-8'))['project']
    dependencies = project['dependencies'] + project['optional-dependencies']['gui']
    for dependency in dependencies:
        requirement = Requirement(dependency)
        if requirement.marker is None or requirement.marker.evaluate():
            if metadata.version(requirement.name) not in requirement.specifier:
                raise ValueError(f'Build environment does not satisfy {dependency}')
    run([sys.executable, '-m', 'pip', 'check'])
    from PIL import Image, ImageDraw
    from PyInstaller.utils.win32.versioninfo import (
        FixedFileInfo, StringFileInfo, StringStruct, StringTable, VarFileInfo, VarStruct, VSVersionInfo,
    )
    icon = output / 'MyGPR.ico'
    logo = Image.new('RGBA', (256, 256), '#0078d4')
    drawing = ImageDraw.Draw(logo)
    drawing.line([(24, 128), (64, 128), (88, 48), (120, 208), (152, 80), (184, 128), (232, 128)],
                 fill='white', width=14, joint='curve')
    logo.save(icon, sizes=[(size, size) for size in (16, 32, 48, 64, 128, 256)])
    version_tuple = (*map(int, version.split('.')), 0)
    resource = VSVersionInfo(
        ffi=FixedFileInfo(filevers=version_tuple, prodvers=version_tuple,
                          mask=0x3f, flags=0, OS=0x40004, fileType=1, subtype=0, date=(0, 0)),
        kids=[StringFileInfo([StringTable('040904B0', [StringStruct(key, value) for key, value in {
            'CompanyName': 'MyGPR Project', 'FileDescription': 'MyGPR',
            'FileVersion': version, 'InternalName': 'MyGPR', 'OriginalFilename': 'MyGPR.exe',
            'ProductName': 'MyGPR', 'ProductVersion': version,
        }.items()])]), VarFileInfo([VarStruct('Translation', [1033, 1200])])],
    )
    version_info = output / 'version-info.txt'
    version_info.write_text(str(resource), encoding='utf-8')
    notices = output / 'third-party-licenses'
    notices.mkdir(exist_ok=True)
    packages = []
    for distribution in metadata.distributions():
        name = distribution.metadata['Name']
        packages.append({'name': name, 'version': distribution.version,
                         'license': distribution.metadata.get('License-Expression') or
                                    distribution.metadata.get('License', '')})
        for file in distribution.files or []:
            if any(part.lower().startswith(('license', 'copying', 'notice')) for part in file.parts):
                origin = Path(distribution.locate_file(file))
                if origin.is_file():
                    destination = notices / name / str(file).replace('../', '')
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(origin, destination)
    inventory = output / 'build-environment.json'
    inventory.write_text(json.dumps({'scope': 'build environment, including QA/build tools; not a bundled-module claim',
                                    'packages': sorted(packages, key=lambda row: row['name'].lower())},
                                   indent=2), encoding='utf-8')
    run([sys.executable, '-m', 'cyclonedx_py', 'environment', '--output-reproducible',
         '--of', 'JSON', '-o', output / 'windows-build-environment.cdx.json'])
    run([sys.executable, '-m', 'piplicenses', '--format=json', '--with-urls',
         '--output-file', output / 'windows-environment-licenses.json'])
    environment = os.environ.copy()
    environment.pop('PYTHONPATH', None)
    environment['PATH'] = os.pathsep.join([str(Path(sys.executable).parent),
                                          str(Path(os.environ['SystemRoot']) / 'System32'),
                                          os.environ['SystemRoot']])
    environment.update(MYGPR_FREEZE_SOURCE=str(source), MYGPR_FREEZE_ICON=str(icon),
                       MYGPR_FREEZE_VERSION_INFO=str(version_info))
    run([sys.executable, '-m', 'PyInstaller', '--noconfirm', '--clean',
         '--distpath', output / 'dist', '--workpath', output / 'build', RECIPE / 'MyGPR.spec'],
        cwd=source, env=environment)
    dist = output / 'dist/MyGPR'
    shutil.copytree(notices, dist / '_internal/third-party-licenses', dirs_exist_ok=True)
    shutil.copy2(inventory, dist / '_internal/build-environment.json')
    for name in ('windows-build-environment.cdx.json', 'windows-environment-licenses.json'):
        shutil.copy2(output / name, dist / '_internal' / name)
    recipe_hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                     for path in sorted(RECIPE.iterdir()) if path.is_file()}
    provenance = {'version': version, 'application_commit': commit, 'application_ref': args.source_ref,
                  'python': sys.version, 'platform': platform.platform(), 'architecture': platform.machine(),
                  'build_environment': 'build-environment.json', 'recipe_sha256': recipe_hashes,
                  'pyinstaller': metadata.version('PyInstaller'), 'signed': False}
    (dist / '_internal/windows-build.json').write_text(json.dumps(provenance, indent=2), encoding='utf-8')
    check_environment = environment.copy()
    check_environment.update(QT_QPA_PLATFORM='offscreen',
                             MYGPR_SETTINGS_FILE=str(output / 'build-check/config/settings.json'))
    run([dist / 'MyGPR.exe', '--package-check', output / 'frozen-build-check.json'],
        cwd=output, env=check_environment, timeout=120)
    run([args.inno.resolve(), f'/DAppVersion={version}', f'/DDistDir={dist}',
         f'/DOutputDir={output / "assets"}', f'/DIconPath={icon}', RECIPE / 'MyGPR.iss'])
    (output / 'build-provenance.json').write_text(json.dumps(provenance, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
