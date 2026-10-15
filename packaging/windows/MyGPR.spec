# Build the app from a clean, pinned source export; keep the build recipe separate.
import os
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_submodules, copy_metadata
import PyQt6

source = Path(os.environ["MYGPR_FREEZE_SOURCE"]).resolve()
recipe = Path(SPECPATH).resolve()
sys.path.insert(0, str(source))
hiddenimports = []
datas = []
for package in ("core", "mygpr", "ui", "PythonModule"):
    hiddenimports.extend(collect_submodules(package))
    datas.extend(collect_data_files(package, include_py_files=False))

# These Cython extensions import Python modules dynamically at runtime.
for package in ("rasterio", "fiona"):
    hiddenimports.extend(collect_submodules(package))
datas.extend(copy_metadata("mygpr"))
for name in ("VERSION", "pyproject.toml", "LICENSE", "THIRD_PARTY_NOTICES.md", "README.md"):
    datas.append((str(source / name), "."))
for path in (source / "config").rglob("*"):
    if path.is_file():
        relative = path.parent.relative_to(source)
        datas.append((str(path), relative.as_posix()))
datas.append((os.environ["MYGPR_FREEZE_ICON"], "."))

a = Analysis(
    [str(recipe / "launcher.py")],
    pathex=[str(source)],
    binaries=[],
    datas=datas,
    hiddenimports=sorted(set(hiddenimports)),
    hookspath=[],
    runtime_hooks=[str(recipe / "runtime_hook.py")],
    excludes=["pytest", "_pytest", "IPython", "jupyter", "notebook", "tkinter",
              "PyQt5", "PySide2", "PySide6"],
    noarchive=False,
)
# Pin the redistributable runtime supplied by Qt rather than an older Python
# installation's copy. The bootloader loads the root copy before runtime hooks.
qt_bin = Path(PyQt6.__file__).parent / 'Qt6' / 'bin'
runtime_names = ('VCRUNTIME140.dll', 'VCRUNTIME140_1.dll', 'MSVCP140.dll', 'MSVCP140_1.dll', 'MSVCP140_2.dll')
a.binaries = [entry for entry in a.binaries if entry[0] not in runtime_names]
a.binaries += [(name, str(qt_bin / name), 'BINARY') for name in runtime_names]
# A developer PATH may contain unrelated Poppler/ICU DLLs. Refuse to ship any
# native dependency from outside the selected Python/venv or Windows itself.
allowed = [Path(sys.prefix).resolve(), Path(sys.base_prefix).resolve(),
           Path(os.environ['SystemRoot']).resolve(), source, recipe]
for name, origin, kind in a.binaries:
    if not any(Path(origin).resolve().is_relative_to(root) for root in allowed):
        raise ValueError(f'Native dependency outside the build environment: {name}: {origin}')
pyz = PYZ(a.pure)
exe = EXE(
    pyz, a.scripts, [], exclude_binaries=True,
    name="MyGPR", debug=False, bootloader_ignore_signals=False,
    strip=False, upx=False, console=False,
    icon=os.environ["MYGPR_FREEZE_ICON"],
    version=os.environ["MYGPR_FREEZE_VERSION_INFO"],
    contents_directory="_internal",
)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name="MyGPR")
