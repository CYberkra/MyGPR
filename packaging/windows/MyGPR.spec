# Build the app from a clean, pinned source export; keep the build recipe separate.
import os
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_submodules, copy_metadata

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
