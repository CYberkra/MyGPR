# Start MyGPR

Python **3.12–3.13**. See [README](README.md) for exact dependency versions and [GUI guide](GUI_README.md) for desktop usage.

```powershell
python -m pip install -r requirements.txt -r requirements-gui.txt
python app_qt.py
```

For headless use, install only `requirements.txt` (the `requirements-core.txt` alias remains supported). Use `python cli_batch.py --help`, `python backend_smoke.py`, or the public `MyGPRBackend` API. Console scripts such as `mygpr-batch` require installing the package with `python -m pip install -e .`.

The design SPECs remain in the [specification index](docs/superpowers/specs/README.md); the old PyInstaller `.spec` is historical, not a current build entry.
