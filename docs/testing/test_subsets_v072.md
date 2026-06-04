# MyGPR V0.7.2 Test Subsets

V0.7.2 adds pytest markers and a small subset runner so full-repository checks can be staged instead of run as one opaque block.

Recommended commands:

```bash
python scripts/run_test_subset.py preflight
python scripts/run_test_subset.py unit
python scripts/run_test_subset.py gui
python scripts/run_test_subset.py integration
python scripts/run_test_subset.py gprmax
python scripts/run_test_subset.py wavelet
```

Marker meanings:

- `unit`: fast, headless core/helper tests.
- `gui`: tests requiring PyQt6/qfluentwidgets or a Qt event loop.
- `integration`: multi-module workflow, CLI, sidecar, report, or evidence path tests.
- `slow`: long-running validation/benchmark/runner tests.
- `gprmax`: gprMax campaign/conversion/pairing/benchmark contract tests.
- `wavelet`: tests requiring PyWavelets/pywt.

Full pytest remains available with:

```bash
python scripts/run_test_subset.py all
```

For headless Linux CI, keep `QT_QPA_PLATFORM=offscreen` or run through `xvfb-run` when needed.
