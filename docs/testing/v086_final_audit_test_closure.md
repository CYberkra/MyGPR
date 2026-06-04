# V0.8.6 Final Audit Test Closure

V0.8.6 addresses the remaining test-boundary issues found in the V0.8.5 final audit.

## What changed

- `test_bscan_display_export.py` now checks high-resolution export across the correct module boundary:
  - toolbar export in `app_qt.py`
  - report package export in `ui/report_export_controller.py`
- `wavelet` subset is now treated as a Qt-capable headless subset and uses `xvfb-run` on POSIX when available.
- `wavelet` subset runs focused `wavelet or pywt` tests rather than every test in every file that mentions wavelet.
- Test runner sets `MPLBACKEND=Agg` for non-GUI subsets and `QtAgg` only for display-capable subsets.
- `integration` subset uses file-level staged execution to avoid cross-file Qt teardown/plugin hangs in headless sandboxes.

## Recommended commands

```bash
python scripts/run_test_subset.py run baseline
python scripts/run_test_subset.py run gui-smoke -- -q
python scripts/run_test_subset.py run wavelet -- -q
python scripts/run_test_subset.py run integration -- -q
```

`slow`, `gprmax`, and `all` remain budgeted lanes and should be run explicitly when touching simulation, long-running validation, or release packaging.

## Verified in this audit

- preflight: passed
- unit: 303 passed, 18 warnings
- gui-smoke: 10 passed
- wavelet: 8 passed, 158 deselected
- B-scan/export static regression: 4 passed

Integration was also checked by targeted file groups to confirm the V0.8.5 stale static failure is resolved and the main export/evidence/sidecar/profile files pass.
