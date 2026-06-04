# MyGPR GPR input format compatibility baseline

Version baseline: 0.8.40

## Research basis

Common GPR processing ecosystems treat vendor/profile formats as a practical compatibility target, not just CSV. RGPR lists Sensors & Software `.DT1/.HD`, GSSI `.dzt`, MALÅ `.rd3/.rd7 + .rad`, ImpulseRadar `.iprb + .iprh`, SEG-Y `.sgy/.segy`, ENVI BSQ `.dat/.hdr`, ASCII `.txt`, Geotech OKO `.gpr/.gpr2`, IDS `.dt`, and other vendor formats as GPR import targets. GPRPy similarly documents support for `.DT1`, `.DZT`, `.rd3`, ENVI BSQ and native `.gpr` profiles.

MyGPR V0.8.40 therefore separates compatibility into three statuses:

- **native**: can be read directly by MyGPR's own loader.
- **native-subset**: MyGPR reads the documented common/simple profile subset and fails clearly for unsupported variants.
- **recognized**: MyGPR identifies the format as common GPR data, but does not silently decode it yet; users should convert through device software/RGPR/GPRPy or a future adapter.

## Current MyGPR compatibility table

| Format family | Extensions | Status | Notes |
|---|---|---|---|
| MyGPR / matrix CSV | `.csv`, `.txt` | native | Matrix B-scan and MyGPR UAV stacked CSV. |
| A-scan folder | directory of CSV | native | Existing folder reader. |
| gprMax output | `.out` with optional `.in` | native | HDF5 gprMax output plus `.in` metadata. |
| NumPy array | `.npy`, `.npz` | native | Internal/research exchange; must be 2-D matrix. |
| MALÅ RD3/RD7 | `.rd3`, `.rd7` + `.rad` | native-subset | Reads signed sequential trace data using `.rad` `SAMPLES`, `TIME WINDOW`, `DISTANCE INTERVAL`. |
| ImpulseRadar IPRB | `.iprb` + `.iprh` | native-subset | Reads 16/32-bit signed sequential trace data using `.iprh` `SAMPLES` and `DATA VERSION`. |
| SEG-Y fixed-length profile | `.sgy`, `.segy` | native-subset | Reads conservative fixed-length profile subset with common int16/int32/float32 big-endian samples. |
| ENVI BSQ | `.dat` + `.hdr` | native-subset | Reads basic ENVI band-sequential arrays. |
| Sensors & Software | `.dt1`, `.hd` | recognized | Format is recognized; direct decoder not yet enabled. |
| GSSI | `.dzt`, `.dzg` | recognized | Format is recognized; direct DZT decoder not yet enabled. |
| Geotech OKO | `.gpr`, `.gpr2` | recognized | Format is recognized; direct decoder not yet enabled. |

## Implementation boundary

V0.8.40 does not claim full vendor-format coverage. The goal is to avoid the previous CSV-only user experience and to create a stable import contract:

1. known file types appear in the file dialog;
2. native/lightweight readers return `data`, `header_info`, `path`, `format`;
3. recognized-but-not-native formats fail with an explicit conversion message;
4. no vendor binary is parsed as generic raw bytes without metadata.

## Next compatibility priorities

1. Add real `.DT1/.HD` decoder or adapter.
2. Add GSSI `.DZT` decoder or optional `readgssi`/GPRPy adapter.
3. Expand SEG-Y reader for extended textual headers and more sample formats.
4. Add sample fixture tests for each accepted native-subset format.
5. Add UI import diagnostics panel listing sidecar files found/missing.
