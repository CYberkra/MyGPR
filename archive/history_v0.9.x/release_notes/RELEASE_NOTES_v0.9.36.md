# MyGPR v0.9.36 Release Notes

## Interpretation Lab migration

MyGPR Studio now provides a clean-room, reversible Interpretation Lab for continuous cover–bedrock interface production. The release adds assisted full-line or interval tracing, local signal snapping, curve smoothing and vertical shift, uncertainty bands, semantic-zone editing, undo/redo, screenshots and traceable training-label exports.

### Main capabilities

- Full-line and interval assisted tracing with bounded search, step, smoothing and output-size controls.
- Snap existing picks to local strong reflections without changing their original trace anchors.
- Edit interface type, confidence, notes and uncertainty width.
- Split, delete and adjust semantic zones, with all changes included in the reversible session history.
- Export `mygpr.interpretation_labels.v1` packages containing JSON labels, training NPZ arrays, source/result lineage, edit audit and SHA-256 integrity metadata.
- Persist new fields while remaining compatible with older annotation JSON documents.

### Architecture and safety

- Studio calls only the clean-room `frontend_sdk` contract.
- `InterpretationEditService` owns editing sessions and formal persistence.
- Assisted tracing resides in the pure domain layer rather than a legacy UI or `core` dependency.
- Existing Backend API v1 remains compatible.
- The 81-file frozen legacy frontend remains unchanged.

### Known release-environment limitation

Native Windows PyQt6 rendering, 100/125/150/175% DPI, EXE and installer acceptance still require the Windows release machine. Headless contract tests do not replace those checks.
