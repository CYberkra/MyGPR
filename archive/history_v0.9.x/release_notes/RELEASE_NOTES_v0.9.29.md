# MyGPR 0.9.29 — Studio Frontend Integration

This release adds the clean-room PyQt6 Studio frontend while retaining the Phase-12 backend and legacy UI as a rollback path.

## New Studio workspaces

- Project management
- Measurement-line processing
- Interface interpretation and borehole validation
- Spatial results
- Reports and delivery
- Job center
- gprMax/SFCW research-mode export

## Backend integration changes

- Line import now preserves the user-selected dielectric constant through the public backend API.
- Processed artifacts can be previewed through the Studio adapter.
- Interpretation, borehole, spatial-result and simulation adapters are included without exposing Qt objects to the backend.

## Launch

```bash
python mygpr_studio.py --backend phase12
```

Use `--backend mock` for deterministic UI development data.
