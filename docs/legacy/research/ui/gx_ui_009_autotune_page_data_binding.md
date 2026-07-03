# GX-UI-009 AutoTune Page Data Binding

## Purpose

This change binds the new `AutoTuneTuningPage` / `AutoTune 参数推荐` page to the dataset currently loaded in the main MyGPR application.

The goal is metadata synchronization only. It does not run AutoTune, does not change production scoring logic, does not run gprMax, and does not write Evidence artifacts.

## Public API added

`ui/autotune_tuning_page.py` now exposes:

```python
def set_loaded_dataset(
    *,
    file_path: str | None = None,
    data_shape: tuple[int, int] | None = None,
    data_type: str | None = None,
    component: str | None = None,
    processing_stage: str | None = None,
    source_label: str | None = None,
) -> None
```

and:

```python
def clear_loaded_dataset() -> None
```

These methods update only the UI-local recommendation state and visible labels.

## Main-window binding

`app_qt.py` now calls `_sync_auto_tune_page_dataset_state(payload)` from `_on_shared_data_changed()`.

When `SharedDataState` emits `reason == "loaded"` or another data-state event, the binding extracts:

- current file path
- current data shape
- data type inferred from path/header/source
- current processing stage label
- optional component metadata, if present

The AutoTune page then updates from `未载入` to `已载入` and displays the dataset metadata in the header, preview cards, metrics/audit text, and recommendation text.

## Preview behavior

This task keeps the preview lightweight. The Raw/Input card displays:

- file name
- shape
- data type
- processing stage
- optional component

Real B-scan preview rendering is intentionally deferred.

## Boundaries

This change does not:

- execute AutoTune
- modify scoring logic
- run gprMax
- modify GX-008/GX-009 models
- modify MyGPR-Evidence
- add PyVista/PyVistaQt
- enable Evidence export
- remove legacy pages

## Known limitations

- Shape convention is displayed as `samples × traces` based on current array shape.
- Preview remains text-based.
- Top action buttons remain disabled/placeholder.
- AutoTune execution is still a future task.
