# MyGPR Field Workbench Project Data Contract

Round 3 fixes the project data layout used by the 1080P field workbench UI.  The purpose is to make the product shell durable: opening the application, processing a line, editing targets, and exporting spatial coordinates all update project files on disk.

## Standard project layout

```text
MyGPR_Project/
├─ project.json
├─ raw/
├─ processed/
├─ targets/
├─ spatial/
├─ reports/
└─ logs/
```

The field workbench starts with no project open. A project directory is created only after the user explicitly chooses **New Project**, or loaded after **Open Project**. No demo project or synthetic field data is created by the production UI.

## `project.json`

Schema: `mygpr.field_project.v1`

Important fields:

| Field | Meaning |
| --- | --- |
| `project_id` | Stable UUID for the project. |
| `project_no` | Human-facing project number. |
| `name` | Project name shown in the header and project summary. |
| `location` | Survey area description. |
| `device_model` | GPR/RTK device label. |
| `coordinate_system` | Horizontal coordinate reference. |
| `vertical_datum` | Height datum. |
| `created_at`, `updated_at` | Local timestamps. |
| `operator` | Current operator label. |
| `status` | Project health state. |
| `lines` | List of line records. |

Each line record contains:

| Field | Meaning |
| --- | --- |
| `line_id` | Line ID, such as `L03`. |
| `name` | Display name. |
| `length_m` | Line length in meters. |
| `data_quality` | UI quality rating. |
| `rtk_status` | `固定解`, `浮动解`, or `未定位`. |
| `processing_status` | `未处理`, `已导入`, `处理中`, `已完成`, etc. |
| `raw_path`, `raw_rows`, `raw_size_mb` | Imported raw file reference and metadata. |
| `target_count` | Current saved target count. |
| `processed_result` | Relative path to the latest `.npy` result. |
| `params_path` | Relative path to the latest processing parameter JSON. |

## Raw data

Imported data is copied into:

```text
raw/<line_id>/<source_file>
```

For CSV files, the current UI records row count and size in `project.json`.

## Processed results

When the user clicks **保存处理结果**, the field UI writes:

```text
processed/<line_id>/<line_id>_processed_<YYYYMMDD_HHMMSS>.npy
processed/<line_id>/<line_id>_params.json
```

The parameter JSON contains the selected processing recipe, background-removal settings, band-pass settings, SEC gain factor, and motion-compensation settings.  `project.json` is updated to point to the latest files.

## Target annotations

Target annotations are saved as:

```text
targets/<line_id>_targets.csv
```

Columns:

```text
target_id,line_id,distance_m,depth_m,x,y,type,confidence,status,note,created_at,updated_at,source_result_id
```

This CSV is the source of truth for the target table, right-side target detail panel, and downstream spatial export.

## Spatial target export

Saving targets also regenerates:

```text
spatial/<line_id>_targets_xy.csv
```

Columns:

```text
target_id,line_id,x,y,distance_m,depth_m,type,status,confidence
```

This provides the first 2D spatial results bridge for GIS, CAD, and later report generation.

## Logs

The field workbench appends durable state changes to:

```text
logs/field_workbench.log
```

Typical entries include line import, processing result save, and target CSV save.

## Current scope

Round 3 intentionally keeps the existing processing backend untouched.  It establishes the file contract and persistence path so later rounds can replace deterministic UI data with real GPR arrays, real RTK/IMU trajectories, and report-generation artifacts without redesigning the project layout.
