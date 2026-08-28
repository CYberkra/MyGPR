# MyGPR Domain Context

## Core Concepts

### Project
 fieldwork task. A project groups all survey lines captured during a single fieldwork assignment. It owns a root directory, a manifest (`project.json`), and subdirectories for raw data, processed results, targets, and spatial exports.

### Line
The data for a single survey line. Each line has a stable identifier (e.g. `L01`), a GPR dataset file, metadata, and may have zero or more processed artifacts.

### Dataset
The raw GPR matrix and its metadata for one line. Loaded via `load_gpr_dataset(line_id)` and accessed in bounded windows via `read_window()`.

### Artifact
An immutable processing result produced by running a processing pipeline on a line. Stored under `processed/<line_id>/` with a descriptor, manifest, and params JSON.

### Processing Pipeline
A sequence of processing steps submitted as a job. Submitted via `submit_project_pipeline()` and watched through the job bridge.

### Target
An annotation object (e.g. suspected pipe, cavity) placed on a line. Stored as CSV under `targets/<line_id>_targets.csv`. Currently only storage is implemented; the annotation UI page is not yet developed.

### Interface
A basal/overburden interface annotation for a line. Stored separately from targets. The interpretation UI controller exists but is not yet fully wired.

### Spatial Result
A derived coordinate export (trajectories, targets, interfaces) aggregated at the project level. Distinct from per-line processing artifacts.

### Job
An asynchronous backend task (pipeline run, import, quality check, etc.). Managed by `JobBridge`, which polls and emits Qt signals.

### Preview Bundle
A bounded, downsampled view of a dataset or artifact for display in the UI. Never loads the full matrix into memory.

## Unresolved / Pending Clarification

- **Artifact vs Result**: code uses both terms; the business distinction is not yet confirmed.
- **Processing as verb**: whether "processing" refers only to running algorithms, or also to configuring parameters (drafts).