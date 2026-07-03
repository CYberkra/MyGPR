# Processing Lineage Compare UX — V0.8.8

V0.8.8 narrows the processing-lineage controls to two explicit actions:

- **与当前对比**: compare the selected historical step against the formal current result.
- **退出对比**: close lineage slider compare and return to the single-image current result.

The compare pair is no longer implicit. The selected historical chip is always the left side, and the formal current result is always the right side. The display controls are synchronized to the visible `滑动对比` radio mode when lineage compare starts, and synchronized back to `单图` when compare closes.

This version intentionally removes the `复制链路` button from the main B-scan stepper to keep the primary analysis area focused. Text/export versions of the processing chain remain available internally for Evidence package generation.

Workflow editing is explicitly out of scope for V0.8.8. Reordering, deleting, inserting, or recomputing chain steps must be implemented as a separate workflow-editing feature with clear recomputation and Evidence rules.
