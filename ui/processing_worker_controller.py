# -*- coding: utf-8 -*-
"""Processing worker completion controller for the main MyGPR window.

This module extracts the large worker-finished callback from ``app_qt.py``. It
keeps behavior delegated to the host window while making the main window shell
smaller and easier to audit.
"""

from __future__ import annotations

import time

from core.preset_profiles import compute_quality_metrics
from core.runtime_warnings import build_runtime_warning


class ProcessingWorkerController:
    """Own processing-worker completion handling for a host ``GPRGuiQt``."""

    def __init__(self, host):
        self.host = host

    def on_worker_finished(self, result: dict) -> None:
        """工作线程完成回调"""
        host = self.host
        host._set_busy(False, text="就绪")
        outputs = result.get("outputs", [])
        for item in outputs:
            host._append_runtime_warnings(
                item.get("runtime_warnings", []),
                source=item.get("method_key") or item.get("method_name"),
                log=False,
            )
        final_data = result.get("final_data")
        final_header_info = result.get("final_header_info")
        final_trace_metadata = result.get("final_trace_metadata")
        final_display_data = result.get("final_display_data")
        final_display_header_info = result.get("final_display_header_info")
        final_display_trace_metadata = result.get("final_display_trace_metadata")
        cancelled = result.get("cancelled", False)
        ctx = host._current_run_context or {}
        run_type = ctx.get("run_type", "")
        run_metadata = ctx.get("run_metadata") if isinstance(ctx.get("run_metadata"), dict) else {}
        autotune_scoring_record = dict(run_metadata.get("autotune_scoring_record") or {})
        autotune_recipe_plan = dict(run_metadata.get("autotune_recipe_plan") or {})

        if cancelled:
            host._log("处理已取消。")
            host.status_label.setText("已取消")
            host.page_basic.set_apply_button_state("idle", "处理已取消，当前数据保持不变。")
            host._set_runtime_summary("状态：处理已取消", "warning")
            # Live step previews are display-only.  If the run is cancelled,
            # restore the formally committed B-scan so users do not mistake an
            # uncommitted intermediate preview for current data.
            try:
                host._clear_display_override()
                if host.data is not None:
                    host.plot_data(host.data)
            except Exception as exc:
                host._log(f"取消后恢复正式 B-scan 失败: {exc}", event_type="WARN", source="processing")
        elif final_data is not None:
            is_kirchhoff = (
                len(outputs) == 1
                and outputs[0].get("method_key") == "kirchhoff_migration"
            )
            if len(outputs) == 1:
                snap_label = outputs[0].get(
                    "method_name", outputs[0].get("method_key", "处理")
                )
            else:
                names = [
                    o.get("method_name", o.get("method_key", "?")) for o in outputs
                ]
                snap_label = (
                    f"{names[0]}+{len(names) - 1}步" if len(names) > 1 else names[0]
                )
            # Preserve each executed step in the formal lineage for sequential
            # recipe/pipeline runs.  The final output remains the current result,
            # so only intermediate outputs are appended as history snapshots.
            if result.get("execution_mode") != "independent" and len(outputs) > 1:
                for idx, item in enumerate(outputs[:-1], start=1):
                    step_data = item.get("data")
                    if step_data is None:
                        continue
                    step_label = item.get("method_name") or item.get("method_key") or f"步骤 {idx}"
                    step_header = dict(item.get("header_info") or {})
                    if autotune_scoring_record:
                        step_header.setdefault("autotune_scoring_record", autotune_scoring_record)
                    if autotune_recipe_plan:
                        step_header.setdefault("autotune_recipe_plan", autotune_recipe_plan)
                    try:
                        host.shared_data.append_history_snapshot(
                            step_data,
                            label=str(step_label),
                            header_info=step_header,
                            trace_metadata=item.get("trace_metadata"),
                        )
                    except Exception as exc:
                        host._log(f"链路步骤记录失败: {step_label} | {exc}", event_type="WARN", source="lineage")

            final_header_for_record = dict(final_header_info or {})
            if autotune_scoring_record:
                final_header_for_record.setdefault("autotune_scoring_record", autotune_scoring_record)
            if autotune_recipe_plan:
                final_header_for_record.setdefault("autotune_recipe_plan", autotune_recipe_plan)
            host.shared_data.apply_current_data(
                final_data,
                push_history=False,
                source=run_type or "worker",
                label=outputs[-1].get("method_name", snap_label) if outputs else snap_label,
                header_info=final_header_for_record,
                trace_metadata=final_trace_metadata,
            )
            host._mark_data_changed()
            if final_display_data is not None:
                host._set_display_override(
                    final_display_data,
                    header_info=final_display_header_info,
                    trace_metadata=final_display_trace_metadata,
                )
            if is_kirchhoff and host.page_advanced.compare_var.isChecked():
                host.page_advanced.compare_var.setChecked(False)
                host._log("Kirchhoff 迁移结果已切换为单图显示。")
            host._refresh_compare_snapshots_from_state()
            host._update_empty_state_and_brief()
            host.plot_data(host.data)
            host._log(f"处理完成：共 {len(outputs)} 个步骤")
            host.page_basic.mark_params_applied(f"已应用：{snap_label}")
            host._set_runtime_summary(f"状态：已完成 · {snap_label}", "good")

            # Log processing results (for both Kirchhoff and normal cases)
            for k, item in enumerate(outputs, start=1):
                name = item.get("method_name", item.get("method_key", f"step-{k}"))
                ms = item.get("elapsed_ms")
                mapped = (item.get("meta") or {}).get("mapped_params", {})
                backend = mapped.get("execution_backend")
                fallback_reason = mapped.get("fallback_reason")
                if ms is not None:
                    suffix = f" | backend={backend}" if backend else ""
                    host._log(f"  [{k}] {name}: {ms:.1f} ms{suffix}")
                else:
                    host._log(f"  [{k}] {name}")
                if fallback_reason:
                    host._log(f"      fallback: {fallback_reason}")
                    host._append_runtime_warnings(
                        [
                            build_runtime_warning(
                                "method_fallback",
                                "方法执行触发了回退路径。",
                                method=name,
                                reason=fallback_reason,
                            )
                        ],
                        source=item.get("method_key") or name,
                    )
                host._append_runtime_warnings(
                    item.get("runtime_warnings", []),
                    source=item.get("method_key") or name,
                )
            host.status_label.setText(f"完成: {len(outputs)} 步骤")

            # 计算质量指标
            start_ts = time.perf_counter()
            metrics = compute_quality_metrics(host.data)
            metrics["time_ms"] = (time.perf_counter() - start_ts) * 1000.0
            host._set_quality_metrics(metrics)

            # 对于一键/推荐流程，自动导出对比图
            if run_type in {"pipeline", "recommended"}:
                compare_path = host._save_pipeline_comparison(outputs)
                if compare_path:
                    host._log(f"对比图已导出：{compare_path}")

            host._set_last_run_summary(
                run_type=run_type,
                label=ctx.get("run_label")
                or (outputs[-1].get("method_name") if outputs else run_type),
                steps=[
                    {
                        "method_key": item.get("method_key"),
                        "method_name": item.get("method_name"),
                        "params": item.get("params", {}),
                        "elapsed_ms": item.get("elapsed_ms"),
                        "recipe_step": item.get("recipe_step"),
                        "autotune_scoring_record": item.get("autotune_scoring_record") or autotune_scoring_record,
                    }
                    for item in outputs
                ],
                preset_key=ctx.get("preset_key"),
                profile_key=ctx.get("profile_key"),
                warnings=list(host._runtime_warnings),
                autotune_scoring_record=autotune_scoring_record,
                autotune_recipe_plan=autotune_recipe_plan,
            )
            if autotune_scoring_record:
                try:
                    from core.autotune_scoring_record import summarize_record

                    host.page_quality.append_record("[AutoTune scoring v2] " + summarize_record(autotune_scoring_record).replace("\n", " | "))
                except Exception:
                    pass

        host._cleanup_worker()

        # 恢复方法选择
        if ctx.get("restore_method_idx") is not None:
            host.page_basic.method_combo.setCurrentIndex(ctx["restore_method_idx"])
