#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Project-management page and project operation callbacks for MyGPR."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from PyQt6.QtCore import Qt, QUrl, QSettings
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QInputDialog,
    QMenu,
    QPushButton,
    QComboBox,
    QTabWidget,
    QSizePolicy,
    QTableWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from core.field_project_models import FieldLineRecord
from core.field_data_quality import DataOrientation
from core.field_project_operations import (
    delete_project_line,
    backup_project_archive,
    check_project_source_files,
    create_project,
    delete_project_permanently,
    export_line_manifest_csv,
    export_project_source_manifest_csv,
    import_line_data,
    import_trajectory_file,
    next_line_id,
    open_project,
    preview_import_source,
    remove_recent_project,
    preflight_project_delete,
    prune_missing_recent_projects,
    relink_project_line_source,
    project_dialog_filter,
    update_project_metadata,
)
from PyQt6.QtGui import QDesktopServices
from core.source_file_registry import get_line_source_record, load_source_registry, source_summary
from core.project_events import ProjectEventType
from ui.field_panels.batch_import_dialog import BatchImportProgressDialog
from ui.field_panels.project_dialogs import (
    ImportLineDialog,
    ProjectCreateDialog,
    ProjectSettingsDialog,
    QualityReportDialog,
    SourceFilesDialog,
)
from ui.field_panels.layout_metrics import layout_metrics_for
from ui.field_panels.widgets import Card, CollapsibleSidePanel, PlotCard


class ProjectPageMixin:
    def _action_new_project_dialog(self) -> None:
        default_dir = str(self.project_root.parent if self.project_root else Path.home() / "MyGPRProjects")
        dialog = ProjectCreateDialog(self, default_dir=default_dir)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        values = dialog.values()
        try:
            store = create_project(
                values["parent_dir"],
                name=values["name"],
                location=values["location"],
                operator=values["operator"],
                project_no=values.get("project_no", ""),
                device_model=values.get("device_model", ""),
                coordinate_system=values.get("coordinate_system", ""),
                vertical_datum=values.get("vertical_datum", ""),
                recent_store=self.recent_projects,
            )
            self._set_active_project_store(store, status_message=f"已新建正式项目：{store.root}")
            self._post_project_operation_refresh(kind="project_open")
            self._refresh_recent_projects_combo()
        except Exception as exc:
            self._show_operation_error("新建项目", exc)

    def _action_open_project_dialog(self) -> None:
        start_dir = str(self.project_root.parent if self.project_root else Path.home())
        project_dir = QFileDialog.getExistingDirectory(self, "打开 MyGPR 项目", start_dir)
        if not project_dir:
            return
        try:
            store = open_project(project_dir, recent_store=self.recent_projects)
            self._set_active_project_store(store, status_message=f"已打开项目：{store.root}")
            self._post_project_operation_refresh(kind="project_open")
        except Exception as exc:
            self._show_operation_error("打开项目", exc)

    def _action_project_settings_dialog(self) -> None:
        if self.project_store is None or self.project_manifest is None:
            QMessageBox.warning(self, "项目设置", "请先新建或打开 MyGPR 项目。")
            return
        dialog = ProjectSettingsDialog(self, manifest=self.project_manifest)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        values = dialog.values()
        try:
            store = update_project_metadata(
                self.project_store,
                name=values["name"],
                location=values["location"],
                operator=values["operator"],
                project_no=values["project_no"],
                device_model=values["device_model"],
                coordinate_system=values["coordinate_system"],
                vertical_datum=values["vertical_datum"],
                recent_store=self.recent_projects,
            )
            self._set_active_project_store(store, status_message="项目设置已保存到 project.json。")
            self._post_project_operation_refresh()
        except Exception as exc:
            self._show_operation_error("项目设置", exc)

    def _action_import_line_dialog(self) -> None:
        if self.project_store is None:
            QMessageBox.warning(self, "导入测线数据", "请先新建或打开 MyGPR 项目。")
            return
        path, _ = QFileDialog.getOpenFileName(self, "导入测线数据", "", project_dialog_filter())
        if not path:
            return
        default_line_id = next_line_id(self.project_store)
        preview = preview_import_source(path, line_id=default_line_id)
        dialog = ImportLineDialog(
            self,
            source_path=path,
            default_line_id=default_line_id,
            default_name=f"导入测线 {default_line_id}",
            preview_lines=preview.to_log_lines(),
            can_import=preview.can_import,
        )
        if dialog.exec() != QDialog.DialogCode.Accepted:
            if not preview.can_import:
                self._line_status_message = f"导入预检未通过：{preview.message}"
                self._refresh_project_widgets()
            return
        values = dialog.values()
        try:
            line = import_line_data(self.project_store, values["source_path"], line_id=values["line_id"], name=values["name"] or None)
            self.selected_line = line.line_id
            self.active_gpr_dataset = self.project_store.load_gpr_dataset(line.line_id)
            try:
                self.trajectory_model = self._load_line_trajectory_if_present(line.line_id)
            except Exception:
                self.trajectory_model = None
            self._line_status_message = f"已导入测线数据：{line.line_id} / {Path(path).name}；{preview.shape_text}，GPR 矩阵已归一化。"
            if getattr(self, "linkage_controller", None) is not None:
                self.linkage_controller.emit(ProjectEventType.LINE_IMPORTED, line_id=line.line_id, reason=f"{line.line_id} 测线数据已导入", refresh=False)
            self._post_project_operation_refresh(switch_to="processing_lab")
        except Exception as exc:
            self._show_operation_error("导入测线数据", exc)

    def _action_batch_import_lines_dialog(self) -> None:
        if self.project_store is None:
            QMessageBox.warning(self, "批量导入测线", "请先新建或打开 MyGPR 项目。")
            return
        paths, _ = QFileDialog.getOpenFileNames(self, "批量导入测线 CSV", "", project_dialog_filter())
        if not paths:
            return
        dialog = BatchImportProgressDialog(self, store=self.project_store, sources=[Path(path) for path in paths])
        dialog.exec()
        if dialog.error_message:
            self._show_operation_error("批量导入测线", RuntimeError(dialog.error_message))
            return
        summary = dialog.summary
        if summary is None:
            self._line_status_message = "批量导入已取消，未产生导入汇总。"
            self._refresh_project_widgets()
            return
        first_success = next((row for row in summary.results if row.success), None)
        if first_success is not None:
            self.selected_line = first_success.line_id
            try:
                self.active_gpr_dataset = self.project_store.load_gpr_dataset(first_success.line_id)
            except Exception:
                self.active_gpr_dataset = None
            try:
                self.trajectory_model = self._load_line_trajectory_if_present(first_success.line_id)
            except Exception:
                self.trajectory_model = None
        self._line_status_message = f"批量导入完成：成功 {summary.succeeded}/{summary.total}，失败 {summary.failed}。"
        detail = "\n".join(summary.to_log_lines()[:10])
        if len(summary.results) > 9:
            detail += f"\n……其余 {len(summary.results) - 9} 项请查看项目日志。"
        QMessageBox.information(self, "批量导入测线", detail)
        if first_success is not None and getattr(self, "linkage_controller", None) is not None:
            self.linkage_controller.emit(ProjectEventType.LINE_IMPORTED, line_id=first_success.line_id, reason=f"批量导入测线完成：成功 {summary.succeeded}/{summary.total}", refresh=False)
        self._post_project_operation_refresh(switch_to="data_management")

    def _action_import_trajectory_dialog(self) -> None:
        if self.project_store is None:
            QMessageBox.warning(self, "导入 RTK/IMU", "请先新建或打开 MyGPR 项目。")
            return
        path, _ = QFileDialog.getOpenFileName(self, "导入 RTK/IMU 文件", "", "轨迹/表格文件 (*.csv *.txt *.json *.log);;所有文件 (*)")
        if not path:
            return
        try:
            import_trajectory_file(self.project_store, path, line_id=self.selected_line)
            self.trajectory_model = self._load_line_trajectory_if_present(self.selected_line)
            self._line_status_message = f"已附加 RTK/IMU 文件：{Path(path).name}。"
            if getattr(self, "linkage_controller", None) is not None:
                self.linkage_controller.emit(ProjectEventType.TRAJECTORY_IMPORTED, line_id=self.selected_line, reason=f"{self.selected_line} RTK/IMU 文件已更新", refresh=False)
            self._post_project_operation_refresh()
        except Exception as exc:
            self._show_operation_error("导入 RTK/IMU", exc)

    def _action_run_quality_check(self) -> None:
        if self.project_store is None:
            QMessageBox.warning(self, "运行数据质检", "请先新建或打开 MyGPR 项目。")
            return
        try:
            reports = self.project_store.run_project_quality_check()
            self._sync_project_lines_to_ui()
            self._refresh_project_status_snapshot()
            st = self.project_status
            passed = sum(1 for report in reports if report.status == "passed")
            warnings = sum(1 for report in reports if report.status == "warning")
            failed = sum(1 for report in reports if report.status == "failed")
            self._line_status_message = f"质检完成：通过 {passed} 条，警告 {warnings} 条，失败 {failed} 条。"
            lines = [self._line_status_message]
            for report in reports[:8]:
                lines.append(f"{report.line_id}: {report.status_label}；{report.sample_count}×{report.trace_count}；{report.orientation_message}")
            if len(reports) > 8:
                lines.append(f"……其余 {len(reports) - 8} 条请查看 raw/<line_id>/*_quality_report.json。")
            QMessageBox.information(self, "运行数据质检", "\n".join(lines))
            if getattr(self, "linkage_controller", None) is not None:
                self.linkage_controller.emit(ProjectEventType.QC_UPDATED, line_id=self.selected_line, reason=f"项目质检完成：通过 {passed} 条，警告 {warnings} 条，失败 {failed} 条", refresh=False)
            self._post_project_operation_refresh(switch_to="data_management")
        except Exception as exc:
            self._show_operation_error("运行数据质检", exc)

    def _current_quality_report(self):
        if self.project_store is None:
            return None
        try:
            report = self.project_store.load_quality_report(self.selected_line)
            if report is None:
                report = self.project_store.run_line_quality_check(self.selected_line)
            return report
        except Exception:
            return None

    def _action_show_quality_detail_dialog(self) -> None:
        if self.project_store is None:
            QMessageBox.warning(self, "查看质检详情", "请先新建或打开 MyGPR 项目。")
            return
        report = self._current_quality_report()
        can_fix = bool(report and getattr(report, "orientation", "") == DataOrientation.TRANSPOSE_RISK)
        dialog = QualityReportDialog(self, line_id=self.selected_line, report=report, can_fix_orientation=can_fix)
        dialog.exec()
        if dialog.fix_requested:
            self._action_fix_bscan_orientation()

    def _action_fix_bscan_orientation(self) -> None:
        if self.project_store is None:
            QMessageBox.warning(self, "修正 B-scan 方向", "请先新建或打开 MyGPR 项目。")
            return
        report = self._current_quality_report()
        if report is None:
            QMessageBox.information(self, "修正 B-scan 方向", "当前测线没有可用 GPR 数据或质检报告。")
            return
        orientation = getattr(report, "orientation", "")
        orientation_message = getattr(report, "orientation_message", "")
        risk_note = "质检已判定为转置风险。" if orientation == DataOrientation.TRANSPOSE_RISK else "质检未判定为转置风险。"
        reply = QMessageBox.question(
            self,
            "确认修正 B-scan 方向",
            (
                f"测线：{self.selected_line}\n"
                f"当前判断：{orientation_message}\n"
                f"{risk_note}\n\n"
                "转置修正会改写标准化 B-scan 数据，并把修正前 NPZ 备份到 raw/<line_id>/orientation_fixes/。\n"
                "请确认你已经查看质检详情或图像方向确实异常。是否继续？"
            ),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        try:
            new_report = self.project_store.transpose_gpr_dataset(self.selected_line)
            self.active_gpr_dataset = self.project_store.load_gpr_dataset(self.selected_line)
            self.processed_gpr_dataset = None
            self.processing_applied = False
            try:
                self.trajectory_model = self.project_store.load_trajectory(self.selected_line)
            except Exception:
                self.trajectory_model = None
            self._line_status_message = (
                f"{self.selected_line} 已完成 B-scan 转置修正并重新质检：{new_report.status_label}；"
                f"{new_report.sample_count}×{new_report.trace_count}。"
            )
            QMessageBox.information(self, "修正 B-scan 方向", self._line_status_message)
            if getattr(self, "linkage_controller", None) is not None:
                self.linkage_controller.emit(ProjectEventType.BSCAN_ORIENTATION_FIXED, line_id=self.selected_line, reason=f"{self.selected_line} B-scan 方向已修正", refresh=False)
            self._post_project_operation_refresh(switch_to="data_management")
        except Exception as exc:
            self._show_operation_error("修正 B-scan 方向", exc)

    def _action_export_line_manifest(self) -> None:
        if self.project_store is None:
            QMessageBox.warning(self, "导出测线清单", "请先新建或打开 MyGPR 项目。")
            return
        default_path = str(self.project_store.root / "reports" / "line_manifest.csv")
        path, _ = QFileDialog.getSaveFileName(self, "导出测线清单", default_path, "CSV 文件 (*.csv);;所有文件 (*)")
        if not path:
            return
        try:
            out = export_line_manifest_csv(self.project_store, path)
            self._line_status_message = f"已导出测线清单：{out}。"
            QMessageBox.information(self, "导出测线清单", f"已导出：\n{out}")
            self._post_project_operation_refresh(switch_to="data_management")
        except Exception as exc:
            self._show_operation_error("导出测线清单", exc)

    def _action_backup_project(self) -> None:
        if self.project_store is None or self.project_root is None:
            QMessageBox.warning(self, "项目备份", "请先新建或打开 MyGPR 项目。")
            return
        try:
            result = backup_project_archive(self.project_store)
            self._line_status_message = f"项目备份完成：{result.archive_path}，{result.file_count} 个文件，{result.size_mb:.3f} MB。"
            QMessageBox.information(self, "项目备份", self._line_status_message)
            self._post_project_operation_refresh(switch_to="data_management")
        except Exception as exc:
            self._show_operation_error("项目备份", exc)

    def _action_check_source_files(self) -> None:
        if self.project_store is None:
            QMessageBox.warning(self, "检查源文件", "请先新建或打开 MyGPR 项目。")
            return
        try:
            records = check_project_source_files(self.project_store)
            summary = source_summary(records)
            self._line_status_message = (
                f"源文件检查完成：正常 {summary.get('available', 0)}，"
                f"缺失 {summary.get('missing', 0)}，已变更 {summary.get('modified', 0)}。"
            )
            if getattr(self, "linkage_controller", None) is not None:
                self.linkage_controller.emit(ProjectEventType.LINE_SOURCE_STATUS_CHECKED, line_id=self.selected_line, reason="源文件状态检查完成", refresh=False)
            self._post_project_operation_refresh(switch_to="data_management")
            dialog = SourceFilesDialog(self, records=records, summary_text=self._line_status_message)
            dialog.exec()
        except Exception as exc:
            self._show_operation_error("检查源文件", exc)

    def _action_open_data_ops_center(self) -> None:
        if self.project_store is None:
            QMessageBox.warning(self, "项目数据运维中心", "请先新建或打开 MyGPR 项目。")
            return
        records = load_source_registry(self.project_store)
        summary = source_summary(records)
        text = (
            f"已记录源文件 {summary.get('total', 0)} 个；"
            f"正常 {summary.get('available', 0)}，缺失 {summary.get('missing', 0)}，"
            f"已变更 {summary.get('modified', 0)}。"
        )
        dialog = SourceFilesDialog(self, records=records, summary_text=text)
        dialog.exec()

    def _action_relocate_current_source(self) -> None:
        if self.project_store is None:
            QMessageBox.warning(self, "重新定位源文件", "请先新建或打开 MyGPR 项目。")
            return
        line = self._selected_line_record()
        line_id = str(line.get("id", self.selected_line))
        path, _ = QFileDialog.getOpenFileName(self, f"重新定位 {line_id} 的原始源文件", "", project_dialog_filter())
        if not path:
            return
        try:
            record = relink_project_line_source(self.project_store, line_id, path)
            self._line_status_message = f"已重新定位 {line_id} 源文件：{record.source_filename}；状态：{record.status_label}。"
            QMessageBox.information(self, "重新定位源文件", self._line_status_message)
            if getattr(self, "linkage_controller", None) is not None:
                self.linkage_controller.emit(ProjectEventType.LINE_SOURCE_RELINKED, line_id=line_id, reason=f"{line_id} 源文件已重新定位", refresh=False)
            self._post_project_operation_refresh(switch_to="data_management")
        except Exception as exc:
            reply = QMessageBox.question(
                self,
                "源文件不匹配",
                f"重新定位文件与原记录不匹配：\n{exc}\n\n是否仍强制绑定该文件？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return
            try:
                record = relink_project_line_source(self.project_store, line_id, path, allow_mismatch=True)
                self._line_status_message = f"已强制绑定 {line_id} 源文件：{record.source_filename}；请复核数据一致性。"
                QMessageBox.warning(self, "重新定位源文件", self._line_status_message)
                if getattr(self, "linkage_controller", None) is not None:
                    self.linkage_controller.emit(ProjectEventType.LINE_SOURCE_RELINKED, line_id=line_id, reason=f"{line_id} 源文件已强制重新绑定", refresh=False)
                self._post_project_operation_refresh(switch_to="data_management")
            except Exception as final_exc:
                self._show_operation_error("重新定位源文件", final_exc)

    def _action_export_source_manifest(self) -> None:
        if self.project_store is None:
            QMessageBox.warning(self, "导出来源清单", "请先新建或打开 MyGPR 项目。")
            return
        default_path = str(self.project_store.root / "reports" / "source_file_manifest.csv")
        path, _ = QFileDialog.getSaveFileName(self, "导出来源清单", default_path, "CSV 文件 (*.csv);;所有文件 (*)")
        if not path:
            return
        try:
            out = export_project_source_manifest_csv(self.project_store, path)
            self._line_status_message = f"已导出来源清单：{out}。"
            QMessageBox.information(self, "导出来源清单", f"已导出：\n{out}")
            self._post_project_operation_refresh(switch_to="data_management")
        except Exception as exc:
            self._show_operation_error("导出来源清单", exc)

    def _action_open_current_source_dir(self) -> None:
        if self.project_store is None:
            QMessageBox.warning(self, "打开源文件目录", "请先新建或打开 MyGPR 项目。")
            return
        line = self._selected_line_record()
        line_id = str(line.get("id", self.selected_line))
        record = get_line_source_record(self.project_store, line_id)
        if record is None or not record.source_path:
            QMessageBox.information(self, "打开源文件目录", f"{line_id} 尚未记录外部源文件路径。")
            return
        source_path = Path(record.source_path).expanduser()
        folder = source_path.parent if source_path.parent.exists() else source_path
        if not folder.exists():
            QMessageBox.warning(self, "打开源文件目录", f"源文件目录不存在：\n{folder}")
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder)))

    def _action_delete_current_line(self) -> None:
        if self.project_store is None:
            QMessageBox.warning(self, "删除测线", "请先新建或打开 MyGPR 项目。")
            return
        line = self._selected_line_record()
        line_id = str(line.get("id", self.selected_line))
        line_name = str(line.get("name", ""))
        reply = QMessageBox.question(
            self,
            "确认删除测线",
            (
                f"测线：{line_id} {line_name}\n\n"
                "该操作会从 project.json 中移除该测线，并直接删除项目文件夹内 raw / processed / targets / spatial 中的关联文件。\n"
                "不会删除项目目录之外的原始导入来源文件，但已有报告会标记为需重新生成。是否继续？"
            ),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        try:
            result = delete_project_line(self.project_store, line_id)
            self._line_status_message = (
                f"已删除测线 {result.line_id} 的项目内关联文件 {len(result.deleted_paths)} 项；"
                f"剩余 {result.remaining_line_count} 条测线。"
            )
            self._sync_project_lines_to_ui()
            if getattr(self, "linkage_controller", None) is not None:
                self.linkage_controller.emit(ProjectEventType.LINE_DELETED, line_id=result.line_id, reason=f"测线 {result.line_id} 已删除", refresh=False)
            QMessageBox.information(self, "删除测线", self._line_status_message)
            self._post_project_operation_refresh(switch_to="data_management")
        except Exception as exc:
            self._show_operation_error("删除测线", exc)

    def _action_remove_recent_project(self) -> None:
        if self.recent_project_combo is None:
            return
        project_path = self.recent_project_combo.currentData()
        if not project_path:
            QMessageBox.information(self, "移除最近项目", "当前没有可移除的最近项目。")
            return
        removed = remove_recent_project(project_path, recent_store=self.recent_projects)
        self._refresh_recent_projects_combo()
        self._refresh_project_selector_combo()
        self._line_status_message = f"已从最近项目列表移除 {removed} 条记录。"
        QMessageBox.information(self, "移除最近项目", self._line_status_message)

    def _action_delete_current_project(self) -> None:
        if self.project_store is None or self.project_root is None:
            QMessageBox.warning(self, "删除项目", "请先新建或打开 MyGPR 项目。")
            return
        project_name = str(getattr(self.project_manifest, "name", self.project_root.name))
        try:
            preview = preflight_project_delete(self.project_store, recent_store=self.recent_projects)
        except Exception as exc:
            self._show_operation_error("删除项目预检", exc)
            return
        text, ok = QInputDialog.getText(
            self,
            "确认删除项目",
            (
                "删除前预检：\n"
                + "\n".join(preview.to_lines())
                + "\n\n该操作只删除 MyGPR 项目文件夹内的数据，不会删除项目目录外的原始来源文件。"
                + "\n删除后会清理最近项目并回到未打开项目状态。"
                + "\n\n请输入项目名称以确认："
            ),
        )
        if not ok:
            return
        if text.strip() != project_name:
            QMessageBox.warning(self, "删除项目", "输入的项目名称不匹配，已取消删除。")
            return
        try:
            result = delete_project_permanently(self.project_store, recent_store=self.recent_projects)
            stale = prune_missing_recent_projects(recent_store=self.recent_projects)
            self.project_store = None
            self.project_manifest = None
            self.project_root = None
            self.line_records = []
            self.selected_line = "L01"
            self.active_gpr_dataset = None
            self.processed_gpr_dataset = None
            self.trajectory_model = None
            self.targets = []
            self.current_target_index = 0
            self.current_target_source_id = "L01_raw"
            self._line_status_message = f"项目文件夹已删除：{result.deleted_path}；已清理最近项目 {result.removed_recent_count + stale} 条。"
            self._refresh_project_status_snapshot()
            self._refresh_project_selector_combo()
            self._refresh_recent_projects_combo()
            self._rebuild_workspace_pages()
            self._refresh_project_widgets()
            self._refresh_processing_preview()
            self._refresh_target_source_options()
            self._refresh_target_widgets()
            self.switch_workspace("data_management")
            QMessageBox.information(
                self,
                "删除项目",
                f"项目文件夹已删除：\n{result.deleted_path}\n\n原始来源文件若在项目目录之外，不会被删除。",
            )
        except Exception as exc:
            self._show_operation_error("删除项目", exc)

    def _refresh_recent_projects_combo(self) -> None:
        if self.recent_project_combo is None:
            return
        self.recent_project_combo.clear()
        try:
            self.recent_projects.prune_missing()
        except Exception:
            pass
        records = self.recent_projects.load()
        if not records:
            self.recent_project_combo.addItem("暂无最近项目", "")
            self.recent_project_combo.setEnabled(False)
            return
        self.recent_project_combo.setEnabled(True)
        for record in records[:8]:
            label = f"{record.name}  ·  {Path(record.path).name}"
            self.recent_project_combo.addItem(label, record.path)

    def _action_open_recent_project(self) -> None:
        if self.recent_project_combo is None:
            return
        project_path = self.recent_project_combo.currentData()
        if not project_path:
            QMessageBox.information(self, "最近项目", "暂无可打开的最近项目。")
            return
        try:
            store = open_project(project_path, recent_store=self.recent_projects)
            self._set_active_project_store(store, status_message=f"已打开最近项目：{store.root}")
            self._post_project_operation_refresh(kind="project_open")
            self._refresh_recent_projects_combo()
        except Exception as exc:
            try:
                self.recent_projects.prune_missing()
                self._refresh_recent_projects_combo()
                self._refresh_project_selector_combo()
            except Exception:
                pass
            self._show_operation_error("打开最近项目", exc)

    def _page_project_management(self) -> QWidget:
        widget = QWidget()
        v = QVBoxLayout(widget)
        v.setContentsMargins(0, 0, 0, 0)
        lm = layout_metrics_for(self)
        v.setSpacing(lm.spacing)

        metrics = QHBoxLayout()
        metrics.setSpacing(lm.spacing)
        st = self.project_status
        raw_value, raw_suffix = self._format_mb(st.raw_size_mb)
        for card in [
            self._metric_card(self.project_metric_cards, "lines", "📊", "测线总数", str(st.line_count), "条"),
            self._metric_card(self.project_metric_cards, "raw", "💾", "已导入数据", raw_value, raw_suffix),
            self._metric_card(self.project_metric_cards, "trajectory", "📍", "辅助定位文件", str(st.trajectory_file_count), "个"),
            self._metric_card(self.project_metric_cards, "reports", "▤", "报告状态", st.report_status, "", f"交付文件 {st.report_file_count} 个"),
            self._metric_card(self.project_metric_cards, "status", "◈", "项目状态", st.data_health_label, "", f"最后更新：{st.latest_update}"),
        ]:
            metrics.addWidget(card)
        v.addLayout(metrics)

        # Status strip -- compact project context bar (EKKO_Project pattern)
        status_parts = []
        if st.line_count > 0:
            status_parts.append(f"共 {st.line_count} 条测线")
        if st.processed_line_count > 0:
            status_parts.append(f"{st.processed_line_count} 条已处理")
        if st.trajectory_file_count > 0:
            status_parts.append(f"{st.trajectory_file_count} 条已定位")
        status_text = " ｜ ".join(status_parts) if status_parts else "暂无数据"
        status_label = QLabel(f"项目概况：{status_text}")
        status_label.setObjectName("activityDesc")
        status_label.setWordWrap(False)
        status_label.setMaximumHeight(16)
        v.addWidget(status_label)

        mid = QHBoxLayout()
        mid.setSpacing(lm.spacing)
        mid.addWidget(self._project_summary_card(), 3)
        mid.addWidget(self._line_list_card(), 6)
        import_panel = CollapsibleSidePanel(
            title="项目操作",
            content=self._import_qc_card(),
            expanded_width=lm.project_ops_max_w,
            collapsed_width=34,
        )
        import_panel.setProperty("layoutKey", "projectOpsSidePanel")
        mid.addWidget(import_panel, 0)
        v.addLayout(mid, 4)

        bottom = QHBoxLayout()
        bottom.setSpacing(lm.spacing)
        bottom.addWidget(self._task_tabs_card(), 7)
        bottom.addWidget(self._quick_preview_card(), 3)
        v.addLayout(bottom, 1)
        return widget

    def _project_summary_card(self) -> Card:
        card = Card(title="项目概览")
        card.setProperty("layoutKey", "projectSummaryCard")
        card.setMinimumWidth(0)
        card.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        lm = layout_metrics_for(self)

        manifest = self.project_manifest
        project_name = getattr(manifest, "name", "未打开项目")
        name = QLabel(project_name)
        name.setObjectName("sectionTitle")
        name.setToolTip(project_name)
        card.layout.addWidget(name)

        def compact_text(value: object, limit: int = 30) -> str:
            text = str(value or "--")
            if len(text) <= limit:
                return text
            keep_left = max(8, limit // 2 - 2)
            keep_right = max(8, limit - keep_left - 1)
            return f"{text[:keep_left]}…{text[-keep_right:]}"

        project_path = str(self.project_root) if self.project_root else "--"
        raw_size = f"{self.project_store.total_raw_size_mb():.2f} MB" if self.project_store is not None else "0.00 MB"
        infos = [
            ("编号", getattr(manifest, "project_no", "--"), 28),
            ("创建", getattr(manifest, "created_at", "--"), 24),
            ("测区", getattr(manifest, "location", "--"), 26),
            ("设备", getattr(manifest, "device_model", "--"), 28),
            ("坐标", getattr(manifest, "coordinate_system", "--"), 28),
            ("高程", getattr(manifest, "vertical_datum", "--"), 28),
            ("大小", raw_size, 20),
            ("路径", project_path, 28),
        ]
        for key, value, limit in infos:
            row = QHBoxLayout()
            row.setSpacing(lm.spacing)
            key_label = QLabel(key)
            key_label.setObjectName("keyLabel")
            key_label.setFixedWidth(38 if lm.compact else 46)
            value_label = QLabel(compact_text(value, limit))
            value_label.setObjectName("valueLabel")
            value_label.setToolTip(str(value or "--"))
            value_label.setWordWrap(False)
            row.addWidget(key_label)
            row.addWidget(value_label, 1)
            card.layout.addLayout(row)

        map_preview = PlotCard(None, height=max(50, min(lm.project_summary_map_h, 95)))
        map_preview.setProperty("layoutKey", "projectSummaryMapCard")
        map_preview.canvas.setObjectName("projectSummaryMapCanvas")
        map_preview.layout.setContentsMargins(0, 0, 0, 0)
        if self.line_records:
            self._draw_current_line_strip(map_preview.canvas)
        else:
            placeholder = QLabel("导入测线后显示轨迹地图")
            placeholder.setObjectName("hintText")
            placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
            map_preview.layout.addWidget(placeholder)
        card.layout.addWidget(map_preview)
        return card

    def _line_list_card(self) -> Card:
        card = Card(title="测线清单")
        card.setProperty("layoutKey", "projectLineListCard")
        card.setMinimumWidth(0)
        card.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        lm = layout_metrics_for(self)
        bar = QHBoxLayout()
        bar.addStretch(1)
        for text in ["＋ 新建", "🗑 删除", "⇩ 导出", "↻ 刷新"]:
            btn = QPushButton(text)
            btn.setObjectName("smallButton")
            if "刷新" in text:
                btn.clicked.connect(self._refresh_project_widgets)
            elif "新建" in text:
                btn.clicked.connect(self._add_preview_line)
            elif "删除" in text:
                btn.clicked.connect(self._action_delete_current_line)
            elif "导出" in text:
                btn.clicked.connect(self._action_export_line_manifest)
            bar.addWidget(btn)
        card.layout.insertLayout(1, bar)

        if not self.line_records:
            empty = QLabel("暂无测线数据。\n\n请点击「＋ 新建测线」创建测线，\n或使用右侧面板「导入测线」导入原始数据文件。")
            empty.setObjectName("hintText")
            empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
            empty.setWordWrap(True)
            card.layout.addWidget(empty, 1)
            foot = QHBoxLayout()
            foot.addStretch(1)
            self.line_status_label = QLabel(self._line_status_message)
            self.line_status_label.setObjectName("activityDesc")
            foot.addWidget(self.line_status_label)
            foot.addSpacing(14)
            foot.addWidget(QLabel("共 0 条"))
            card.layout.addLayout(foot)
            return card

        headers = ["测线", "源文件", "长度", "质量", "定位", "处理", "时间"] if getattr(self, "compact_mode", True) else ["测线名称", "源文件", "长度 (m)", "数据质量", "定位状态", "处理状态", "最近更新时间", "操作"]
        row_count = max(len(self.line_records), 2) if len(self.line_records) <= 3 else max(len(self.line_records), 4)
        table = self._table(headers, row_count)
        table.setMinimumHeight(lm.project_table_min_h)
        table.setObjectName("projectLineTable")
        # Allow multi-select and column resizing for dense line tables.
        table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        header = table.horizontalHeader()
        for i in range(table.columnCount()):
            header.setSectionResizeMode(i, QHeaderView.ResizeMode.Interactive)
        header.setStretchLastSection(True)
        self.line_table = table
        self._fill_table(table, self._line_rows(), highlight_row=self._selected_line_row(), sort_column=0)
        self._restore_line_table_column_widths(table)
        header.sectionResized.connect(lambda _l, _o, _n: self._save_line_table_column_widths())

        def _line_table_key(event) -> None:
            if event.key() == Qt.Key.Key_Delete:
                self._action_delete_selected_lines()
            else:
                QTableWidget.keyPressEvent(table, event)

        table.keyPressEvent = _line_table_key
        table.cellClicked.connect(self._select_line_from_table)
        card.layout.addWidget(table, 1)
        card.layout.addWidget(self._selected_line_detail_strip(), 0)
        foot = QHBoxLayout()
        foot.addWidget(QLabel("‹   1    2   ›"))
        foot.addStretch(1)
        self.line_status_label = QLabel(self._line_status_message)
        self.line_status_label.setObjectName("activityDesc")
        foot.addWidget(self.line_status_label)
        foot.addSpacing(14)
        foot.addWidget(QLabel(f"共 {len(self.line_records)} 条"))
        card.layout.addLayout(foot)
        return card

    def _selected_line_detail_strip(self) -> QFrame:
        """Compact selected-line digest shown below the line list.

        The line table only has a handful of rows in typical field projects.
        This strip turns the remaining space into useful context instead of a
        large blank table body.
        """
        lm = layout_metrics_for(self)
        line = self._selected_line_record()
        frame = QFrame()
        frame.setObjectName("lineDetailPanel")
        frame.setMinimumHeight(36 if lm.compact else 42)
        frame.setMaximumHeight(42 if lm.compact else 50)
        row = QHBoxLayout(frame)
        row.setContentsMargins(6, 4, 6, 4)
        row.setSpacing(8)

        title_col = QVBoxLayout()
        title_col.setSpacing(2)
        title = QLabel(f"当前测线  {line.get('id', '--')}  {line.get('name', '暂无测线')}")
        title.setObjectName("detailTitle")
        title.setToolTip(title.text())
        subtitle = QLabel(f"源文件：{line.get('source', '未记录')}｜最近更新：{line.get('updated', '--')}")
        subtitle.setObjectName("detailSubtitle")
        subtitle.setToolTip(subtitle.text())
        title_col.addWidget(title)
        title_col.addWidget(subtitle)
        row.addLayout(title_col, 1)

        def stat(label: str, value: object, suffix: str = "") -> QWidget:
            box = QFrame()
            box.setObjectName("miniStatBox")
            box_layout = QVBoxLayout(box)
            box_layout.setContentsMargins(6, 3, 6, 3)
            box_layout.setSpacing(0)
            v = QLabel(f"{value}{suffix}")
            v.setObjectName("miniStatValue")
            k = QLabel(label)
            k.setObjectName("miniStatLabel")
            box_layout.addWidget(v)
            box_layout.addWidget(k)
            return box

        row.addWidget(stat("长度", f"{float(line.get('length', 0.0)):.1f}", " m"), 0)
        row.addWidget(stat("定位", str(line.get('rtk', '--')).replace('● ', '')), 0)
        row.addWidget(stat("处理", str(line.get('status', '--')).replace('● ', '')), 0)
        return frame

    def _add_preview_line(self) -> None:
        next_id = f"L{len(self.line_records) + 1:02d}"
        if self.project_store is not None:
            self.project_store.upsert_line(FieldLineRecord(next_id, "现场新增测线", 0.00, "--", "未定位", "未处理", "刚刚"))
            self._sync_project_lines_to_ui()
        else:
            self.line_records.append({"id": next_id, "name": "现场新增测线", "length": 0.00, "quality": "--", "rtk": "● 未定位", "status": "● 未处理", "updated": "刚刚", "targets": 0})
        self.selected_line = next_id
        self._line_status_message = f"已新建 {next_id}，project.json 已同步，可继续导入测线数据。"
        self._refresh_project_widgets()

    def _make_action_tile(self, icon: str, label: str, callback: Callable[[], None]) -> QPushButton:
        btn = QPushButton(f"{icon}\n{label}")
        btn.setObjectName("actionTileButton")
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        btn.clicked.connect(callback)
        return btn

    def _make_more_operations_button(self) -> QToolButton:
        btn = QToolButton()
        btn.setObjectName("actionTileButton")
        btn.setText("⋯\n更多操作")
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        btn.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        menu = QMenu(btn)
        menu_items: list[tuple[str, Callable[[], None] | None]] = [
            ("项目设置", self._action_project_settings_dialog),
            ("项目备份", self._action_backup_project),
            ("导出测线清单", self._action_export_line_manifest),
            ("导出来源清单", self._action_export_source_manifest),
            ("检查源文件", self._action_check_source_files),
            ("", None),
            ("删除项目…", self._action_delete_current_project),
        ]
        for label, callback in menu_items:
            if not label:
                menu.addSeparator()
                continue
            action = menu.addAction(label)
            if callback is not None:
                action.triggered.connect(callback)
        btn.setMenu(menu)
        return btn

    def _import_qc_card(self) -> Card:
        lm = layout_metrics_for(self)
        card = Card()
        card.setProperty("layoutKey", "projectImportQcCard")
        card.setMinimumWidth(lm.project_ops_min_w)
        card.setMaximumWidth(lm.project_ops_max_w)
        card.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)

        section_import = QLabel("数据导入")
        section_import.setObjectName("cardTitle")
        card.layout.addWidget(section_import)
        for icon, label, callback in [
            ("⇩", "导入测线", self._action_import_line_dialog),
            ("⇪", "批量导入", self._action_batch_import_lines_dialog),
            ("≋", "导入RTK/IMU", self._action_import_trajectory_dialog),
        ]:
            btn = self._make_action_tile(icon, label, callback)
            btn.setMinimumHeight(28)
            card.layout.addWidget(btn)

        section_maint = QLabel("项目维护")
        section_maint.setObjectName("cardTitle")
        card.layout.addSpacing(1)
        sep = QFrame()
        sep.setObjectName("separatorLine")
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setFixedHeight(1)
        card.layout.addWidget(sep)
        card.layout.addSpacing(1)
        card.layout.addWidget(section_maint)
        for widget in [
            self._make_action_tile("☑", "运行质检", self._action_run_quality_check),
            self._make_action_tile("◎", "数据运维", self._action_open_data_ops_center),
            self._make_more_operations_button(),
        ]:
            widget.setMinimumHeight(28)
            card.layout.addWidget(widget)
        return card

    def _task_tabs_card(self) -> Card:
        card = Card(title="任务与日志")
        card.setProperty("layoutKey", "projectTaskTabsCard")
        bottom_h = layout_metrics_for(self).project_bottom_max_h
        card.setMinimumHeight(bottom_h - 20)
        tabs = QTabWidget()
        tabs.setObjectName("innerTabs")
        self.project_task_tabs = tabs
        table_specs = [
            ("任务", ["任务名称", "类型", "状态", "进度", "开始时间", "结束时间", "操作"], self.project_status.task_rows),
            ("检查提示", ["检查内容", "说明", "状态"], [(title, desc, count) for _icon, title, desc, count in self.project_status.attention_items]),
            ("交付文件", ["文件名称", "类型", "大小", "更新时间", "状态", "操作"], self.project_status.delivery_rows),
            ("日志", ["类型", "事件", "说明", "时间"], self.project_status.activity_rows),
        ]
        empty_hints = {
            "任务": "📋 导入测线数据后，任务列表会自动更新。",
            "检查提示": "🔍 运行质检后，检查结果会显示在此处。",
            "交付文件": "📄 生成成果报告后，交付文件会列在这里。",
            "日志": "📝 操作日志会在此处记录。",
        }
        for name, headers, rows in table_specs:
            if not rows:
                hint = QLabel(empty_hints.get(name, "暂无数据"))
                hint.setObjectName("hintText")
                hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
                hint.setWordWrap(True)
                tabs.addTab(hint, name)
            else:
                table = self._table(headers, max(len(rows), 1))
                self._fill_table(table, list(rows))
                tabs.addTab(table, name)
        card.layout.addWidget(tabs)
        return card

    def _quick_preview_card(self) -> Card:
        line = self._selected_line_record()
        has_lines = bool(self.line_records)
        line_id = str(line.get("id", self.selected_line))
        line_name = str(line.get("name", "当前测线"))
        card = Card(title=f"快速预览（{line_id} {line_name}）")
        card.setProperty("layoutKey", "projectQuickPreviewCard")
        lm = layout_metrics_for(self)
        card.setMinimumHeight(lm.project_bottom_max_h)
        if not has_lines:
            empty_icon = QLabel("📊")
            empty_icon.setObjectName("sectionTitle")
            empty_icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
            empty_icon.setFixedHeight(22)
            card.layout.addWidget(empty_icon)
            hint = QLabel("暂无预览数据")
            hint.setObjectName("sectionTitle")
            hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
            card.layout.addWidget(hint)
            desc = QLabel("导入测线数据后，此处显示 B-scan 预览和轨迹地图。\n可在右侧面板点击「导入测线」开始。")
            desc.setObjectName("hintText")
            desc.setAlignment(Qt.AlignmentFlag.AlignCenter)
            desc.setWordWrap(True)
            card.layout.addWidget(desc)
            return card
        row = QHBoxLayout()
        row.setSpacing(lm.spacing)
        plot = PlotCard(None, height=lm.project_preview_bscan_h)
        plot.setProperty("layoutKey", "projectQuickPreviewBscanCard")
        plot.canvas.setObjectName("projectQuickPreviewBscanCanvas")
        plot.layout.setContentsMargins(0, 0, 0, 0)
        self._draw_current_line_bscan(plot.canvas, title="")
        row.addWidget(plot, 5)
        right = QVBoxLayout()
        right.setSpacing(lm.spacing)
        mini_map = PlotCard("测线轨迹", height=lm.project_preview_map_h)
        mini_map.setProperty("layoutKey", "projectQuickPreviewMapCard")
        mini_map.canvas.setObjectName("projectQuickPreviewMapCanvas")
        mini_map.layout.setContentsMargins(0, 0, 0, 0)
        self._draw_current_line_strip(mini_map.canvas)
        right.addWidget(mini_map)
        info_grid = QGridLayout()
        info_grid.setSpacing(2)
        for r, (lbl, key) in enumerate([
            ("长度", "length"), ("定位", "rtk"), ("处理", "status"),
        ]):
            k = QLabel(lbl)
            k.setObjectName("keyLabel")
            if key == "length":
                v = QLabel(f"{float(line.get(key, 0.0)):.2f} m")
            else:
                v = QLabel(str(line.get(key, "--")))
            v.setObjectName("valueLabel")
            info_grid.addWidget(k, r, 0)
            info_grid.addWidget(v, r, 1)
        right.addLayout(info_grid)
        row.addLayout(right, 2)
        card.layout.addLayout(row)
        return card


    def select_line(self, line_id: str) -> None:
        """Programmatically select a line by id and refresh all dependent UI."""
        for idx, line in enumerate(self.line_records):
            if line.get("id") == line_id:
                self._select_line_from_table(idx, 0)
                return

    def _action_delete_selected_lines(self) -> None:
        """Batch-delete all selected rows from the project line table."""
        table = self.line_table
        if table is None or self.project_store is None:
            return
        rows = sorted({idx.row() for idx in table.selectedIndexes() if idx.row() < len(self.line_records)}, reverse=True)
        if not rows:
            return
        line_ids = [self.line_records[r]["id"] for r in rows]
        names = [f"{self.line_records[r]['id']} {self.line_records[r].get('name', '')}" for r in rows]
        detail = "\n".join(names[:8])
        if len(names) > 8:
            detail += f"\n……其余 {len(names) - 8} 条"
        reply = QMessageBox.question(
            self,
            "确认批量删除测线",
            f"即将删除 {len(line_ids)} 条测线：\n{detail}\n\n"
            "此操作会删除项目内关联文件，不会删除项目目录外的原始来源文件。是否继续？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        failed: list[tuple[str, str]] = []
        for line_id in line_ids:
            try:
                delete_project_line(self.project_store, line_id, reason="batch-delete")
            except Exception as exc:
                failed.append((line_id, str(exc)))
        self._sync_project_lines_to_ui()
        self._clear_line_dependent_processing_state()
        self._refresh_project_widgets()
        self._refresh_processing_preview()
        if failed:
            QMessageBox.warning(
                self,
                "批量删除测线",
                f"成功 {len(line_ids) - len(failed)} 条，失败 {len(failed)} 条：\n"
                + "\n".join(f"{lid}: {msg}" for lid, msg in failed[:8]),
            )
        else:
            self._line_status_message = f"已批量删除 {len(line_ids)} 条测线。"
            self._refresh_project_widgets()

    def _restore_line_table_column_widths(self, table: QTableWidget) -> None:
        settings = QSettings("MyGPR", "MyGPR")
        widths = settings.value("ui/project_line_table_column_widths")
        if not isinstance(widths, list):
            return
        header = table.horizontalHeader()
        for i, w in enumerate(widths):
            if i >= table.columnCount():
                break
            try:
                w_int = int(w)
            except Exception:
                continue
            if w_int > 0:
                header.resizeSection(i, w_int)

    def _save_line_table_column_widths(self) -> None:
        table = self.line_table
        if table is None:
            return
        settings = QSettings("MyGPR", "MyGPR")
        widths = [table.columnWidth(i) for i in range(table.columnCount())]
        settings.setValue("ui/project_line_table_column_widths", widths)


__all__ = ["ProjectPageMixin"]
