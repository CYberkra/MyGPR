#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Project operation dialogs for the MyGPR field workbench."""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QTextEdit,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QVBoxLayout,
    QWidget,
)


class ProjectCreateDialog(QDialog):
    """Compact formal project wizard used by the field workbench."""

    def __init__(self, parent: QWidget | None = None, *, default_dir: str = "") -> None:
        super().__init__(parent)
        self.setWindowTitle("新建 MyGPR 项目")
        self.setMinimumWidth(560)
        layout = QVBoxLayout(self)
        form = QFormLayout()
        self.name_edit = QLineEdit("新建 MyGPR 项目")
        self.project_no_edit = QLineEdit("")
        self.location_edit = QLineEdit("未填写")
        self.operator_edit = QLineEdit("操作员")
        self.device_model_edit = QLineEdit("IDS Stream DP Pro + CX-RTK2")
        self.coordinate_system_edit = QLineEdit("CGCS2000 / 3-degree GK")
        self.vertical_datum_edit = QLineEdit("1985 国家高程基准")
        self.parent_edit = QLineEdit(default_dir or str(Path.home()))
        browse = QPushButton("浏览…")
        browse.clicked.connect(self._browse_parent)
        row = QHBoxLayout()
        row.addWidget(self.parent_edit, 1)
        row.addWidget(browse)
        holder = QWidget()
        holder.setLayout(row)
        form.addRow("项目名称 *", self.name_edit)
        form.addRow("项目编号", self.project_no_edit)
        form.addRow("测区位置", self.location_edit)
        form.addRow("项目操作员", self.operator_edit)
        form.addRow("设备型号", self.device_model_edit)
        form.addRow("坐标系统", self.coordinate_system_edit)
        form.addRow("垂向基准", self.vertical_datum_edit)
        form.addRow("保存目录 *", holder)
        layout.addLayout(form)
        tip = QLabel("说明：将创建空正式项目，并生成 project.json、raw、processed、targets、spatial、reports、logs。")
        tip.setWordWrap(True)
        tip.setObjectName("activityDesc")
        layout.addWidget(tip)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _browse_parent(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "选择新项目保存目录", self.parent_edit.text() or str(Path.home()))
        if path:
            self.parent_edit.setText(path)

    def values(self) -> dict[str, str]:
        return {
            "name": self.name_edit.text().strip(),
            "project_no": self.project_no_edit.text().strip(),
            "location": self.location_edit.text().strip(),
            "operator": self.operator_edit.text().strip(),
            "device_model": self.device_model_edit.text().strip(),
            "coordinate_system": self.coordinate_system_edit.text().strip(),
            "vertical_datum": self.vertical_datum_edit.text().strip(),
            "parent_dir": self.parent_edit.text().strip(),
        }


class ProjectSettingsDialog(QDialog):
    """Edit project metadata persisted in project.json."""

    def __init__(self, parent: QWidget | None = None, *, manifest: object) -> None:
        super().__init__(parent)
        self.setWindowTitle("项目设置")
        self.setMinimumWidth(560)
        layout = QVBoxLayout(self)
        form = QFormLayout()
        self.name_edit = QLineEdit(str(getattr(manifest, "name", "")))
        self.project_no_edit = QLineEdit(str(getattr(manifest, "project_no", "")))
        self.location_edit = QLineEdit(str(getattr(manifest, "location", "")))
        self.operator_edit = QLineEdit(str(getattr(manifest, "operator", "")))
        self.device_model_edit = QLineEdit(str(getattr(manifest, "device_model", "")))
        self.coordinate_system_edit = QLineEdit(str(getattr(manifest, "coordinate_system", "")))
        self.vertical_datum_edit = QLineEdit(str(getattr(manifest, "vertical_datum", "")))
        form.addRow("项目名称 *", self.name_edit)
        form.addRow("项目编号", self.project_no_edit)
        form.addRow("测区位置", self.location_edit)
        form.addRow("项目操作员", self.operator_edit)
        form.addRow("设备型号", self.device_model_edit)
        form.addRow("坐标系统", self.coordinate_system_edit)
        form.addRow("垂向基准", self.vertical_datum_edit)
        layout.addLayout(form)
        tip = QLabel("说明：项目设置会写入 project.json，并用于首页、项目管理页、空间成果和报告交付的元数据展示。")
        tip.setWordWrap(True)
        tip.setObjectName("activityDesc")
        layout.addWidget(tip)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def values(self) -> dict[str, str]:
        return {
            "name": self.name_edit.text().strip(),
            "project_no": self.project_no_edit.text().strip(),
            "location": self.location_edit.text().strip(),
            "operator": self.operator_edit.text().strip(),
            "device_model": self.device_model_edit.text().strip(),
            "coordinate_system": self.coordinate_system_edit.text().strip(),
            "vertical_datum": self.vertical_datum_edit.text().strip(),
        }


class ImportLineDialog(QDialog):
    """Line import preflight dialog with explicit line metadata."""

    def __init__(self, parent: QWidget | None, *, source_path: str, default_line_id: str, default_name: str, preview_lines: list[str], can_import: bool) -> None:
        super().__init__(parent)
        self.setWindowTitle("导入测线数据预检")
        self.setMinimumWidth(620)
        layout = QVBoxLayout(self)
        form = QFormLayout()
        self.line_id_edit = QLineEdit(default_line_id)
        self.name_edit = QLineEdit(default_name)
        self.source_edit = QLineEdit(source_path)
        self.source_edit.setReadOnly(True)
        form.addRow("测线编号 *", self.line_id_edit)
        form.addRow("测线名称", self.name_edit)
        form.addRow("数据文件", self.source_edit)
        layout.addLayout(form)
        preview_box = QTextEdit()
        preview_box.setReadOnly(True)
        preview_box.setMinimumHeight(150)
        preview_box.setPlainText("\n".join(preview_lines))
        layout.addWidget(preview_box)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("确认导入")
        buttons.button(QDialogButtonBox.StandardButton.Ok).setEnabled(bool(can_import))
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def values(self) -> dict[str, str]:
        return {
            "line_id": self.line_id_edit.text().strip(),
            "name": self.name_edit.text().strip(),
            "source_path": self.source_edit.text().strip(),
        }




class SourceFilesDialog(QDialog):
    """Lightweight project data-ops dialog focused on source-file status."""

    def __init__(self, parent: QWidget | None = None, *, records: list[object] | tuple[object, ...], summary_text: str = "") -> None:
        super().__init__(parent)
        self.setWindowTitle("项目数据运维中心 - 源文件")
        self.setMinimumSize(780, 420)
        layout = QVBoxLayout(self)
        title = QLabel("源文件状态")
        title.setObjectName("sectionTitle")
        layout.addWidget(title)
        tip = QLabel(summary_text or "显示每条测线外部原始来源文件的可用状态；项目删除不会删除项目目录外源文件。")
        tip.setWordWrap(True)
        tip.setObjectName("activityDesc")
        layout.addWidget(tip)
        table = QTableWidget(len(records), 7)
        table.setHorizontalHeaderLabels(["测线", "类型", "状态", "文件名", "大小", "检查时间", "说明"])
        table.verticalHeader().setVisible(False)
        table.setAlternatingRowColors(True)
        table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        for row, record in enumerate(records):
            size_bytes = int(getattr(record, "source_size_bytes", 0) or 0)
            values = [
                getattr(record, "line_id", "--"),
                getattr(record, "role", "--"),
                getattr(record, "status_label", getattr(record, "status", "--")),
                getattr(record, "source_filename", "--"),
                f"{size_bytes / (1024 * 1024):.2f} MB" if size_bytes else "--",
                getattr(record, "last_checked_at", "--") or "--",
                getattr(record, "warning", "") or getattr(record, "source_path", "--"),
            ]
            for col, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                item.setToolTip(str(value))
                table.setItem(row, col, item)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(table, 1)
        detail = QLabel("常用操作：在项目管理页选中测线后使用“重新定位源文件”；使用“来源清单”导出 CSV 归档。")
        detail.setObjectName("activityDesc")
        detail.setWordWrap(True)
        layout.addWidget(detail)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)


class QualityReportDialog(QDialog):
    """Display one line quality report and optionally request orientation fix."""

    def __init__(self, parent: QWidget | None = None, *, line_id: str, report: object | None, can_fix_orientation: bool = False) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"数据质检详情 - {line_id}")
        self.setMinimumWidth(680)
        self.fix_requested = False
        layout = QVBoxLayout(self)
        title = QLabel(f"测线 {line_id} 数据质检详情")
        title.setObjectName("sectionTitle")
        layout.addWidget(title)

        detail = QTextEdit()
        detail.setReadOnly(True)
        detail.setMinimumHeight(260)
        detail.setPlainText(self._format_report(report))
        layout.addWidget(detail, 1)

        tip = QLabel("说明：方向修正会转置标准化 B-scan 矩阵、备份修正前数据，并重新生成质检报告。")
        tip.setWordWrap(True)
        tip.setObjectName("activityDesc")
        layout.addWidget(tip)

        buttons = QHBoxLayout()
        buttons.addStretch(1)
        self.fix_button = QPushButton("转置修正并重新质检")
        self.fix_button.setEnabled(bool(can_fix_orientation))
        self.fix_button.clicked.connect(self._request_fix)
        buttons.addWidget(self.fix_button)
        close_button = QPushButton("关闭")
        close_button.clicked.connect(self.accept)
        buttons.addWidget(close_button)
        layout.addLayout(buttons)

    def _format_report(self, report: object | None) -> str:
        if report is None:
            return "尚未生成质检报告。请先运行数据质检。"
        lines = [
            f"状态：{getattr(report, 'status_label', '--')}",
            f"检查时间：{getattr(report, 'checked_at', '--')}",
            f"矩阵尺寸：{getattr(report, 'sample_count', 0)} × {getattr(report, 'trace_count', 0)}",
            f"时间窗：{float(getattr(report, 'time_window_ns', 0.0)):.3f} ns",
            f"测线长度：{float(getattr(report, 'length_m', 0.0)):.3f} m",
            f"振幅范围：{float(getattr(report, 'amplitude_min', 0.0)):.6g} ~ {float(getattr(report, 'amplitude_max', 0.0)):.6g}",
            f"99.5% 振幅：{float(getattr(report, 'amplitude_p995', 0.0)):.6g}",
            f"有限值比例：{float(getattr(report, 'finite_ratio', 0.0)):.4%}",
            f"NaN/Inf 比例：{float(getattr(report, 'nan_ratio', 0.0)):.4%}",
            f"轨迹点数：{getattr(report, 'trajectory_points', 0)}",
            f"方向判断：{getattr(report, 'orientation', '--')}",
            f"方向说明：{getattr(report, 'orientation_message', '--')}",
            f"建议操作：{getattr(report, 'suggested_action', '--')}",
            "",
            "问题列表：",
        ]
        issues = list(getattr(report, "issues", []) or [])
        if not issues:
            lines.append("- 未发现阻断性问题。")
        for issue in issues:
            lines.append(f"- [{getattr(issue, 'severity', '--')}] {getattr(issue, 'code', '--')}: {getattr(issue, 'message', '')}")
            suggestion = getattr(issue, "suggestion", "")
            if suggestion:
                lines.append(f"  建议：{suggestion}")
        return "\n".join(lines)

    def _request_fix(self) -> None:
        self.fix_requested = True
        self.accept()

__all__ = ["ProjectCreateDialog", "ProjectSettingsDialog", "ImportLineDialog", "QualityReportDialog", "SourceFilesDialog"]
