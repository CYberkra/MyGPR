"""Explicit marker draft editor; acceptance follows successful persistence."""
from __future__ import annotations

from dataclasses import asdict, replace
import json

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFormLayout, QLabel, QLineEdit,
    QPlainTextEdit, QScrollArea, QSpinBox, QVBoxLayout, QWidget,
)

from ui.desktop_backend_facade import EventInterval, MarkerAnchor, MarkerRecord, PURPOSES


class MarkerEditor(QDialog):
    save_requested = pyqtSignal(object)

    def __init__(self, view, record, parent=None):
        super().__init__(parent)
        self.view, self.record = view, record
        self.setWindowTitle('事件区间' if isinstance(record, EventInterval) else '点标记')
        self.resize(400, 680)
        self.setMinimumWidth(360)
        layout = QVBoxLayout(self)
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        body = QWidget(scroll)
        form = QFormLayout(body)
        self.name = QLineEdit(record.name, body)
        self.purpose = QComboBox(body)
        self.purpose.setEditable(True)
        self.purpose.addItems(list(dict.fromkeys((*PURPOSES, *(row['purpose'] for row in view.rows)))))
        self.purpose.setCurrentText(record.purpose)
        self.evidence = QComboBox(body)
        self.evidence.setEditable(True)
        self.evidence.addItems(['现场记录', '既有资料', '雷达判读', '其他'])
        self.evidence.setCurrentText(record.evidence)
        self.note = QPlainTextEdit(record.note, body)
        self.note.setMaximumHeight(90)
        self.excluded = QCheckBox('排除使用', body)
        self.excluded.setChecked(record.excluded)
        for title, widget in (('名称', self.name), ('用途', self.purpose), ('信息来源', self.evidence),
                              ('备注', self.note), ('', self.excluded)):
            form.addRow(title, widget)
        self.source = QComboBox(body)
        self.source.addItem('原始采集', view.document.source_id)
        if view.target_id != view.document.source_id:
            self.source.addItem('当前成果', view.target_id)
        anchor = record.start if isinstance(record, EventInterval) else record.anchor
        if anchor and self.source.findData(anchor.source_id) < 0:
            self.source.addItem('历史来源（当前无法定位）', anchor.source_id)
        self.source.setCurrentIndex(max(0, self.source.findData(anchor.source_id if anchor else view.target_id)))
        form.addRow('边界来源', self.source)
        self.start = self._spin(anchor)
        form.addRow('起始道' if isinstance(record, EventInterval) else '道号', self.start)
        self.start_ref = self.end_ref = None
        if isinstance(record, EventInterval):
            self.end = self._spin(record.end)
            form.addRow('结束道', self.end)
            self.start_ref = self._reference(record.start_marker_id)
            self.end_ref = self._reference(record.end_marker_id)
            form.addRow('起始标记', self.start_ref)
            form.addRow('结束标记', self.end_ref)
            self.inc_start = QCheckBox('包含起始道', body)
            self.inc_end = QCheckBox('包含结束道', body)
            self.inc_start.setChecked(record.include_start)
            self.inc_end.setChecked(record.include_end)
            form.addRow(self.inc_start, self.inc_end)
            self.time_start = QLineEdit(record.time_start, body)
            self.time_end = QLineEdit(record.time_end, body)
            for field in (self.time_start, self.time_end):
                field.setPlaceholderText('ISO 8601，含时区；可留空')
            form.addRow('采集时刻起', self.time_start)
            form.addRow('采集时刻止', self.time_end)
            self.review = QCheckBox('已复核当前边界', body)
            self.review.setChecked(record.review_state == 'reviewed')
            form.addRow(self.review)
        elif record.origin == 'acquisition':
            self.start.setEnabled(False)
            self.source.setEnabled(False)
        self.links = QLineEdit(', '.join(record.links), body)
        self.links.setPlaceholderText('关联解释对象 ID，逗号分隔')
        form.addRow('解释关联', self.links)
        details = self._details(view, record)
        self.details = QPlainTextEdit(details, body)
        self.details.setReadOnly(True)
        self.details.setMinimumHeight(130)
        form.addRow('记录详情', self.details)
        scroll.setWidget(body)
        layout.addWidget(scroll, 1)
        self.error = QLabel(self)
        self.error.setWordWrap(True)
        self.error.setAccessibleName('保存状态')
        layout.addWidget(self.error)
        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel, self)
        self.buttons.button(QDialogButtonBox.StandardButton.Save).setText('保存')
        self.buttons.button(QDialogButtonBox.StandardButton.Cancel).setText('取消')
        self.buttons.accepted.connect(self._save)
        self.buttons.rejected.connect(self.reject)
        self.buttons.button(QDialogButtonBox.StandardButton.Save).setEnabled(not view.read_only)
        layout.addWidget(self.buttons)

    def _spin(self, anchor):
        widget = QSpinBox(self)
        widget.setRange(0, 2_000_000_000)
        widget.setSpecialValueText('未设置')
        widget.setValue(anchor.trace + 1 if anchor else 0)
        return widget

    def _reference(self, selected):
        combo = QComboBox(self)
        combo.addItem('直接输入边界', '')
        for marker in self.view.document.markers:
            if not marker.excluded:
                combo.addItem(f'{marker.name or marker.uid[:8]} · {marker.uid[:8]}', marker.uid)
        combo.setCurrentIndex(max(0, combo.findData(selected)))
        return combo

    def _anchor(self, widget):
        return MarkerAnchor(self.source.currentData(), widget.value() - 1) if widget.value() else None

    def _save(self):
        try:
            changes = dict(name=self.name.text().strip(), purpose=self.purpose.currentText().strip() or '其他',
                evidence=self.evidence.currentText().strip(), note=self.note.toPlainText(),
                excluded=self.excluded.isChecked(), links=tuple(item.strip() for item in self.links.text().split(',') if item.strip()))
            if isinstance(self.record, EventInterval):
                changes.update(start=self._anchor(self.start), end=self._anchor(self.end),
                    include_start=self.inc_start.isChecked(), include_end=self.inc_end.isChecked(),
                    time_start=self.time_start.text().strip(), time_end=self.time_end.text().strip(),
                    time_source_id=self.source.currentData(),
                    review_state='reviewed' if self.review.isChecked() else 'unreviewed')
                markers = {item.uid: item for item in self.view.document.markers}
                for side in ('start', 'end'):
                    uid = getattr(self, f'{side}_ref').currentData()
                    mark = markers.get(uid)
                    changes[f'{side}_marker_id'] = uid
                    changes[f'{side}_marker_revision'] = mark.revision if mark else 0
                    if mark:
                        changes[side] = mark.anchor
                if (changes['time_start'] or changes['time_end']) and (changes['start'] or changes['end']):
                    raise ValueError('道号边界与纯时间事件请分别记录；纯时间事件的道号设为“未设置”')
            elif self.record.origin != 'acquisition':
                changes['anchor'] = self._anchor(self.start)
            record = replace(self.record, **changes)
            anchors = (record.start, record.end) if isinstance(record, EventInterval) else (record.anchor,)
            if any(anchor and anchor.source_id == self.view.target_id and anchor.trace >= self.view.count for anchor in anchors):
                raise ValueError('道号超出当前数据范围')
        except (ValueError, TypeError) as exc:
            self.failed(str(exc))
            return
        self.error.setText('正在保存…')
        self.buttons.setEnabled(False)
        self.save_requested.emit(record)

    def failed(self, message):
        self.buttons.setEnabled(True)
        self.error.setText(message)

    @staticmethod
    def _details(view, record):
        row = next((item for item in view.rows if item['uid'] == record.uid), None)
        projection = row['projection'] if row else None
        text = [f'测线：{view.document.line_id}', f'记录版本：{record.revision}',
                f'当前来源：{view.target_id}', f'显示时区：{view.timezone}',
                f'逐道采集时间：{"可按 UTC 选道" if view.time_basis_confirmed else "缺失、回跳或基准未确认"}']
        if row:
            text.append('状态：' + row['status'])
        if projection:
            text.append('当前成果道号：' + ', '.join(f'{value + 1:g}' for value in projection.positions))
            if projection.spans:
                text.append('当前片段：' + ', '.join(f'{a + 1}–{b}' for a, b in projection.spans))
                text.append(f'当前道数：{sum(b - a for a, b in projection.spans)}')
        if isinstance(record, MarkerRecord) and record.original:
            text.extend(['原始采集记录：', json.dumps(record.original, ensure_ascii=False, indent=2)])
        if isinstance(record, EventInterval):
            text.append('边界快照：' + json.dumps(asdict(record), ensure_ascii=False, indent=2))
        return '\n'.join(text)
