"""Collapsible interpretation properties, semantic zones and borehole validation."""

from PyQt6.QtWidgets import QPlainTextEdit
from qfluentwidgets import ComboBox, DoubleSpinBox, LineEdit, PushButton

from ui.widgets.collapsible_card import make_collapsible_card


def _spin(parent, low, high, value=0.0, decimals=2):
    spin = DoubleSpinBox(parent)
    spin.setRange(low, high)
    spin.setDecimals(decimals)
    spin.setValue(value)
    return spin


def build_interpretation_tools(page, layout):
    properties, form = make_collapsible_card('解释属性', collapsed=True)
    page._annotation_name = LineEdit(properties)
    page._annotation_name.setPlaceholderText('基覆界面')
    form.addWidget(page._annotation_name)
    page._interface_type = LineEdit(properties)
    page._interface_type.setText('basal_interface')
    form.addWidget(page._interface_type)
    page._confidence = _spin(properties, 0.0, 1.0, 0.8)
    form.addWidget(page._confidence)
    page._uncertainty = _spin(properties, 0.0, 100000.0, 0.0, 1)
    page._uncertainty.setSuffix(' 采样点')
    form.addWidget(page._uncertainty)
    page._annotation_note = LineEdit(properties)
    page._annotation_note.setPlaceholderText('解释备注')
    form.addWidget(page._annotation_note)
    page._properties_btn = PushButton('应用解释属性', properties)
    page._properties_btn.clicked.connect(page._emit_properties)
    form.addWidget(page._properties_btn)
    layout.addWidget(properties)

    zones, form = make_collapsible_card('语义区段', collapsed=True)
    page._zone_start = _spin(zones, 0.0, 1_000_000_000.0, 0.0, 0)
    page._zone_end = _spin(zones, 0.0, 1_000_000_000.0, 0.0, 0)
    page._zone_sample_start = _spin(zones, 0.0, 1_000_000_000.0, 0.0, 0)
    page._zone_sample_end = _spin(zones, 0.0, 1_000_000_000.0, 0.0, 0)
    for label, widget in (('起道', page._zone_start), ('止道', page._zone_end),
                          ('起采样', page._zone_sample_start), ('止采样', page._zone_sample_end)):
        widget.setPrefix(label + ' ')
        form.addWidget(widget)
    page._zone_kind = ComboBox(zones)
    for text, key in (('弱反射', 'weak'), ('无界面', 'no_interface'), ('忽略', 'ignore')):
        page._zone_kind.addItem(text, userData=key)
    form.addWidget(page._zone_kind)
    page._zone_note = LineEdit(zones)
    page._zone_note.setPlaceholderText('区段备注')
    page._zone_kind.currentTextChanged.connect(
        lambda text: page._zone_note.setPlaceholderText(f'{text}区段备注'))
    form.addWidget(page._zone_note)
    page._zone_btn = PushButton('添加语义区段', zones)
    page._zone_btn.clicked.connect(page._emit_zone)
    form.addWidget(page._zone_btn)
    layout.addWidget(zones)

    boreholes, form = make_collapsible_card('钻孔验证', collapsed=True)
    page._borehole_id = LineEdit(boreholes)
    page._borehole_id.setPlaceholderText('钻孔编号')
    form.addWidget(page._borehole_id)
    page._borehole_trace = _spin(boreholes, 0.0, 1_000_000_000.0, 0.0, 1)
    page._borehole_trace.setPrefix('对应道 ')
    form.addWidget(page._borehole_trace)
    page._borehole_depth = _spin(boreholes, 0.0, 100000.0, 0.0, 3)
    page._borehole_depth.setPrefix('基覆深度/m ')
    form.addWidget(page._borehole_depth)
    page._borehole_save_btn = PushButton('保存钻孔', boreholes)
    page._borehole_save_btn.clicked.connect(page._emit_borehole)
    form.addWidget(page._borehole_save_btn)
    page._borehole_threshold = _spin(boreholes, 0.0, 10000.0, 1.0, 3)
    page._borehole_threshold.setPrefix('误差阈值/m ')
    form.addWidget(page._borehole_threshold)
    page._borehole_compare_btn = PushButton('对比当前测线', boreholes)
    page._borehole_compare_btn.clicked.connect(
        lambda: page.borehole_compare_requested.emit(page._borehole_threshold.value()))
    page._borehole_refresh_btn = PushButton('刷新钻孔', boreholes)
    page._borehole_refresh_btn.clicked.connect(page.borehole_refresh_requested)
    form.addWidget(page._borehole_compare_btn)
    form.addWidget(page._borehole_refresh_btn)
    page._borehole_result_label = QPlainTextEdit(boreholes)
    page._borehole_result_label.setReadOnly(True)
    page._borehole_result_label.setMaximumHeight(120)
    page._borehole_result_label.setPlaceholderText('实测/解释深度与误差结果')
    form.addWidget(page._borehole_result_label)
    layout.addWidget(boreholes)
