# -*- coding: utf-8 -*-
"""ParamForm 参数表单测试：int/float/str/bool 控件构建、值读写、未知类型降级。

回归覆盖：str/bool 参数此前会使表单崩溃（_build_row 一律 DoubleSpinBox + float(default)），
导致约 14 个核心方法（滤波/小波/动补偿/迁移）无法编辑参数。
"""
from __future__ import annotations

import pytest

pytest.importorskip("qfluentwidgets")  # 后端 CI（无 Qt）自动跳过，见 tests/conftest.py qapp 设计
from qfluentwidgets import CheckBox, ComboBox, DoubleSpinBox, LineEdit, SpinBox  # noqa: E402

from ui.widgets.param_form import ParamForm  # noqa: E402


_SCHEMA = [
    {'name': 'filter_type', 'label': '滤波类型', 'type': 'str', 'default': 'bandpass'},
    {'name': 'height_source', 'label': '高度源', 'type': 'str', 'default': 'auto',
     'choices': ('auto', 'height_agl_m', 'flight_height_m')},
    {'name': 'normalize', 'label': '归一化', 'type': 'bool', 'default': False},
    {'name': 'rank', 'label': '秩', 'type': 'int', 'default': 3, 'min': 1},
    {'name': 'gain', 'label': '增益', 'type': 'float', 'default': 1.5},
]


def _make_form(schema):
    form = ParamForm()
    form.set_schema(schema)
    return form


class TestParamFormTypes:
    def test_builds_str_bool_controls_without_crash(self, qapp):
        form = _make_form(_SCHEMA)
        assert isinstance(form._editors['filter_type'], LineEdit)
        assert isinstance(form._editors['height_source'], ComboBox)
        assert isinstance(form._editors['normalize'], CheckBox)
        assert isinstance(form._editors['rank'], SpinBox)
        assert isinstance(form._editors['gain'], DoubleSpinBox)

    def test_default_values_read_back(self, qapp):
        form = _make_form(_SCHEMA)
        values = form.values()
        assert values == {
            'filter_type': 'bandpass',
            'height_source': 'auto',
            'normalize': False,
            'rank': 3,
            'gain': 1.5,
        }

    def test_set_values_roundtrip(self, qapp):
        form = _make_form(_SCHEMA)
        form.set_values({
            'filter_type': 'lowpass',
            'height_source': 'height_agl_m',
            'normalize': True,
            'rank': 7,
            'gain': 2.5,
        })
        assert form.values() == {
            'filter_type': 'lowpass',
            'height_source': 'height_agl_m',
            'normalize': True,
            'rank': 7,
            'gain': 2.5,
        }

    def test_set_values_ignores_unknown_keys(self, qapp):
        form = _make_form(_SCHEMA)
        form.set_values({'does_not_exist': 123})
        assert 'does_not_exist' not in form.values()

    def test_schema_rebuild_keeps_values(self, qapp):
        form = _make_form(_SCHEMA)
        form.set_values({'filter_type': 'lowpass', 'normalize': True})
        # 相同结构重建不丢值（结构签名命中跳过）
        form.set_schema(list(_SCHEMA))
        assert form.values()['filter_type'] == 'lowpass'
        assert form.values()['normalize'] is True


class TestParamFormFallback:
    def test_unknown_type_falls_back_without_crash(self, qapp):
        form = _make_form([
            {'name': 'x', 'label': 'X', 'type': 'enum', 'default': 'a'},
        ])
        # 未知类型不注册编辑器（只读降级行），values() 不包含该键
        assert 'x' not in form._editors
        assert form.values() == {}
