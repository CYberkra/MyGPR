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

    def test_controls_use_consistent_comfortable_metrics(self, qapp):
        form = _make_form(_SCHEMA[:4])
        assert form._layout.spacing() == 8
        heights = {
            name: (type(editor).__name__, editor.minimumHeight(),
                   editor.maximumHeight(), editor.height())
            for name, editor in form._editors.items()
        }
        assert all(height[1] >= 34 for height in heights.values()), heights


class TestParamFormFallback:
    def test_unknown_type_falls_back_without_crash(self, qapp):
        form = _make_form([
            {'name': 'x', 'label': 'X', 'type': 'enum', 'default': 'a'},
        ])
        # 未知类型不注册编辑器（只读降级行），values() 不包含该键
        assert 'x' not in form._editors
        assert form.values() == {}


class TestConditionalParameters:
    def test_mode_rules_control_visibility_enabled_state_and_payload(self, qapp):
        from ui.widgets.param_form import ParamForm

        form = ParamForm()
        form.set_schema([
            {'name': 'window_mode', 'type': 'str', 'default': 'auto',
             'choices': ('auto', 'manual', 'full')},
            {'name': 'mode', 'type': 'str', 'default': 'mark',
             'choices': ('mark', 'interpolate')},
            {'name': 'time_windows_ns', 'type': 'str', 'default': '1:2',
             'visible_when': {'window_mode': 'manual'}},
            {'name': 'max_run_traces', 'type': 'int', 'default': 4,
             'enabled_when': {'mode': 'interpolate'}},
        ])
        time_editor = form._editors['time_windows_ns']
        run_editor = form._editors['max_run_traces']
        assert time_editor.isHidden()
        assert not run_editor.isEnabled()
        assert form.values()['time_windows_ns'] == '1:2'

        form.set_values({'window_mode': 'manual', 'mode': 'interpolate'})
        assert time_editor.isVisibleTo(form)
        assert run_editor.isEnabled()
        assert form.values()['max_run_traces'] == 4

    def test_hidden_omit_can_exclude_value_without_deleting_it(self, qapp):
        from ui.widgets.param_form import ParamForm

        form = ParamForm()
        form.set_schema([
            {'name': 'window_mode', 'type': 'str', 'default': 'auto',
             'choices': ('auto', 'manual')},
            {'name': 'time_windows_ns', 'type': 'str', 'default': '1:2',
             'visible_when': {'window_mode': 'manual'}, 'hidden': 'omit'},
        ])
        assert 'time_windows_ns' not in form.values()
        form.set_values({'window_mode': 'manual'})
        assert form.values()['time_windows_ns'] == '1:2'
        form.set_values({'window_mode': 'auto'})
        assert 'time_windows_ns' not in form.values()
        form.set_values({'window_mode': 'manual'})
        assert form.values()['time_windows_ns'] == '1:2'

    def test_explicit_advanced_metadata_moves_parameter_out_of_basic_group(self, qapp):
        from ui.widgets.param_form import ParamForm

        form = ParamForm()
        form.set_schema([
            {'name': 'basic', 'type': 'int', 'default': 1},
            {'name': 'advanced', 'type': 'int', 'default': 2, 'advanced': True},
        ])
        assert form._advanced_btn is not None
        assert form._advanced_widget is not None
        assert form._advanced_names == {'advanced'}

    def test_condition_rule_changes_rebuild_schema(self, qapp):
        from ui.widgets.param_form import _schema_signature

        base = [{'name': 'mode', 'type': 'str', 'default': 'auto'}]
        assert _schema_signature(base) != _schema_signature([
            {**base[0], 'visible_when': {'mode': 'manual'}}])
