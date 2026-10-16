"""Internal processing-page step editing and parameter presentation."""
import json

from PyQt6.QtCore import Qt
from qfluentwidgets import InfoBar, InfoBarPosition


class ProcessingSteps:
    """Shared page state remains owned by ProcessingPage; public methods stay inherited."""

    def _display_name(self, method_id):
        return str(self._methods_by_id.get(method_id, {}).get('display_name') or method_id)

    def set_methods(self, methods):
        self._methods = [dict(m) for m in methods or []]
        self._methods_by_id = {m.get('method_id', ''): m for m in self._methods}
        if self._method_picker is not None:
            self._method_picker.set_methods(self._methods)

    def set_autotune_result(self, method_id, result):
        result = dict(result or {})
        self._autotune_result = (method_id, result)
        self._autotune_result_label.setText(json.dumps(result, ensure_ascii=False, indent=2, default=str))
        self._autotune_card.set_collapsed(False, animate=False)
        self._refresh_ui()

    def _on_chain_step_toggled(self, index: int, enabled: bool) -> None:
        """启用/禁用：与当前状态不同才翻转（PipelineList 内置翻转语义）。"""
        if self._draft_locked():
            return
        steps = self._pipeline_list.steps()
        if 0 <= index < len(steps) and steps[index]['enabled'] != enabled:
            self._pipeline_list._toggle_enabled(index)

    def _on_chain_step_removed(self, index: int) -> None:
        if self._draft_locked():
            return
        self._pipeline_list._remove_step(index)

    def _on_chain_step_moved(self, source: int, target: int) -> None:
        """chip 拖拽落点 → 步骤换位（target 为插入位语义）。"""
        if self._draft_locked():
            return
        self._pipeline_list._move_step_to(source, target)

    def _open_step_editor(self, index: int) -> None:
        """chip 右键 → 合一弹窗：快照来自 PipelineList（唯一数据源），
        弹窗只持快照；写回经公开 API + 手动 sig_changed.emit()。"""
        if self._draft_locked():
            return
        from ui.widgets.step_editor import StepEditorPopup
        steps = self._pipeline_list.steps()
        if not (0 <= index < len(steps)):
            return
        self._pipeline_list.select_step(index)
        if self._step_editor is not None:
            self._step_editor.close()
        step = steps[index]
        method = self._methods_by_id.get(step.get('method_id', '') or '', {})
        snapshot = {
            'title': step.get('label') or step.get('method_id', ''),
            'method_id': step.get('method_id', ''),
            'params': dict(step.get('params') or {}),
            'enabled': bool(step.get('enabled', True)),
            'is_first': index == 0,
            'is_last': index == len(steps) - 1,
            'schema': method.get('parameter_schema') or [],
        }
        # 每次新建实例：天然无重复信号连接，关窗即销毁（未应用改动随窗丢弃）
        popup = StepEditorPopup(self)
        self._step_editor = popup
        context = self._draft_key()
        def current(callback, *args):
            current_steps = self._pipeline_list.steps()
            if (context == self._draft_key() and not self._draft_locked()
                    and self._step_editor is popup and 0 <= index < len(current_steps)
                    and current_steps[index] == step):
                callback(*args)
        popup.load_snapshot(index, snapshot)
        popup.sig_applied.connect(lambda payload: current(self._on_step_editor_applied, payload))
        popup.sig_move_requested.connect(lambda source, target: current(self._on_chain_step_moved, source, target))
        popup.sig_duplicate_requested.connect(lambda value: current(self._duplicate_step, value))
        popup.sig_replace_requested.connect(lambda value: current(self._on_step_editor_replace, value))
        popup.sig_remove_requested.connect(lambda value: current(self._on_chain_step_removed, value))
        popup.finished.connect(lambda result: self._on_step_editor_finished(result, popup))
        popup.show()
        self._position_step_editor(popup, index)

    def _on_step_editor_finished(self, _result: int, popup=None) -> None:
        """弹窗关闭（应用/结构动作/Esc/外部点击任一路径）：引用释放。"""
        if popup is None or self._step_editor is popup:
            self._step_editor = None
        if popup is not None:
            popup.deleteLater()

    def _on_step_editor_applied(self, payload: dict) -> None:
        """「应用」：参数写回（update_step_params 自动 sig_changed →
        琥珀 hint 点亮）；启停变化走既有 _toggle_enabled（chip 变暗 +
        运行跳过语义不变，配置保留）。"""
        if self._draft_locked():
            return
        payload = dict(payload or {})
        index = int(payload.get('index', -1))
        steps = self._pipeline_list.steps()
        if not (0 <= index < len(steps)):
            return
        self._pipeline_list.update_step_params(
            index, dict(payload.get('params') or {}))
        enabled = bool(payload.get('enabled', True))
        if steps[index]['enabled'] != enabled:
            self._pipeline_list._toggle_enabled(index)

    def _on_step_editor_replace(self, index: int) -> None:
        """「替换算法」→ 打开方法选择器（purpose='replace'，Q2 定案：
        纯浏览态不预选当前方法，双击即替换）。"""
        self._open_method_picker(purpose='replace', index=index)

    def _duplicate_step(self, index: int) -> None:
        """复制步骤：同算法同参数的相邻副本。set_steps 不发信号 →
        手动 emit（参照 _on_adopt_params :1293-1295 模式，琥珀 hint 点亮）。"""
        if self._draft_locked():
            return
        steps = self._pipeline_list.steps()
        if not (0 <= index < len(steps)):
            return
        copy = dict(steps[index])
        copy['params'] = dict(steps[index].get('params') or {})
        steps.insert(index + 1, copy)
        self._pipeline_list.set_steps(steps)
        self._pipeline_list.select_step(index + 1)
        self._pipeline_list.sig_changed.emit()

    def _replace_step_method(self, index: int, method_id: str) -> None:
        """替换第 index 步的算法：位置不变，参数按新 schema 取默认，
        恢复启用。set_steps 不发信号 → 手动 emit（琥珀 hint 点亮）。"""
        if self._draft_locked():
            return
        steps = self._pipeline_list.steps()
        if not (0 <= index < len(steps)):
            return
        method = self._methods_by_id.get(method_id, {})
        label = (method.get('display_name') or method.get('name')
                 or method_id)
        params = {item.get('name'): item.get('default')
                  for item in (method.get('parameter_schema') or [])
                  if item.get('name') is not None}
        steps[index] = {'method_id': method_id, 'label': label,
                        'params': params, 'enabled': True}
        self._pipeline_list.set_steps(steps)
        self._pipeline_list.select_step(index)   # 选中 → 参数表单重载新 schema
        self._pipeline_list.sig_changed.emit()

    def _on_method_selected(self, method_id: str) -> None:
        self._selected_method_id = method_id
        method = self._methods_by_id.get(method_id, {})
        self._autotune_method_label.setText(
            method.get('display_name') or method_id or '--')
        self._update_autotune_enabled()

    def _update_autotune_enabled(self) -> None:
        self._autotune_btn.setEnabled(
            bool(self._selected_method_id) and not self._autotune_running)

    def set_autotune_running(self, running: bool) -> None:
        """AutoTune 运行期间禁用「开始调参」，防重复提交（P2-7）。"""
        self._autotune_running = bool(running)
        self._update_autotune_enabled()

    def _open_method_picker(self, purpose: str = 'append', index: int = -1) -> None:
        """「+」/「添加方法」统一入口：非模态方法选择器（单实例）。

        purpose='append'：双击条目 → 链尾追加；purpose='replace'：双击
        条目 → 替换第 index 步（P0-3 编辑弹窗「替换算法」分流口，T03 消费：
        位置不变、参数按新 schema 取默认）。存活实例唤起并复位状态；
        关窗后引用置 None，再开 = 新实例 = 状态天然复位。
        """
        if self._draft_locked():
            return
        from ui.widgets.method_picker import MethodPickerDialog
        self._picker_purpose = str(purpose) if purpose in ('append', 'replace') \
            else 'append'
        self._picker_index = int(index) if str(index).lstrip('-').isdigit() else -1
        if self._method_picker is not None:
            try:
                self._method_picker._draft_context = self._draft_key()
                self._method_picker._target_epoch = self._draft_edit_epoch
                self._method_picker.reset_state()
                self._method_picker.show()
                self._method_picker.activateWindow()
                self._method_picker.raise_()
                return
            except RuntimeError:      # C++ 侧已销毁（teardown 竞态）
                self._method_picker = None
        picker = MethodPickerDialog(self)
        self._method_picker = picker
        picker._draft_context = self._draft_key()
        picker._target_epoch = self._draft_edit_epoch
        picker.set_methods(self._methods)
        def chosen(method_id):
            if picker is self._method_picker and picker._draft_context == self._draft_key():
                if self._picker_purpose == 'replace' and (
                        picker._target_epoch != self._draft_edit_epoch
                        or not 0 <= self._picker_index < self._pipeline_list.count()):
                    picker.close()
                    InfoBar.warning(title='步骤已变化', content='请重新打开要替换的算法节点',
                                    duration=3000, parent=self)
                    return
                replace = self._picker_purpose == 'replace'
                self._on_method_picked(method_id)
                if replace:
                    picker.close()
        picker.sig_method_chosen.connect(chosen)
        picker.finished.connect(lambda result: self._on_method_picker_finished(result, picker))
        picker.show()

    def _on_method_picked(self, method_id: str) -> None:
        """选择器双击条目：按 purpose 分流（链尾追加 / 就地替换）。"""
        if self._picker_purpose == 'replace' and self._picker_index >= 0:
            self._replace_step_method(self._picker_index, method_id)
        else:
            self._add_method_to_pipeline(method_id)

    def _on_method_picker_finished(self, _result: int, picker=None) -> None:
        """弹窗关闭（Esc/外部点击/任何路径）：引用置 None，再开=新实例。"""
        if picker is None or self._method_picker is picker:
            self._method_picker = None
        if picker is not None:
            picker.deleteLater()

    def _add_method_to_pipeline(self, method_id: str) -> None:
        if self._draft_locked():
            return
        method = self._methods_by_id.get(method_id, {})
        label = method.get('display_name') or method.get('name') or method_id
        params = {item.get('name'): item.get('default')
                  for item in (method.get('parameter_schema') or [])
                  if item.get('name') is not None}
        self._pipeline_list.add_step(method_id, label, params)

    def _on_step_selected(self, index: int) -> None:
        self._selected_step = index if index >= 0 else -1
        self._chain_strip.select_step(self._selected_step)
        steps = self._pipeline_list.steps()
        if not (0 <= self._selected_step < len(steps)):
            self._param_form.clear()
            self._sync_param_empty_hint()
            return
        step = steps[self._selected_step]
        self._on_method_selected(str(step.get('method_id') or ''))
        method = self._methods_by_id.get(step.get('method_id', ''), {})
        schema = method.get('parameter_schema') or []
        if schema:
            self._param_form.set_schema(schema)
            self._param_form.set_values(step.get('params') or {})
        else:
            # 无 schema：保持表单为空
            self._param_form.clear()
        self._sync_param_empty_hint()

    def _sync_param_empty_hint(self) -> None:
        """表单无字段 → 显示空态引导（参数卡不再留白）。"""
        self._param_empty_hint.setVisible(not self._param_form._schema)

    def _auto_write_params_to_selected(self) -> None:
        """参数表单值变化 → 自动写入选中步骤（B1：改值即生效，无按钮）。

        静默写回：表单由本页驱动（set_values 不触发 sig_changed 循环），此处
        仅处理用户编辑；步骤失效时静默丢弃（选中态变化会重新载入表单）。
        """
        if self._draft_locked() or not (0 <= self._selected_step < len(self._pipeline_list.steps())):
            return
        self._pipeline_list.update_step_params(self._selected_step,
                                               self._param_form.values())

    def _on_autotune_clicked(self) -> None:
        self._autotune_context = self._draft_key()
        method_id = self._selected_method_id
        if not method_id:
            InfoBar.warning(title='AutoTune 自动调参',
                            content='请先在方法库中选择方法',
                            orient=Qt.Orientation.Horizontal, isClosable=True,
                            position=InfoBarPosition.TOP, duration=3000,
                            parent=self)
            return
        params_hint = self._param_form.values()
        self.autotune_requested.emit(method_id, params_hint, self._input_artifact_id())

    def _on_adopt_params(self) -> None:
        if self._draft_locked() or self._autotune_context != self._draft_key():
            return
        if self._autotune_result is None:
            return
        method_id, result = self._autotune_result
        best = dict(result.get('best_params') or {})
        if not best:
            return
        # 选中步骤与方法一致 → 写回选中步骤；否则写回处理链中第一个同方法
        # 步骤并选中；处理链中没有同方法步骤时仅载入参数表单
        steps = self._pipeline_list.steps()
        target = -1
        if (0 <= self._selected_step < len(steps)
                and steps[self._selected_step].get('method_id') == method_id):
            target = self._selected_step
        else:
            for i, step in enumerate(steps):
                if step.get('method_id') == method_id:
                    target = i
                    break
        if target >= 0:
            merged = dict(steps[target].get('params') or {})
            merged.update(best)
            steps[target]['params'] = merged
            self._pipeline_list.set_steps(steps)
            self._pipeline_list.select_step(target)
            self._pipeline_list.sig_changed.emit()
        else:
            self._param_form.set_values(best)
        InfoBar.success(title='AutoTune 自动调参', content='已采用最优参数',
                        orient=Qt.Orientation.Horizontal, isClosable=True,
                        position=InfoBarPosition.TOP, duration=2000,
                        parent=self)
