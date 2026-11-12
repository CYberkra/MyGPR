"""Bind gain editing to a frozen project/input/prefix identity."""
from copy import deepcopy
import json
import uuid

from PyQt6 import sip


class ProcessingGainPreview:
    def _gain_preview_target(self):
        if not self._can_edit_chain():
            return None
        if self._picker_context is not None:
            if self._method_browser.current_method_id() != "manual_gain":
                return None
            target = self._picker_context[1]
            if target is not None and target[1] != self._draft_edit_epoch:
                return None
            index = target[0] if target is not None else self._pipeline_list.count()
            return ("library", self._picker_context), index
        index = self._pipeline_list.index_of_step(self._editor_uid)
        steps = self._pipeline_list.steps()
        if index >= 0 and steps[index].get("method_id") == "manual_gain":
            return ("step", self._editor_uid), index
        return None

    def _request_gain_preview(self, dialog):
        target = self._gain_preview_target()
        if not self._project_id or not self._line_id or target is None:
            dialog.preview.set_error("请先选择有效处理输入与手动曲线增益步骤")
            return
        token = uuid.uuid4().hex
        identity, index = target
        prefix = deepcopy(self._pipeline_list.steps()[:index])
        source = self._sources.get(self._selected_source_key, {})
        description = f"{self._line_id} · {source.get('title', '当前输入')}"
        self._gain_preview_requests[token] = {
            "dialog": dialog, "key": self._draft_key(), "epoch": self._activation_epoch,
            "target": identity, "prefix": self._gain_prefix_signature(prefix), "description": description,
        }
        dialog.preview.set_loading(description)
        dialog.finished.connect(lambda *_: self._finish_gain_preview(token))
        dialog.destroyed.connect(lambda *_: self._finish_gain_preview(token))
        self.gain_preview_requested.emit(token, self._project_id, self._line_id,
                                         self._input_artifact_id(), prefix)

    @staticmethod
    def _gain_prefix_signature(prefix):
        return json.dumps(prefix, sort_keys=True, default=str)

    def _gain_preview_is_current(self, request):
        target = self._gain_preview_target()
        if target is None:
            return False
        identity, index = target
        return (request["key"] == self._draft_key() and request["epoch"] == self._activation_epoch
                and identity == request["target"]
                and request["prefix"] == self._gain_prefix_signature(self._pipeline_list.steps()[:index]))

    def set_gain_preview(self, token, payload, error):
        request = self._gain_preview_requests.get(token)
        if request is None:
            return
        dialog = request["dialog"]
        if not self._gain_preview_is_current(request):
            dialog.reject()
            return
        if error:
            dialog.preview.set_error(error)
            return
        bundle = payload["bundle"]
        description = (f"{request['description']} · 已计算 {payload['prefix_count']} 个前置步骤"
                       f" · {bundle.matrix.shape[0]} 采样 × {bundle.matrix.shape[1]} 道")
        if payload.get("nonfinite_count"):
            description += " · 无效点按完整输入均值填充；建议先修复数据再精确比较"
        dialog.preview.set_input(bundle, description)

    def _finish_gain_preview(self, token):
        if self._gain_preview_requests.pop(token, None) is not None and not sip.isdeleted(self):
            self.gain_preview_cancelled.emit(token)

    def _invalidate_gain_previews(self):
        for request in list(self._gain_preview_requests.values()):
            if not self._gain_preview_is_current(request):
                request["dialog"].reject()

    def _close_gain_previews(self):
        for request in list(self._gain_preview_requests.values()):
            request["dialog"].reject()
