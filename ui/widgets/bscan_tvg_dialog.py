# -*- coding: utf-8 -*-
"""TvgGainDialog — TVG 显示增益滑条面板（非模态）。

两根滑条（深端总增益 dB、曲线弯度幂次）拖动**实时生效**——画布上直接看
效果；关闭面板时经视图 notify 路径持久化一次（拖动过程不写盘，防高频
刷盘）。面板生命周期归视图：视图销毁随之销毁；重复打开只唤起既有实例
（见 BScanView._open_tvg_dialog）。

用户在面板打开期间经右键切到其他增益模式时，滑条改动自动失效
（gain_mode 守卫），关面板也不会把模式改回 TVG。
"""
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QDialog, QGridLayout

from qfluentwidgets import BodyLabel, Slider

_DB_MAX = 60                    # 深端总增益上限（dB）
_DB_STEP = 10                   # 滑条分辨率 0.1 dB
_POWER_MIN, _POWER_MAX = 30, 300  # 弯度 0.3~3.0（滑条值 ×100）


class TvgGainDialog(QDialog):
    """TVG 参数滑条面板：拖动 → 实时应用；关闭 → 持久化一次。"""

    def __init__(self, view) -> None:
        super().__init__(view)
        self._view = view
        self.setWindowTitle('TVG 显示增益')
        self.setModal(False)
        self.setWindowFlag(Qt.WindowType.Tool)      # 不进任务栏，随主窗
        self.setMinimumWidth(320)

        grid = QGridLayout(self)
        grid.setContentsMargins(16, 16, 16, 16)
        grid.setVerticalSpacing(10)

        grid.addWidget(BodyLabel('深端总增益 (dB)', self), 0, 0)
        self._db_slider = Slider(Qt.Orientation.Horizontal, self)
        self._db_slider.setRange(0, _DB_MAX * _DB_STEP)
        self._db_slider.setValue(int(round(view._gain_db * _DB_STEP)))
        grid.addWidget(self._db_slider, 1, 0)
        self._db_value = BodyLabel(self._fmt_db(), self)
        self._db_value.setMinimumWidth(56)
        grid.addWidget(self._db_value, 1, 1)

        grid.addWidget(BodyLabel('曲线弯度（>1 集中深部）', self), 2, 0)
        self._power_slider = Slider(Qt.Orientation.Horizontal, self)
        self._power_slider.setRange(_POWER_MIN, _POWER_MAX)
        self._power_slider.setValue(int(round(view._gain_power * 100)))
        grid.addWidget(self._power_slider, 3, 0)
        self._power_value = BodyLabel(self._fmt_power(), self)
        self._power_value.setMinimumWidth(56)
        grid.addWidget(self._power_value, 3, 1)

        self._db_slider.valueChanged.connect(self._on_changed)
        self._power_slider.valueChanged.connect(self._on_changed)
        self.finished.connect(self._on_finished)

    def _fmt_db(self) -> str:
        return f'{self._db_slider.value() / _DB_STEP:.1f}'

    def _fmt_power(self) -> str:
        return f'{self._power_slider.value() / 100:.2f}'

    def _on_changed(self) -> None:
        """拖动实时生效（不写盘，防高频刷盘）；模式被切走时滑条失效。"""
        self._db_value.setText(self._fmt_db())
        self._power_value.setText(self._fmt_power())
        if self._view.gain_mode() != 'tvg':
            return
        self._view.set_gain('tvg',
                            db=self._db_slider.value() / _DB_STEP,
                            power=self._power_slider.value() / 100,
                            notify=False)

    def _on_finished(self, _result: int) -> None:
        """关面板持久化一次（模式仍是 TVG 才发，防把用户切走的模式改回）。"""
        if self._view.gain_mode() == 'tvg':
            self._view.set_gain('tvg',
                                db=self._db_slider.value() / _DB_STEP,
                                power=self._power_slider.value() / 100,
                                notify=True)
