"""Adapters for workbench controls provided by Fluent Widgets 1.11.2."""
from functools import wraps

from PyQt6.QtCore import QPropertyAnimation, Qt
from qfluentwidgets import MaskDialogBase, MenuAnimationType, RoundMenu, SwitchButton
from qfluentwidgets.common.animation import BackgroundAnimationWidget

from ui import motion
from ui.motion import finish_animation


def patch_background_motion():
    """Use immediate Fluent hover and press feedback when motion is reduced."""
    update = BackgroundAnimationWidget._updateBackgroundColor
    if getattr(update, '_mygpr_motion', False):
        return

    @wraps(update)
    def apply(self):
        update(self)
        if not motion.animations_enabled():
            finish_animation(self.backgroundColorAni)

    apply._mygpr_motion = True
    BackgroundAnimationWidget._updateBackgroundColor = apply


class MotionSwitchButton(SwitchButton):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.indicator.toggled.connect(self._on_toggle)

    def _on_toggle(self, _checked):
        if not motion.animations_enabled():
            self.settle_motion()

    def settle_motion(self):
        finish_animation(self.indicator.slideAni)


def _settle_menu(menu):
    manager = menu.aniManager
    if manager is not None:
        finish_animation(getattr(manager, 'aniGroup', manager.ani))


def patch_menu_motion():
    if hasattr(RoundMenu, 'settle_motion'):
        return
    original_exec = RoundMenu.exec

    @wraps(original_exec)
    def show(self, pos, ani=True, aniType=MenuAnimationType.DROP_DOWN):
        # Fluent chooses a destination and mask for each direction, including
        # pull-up combo menus at the bottom of a screen. Its ani flag is ignored.
        # Finish synchronously before the popup's first paint, using that same
        # geometry for mouse, keyboard and submenu paths.
        _settle_menu(self)
        original_exec(self, pos, ani, aniType)
        if not ani or not motion.animations_enabled():
            _settle_menu(self)

    RoundMenu.exec = show
    RoundMenu.settle_motion = _settle_menu


def _settle_dialog(dialog):
    effect = dialog.graphicsEffect()
    if effect is None:
        return
    children = dialog.findChildren(QPropertyAnimation, options=Qt.FindChildOption.FindDirectChildrenOnly)
    for animation in children:
        if animation.targetObject() is effect and animation.propertyName() == b'opacity':
            finish_animation(animation)
            return


def patch_dialog_motion():
    if hasattr(MaskDialogBase, 'settle_motion'):
        return
    original_show = MaskDialogBase.showEvent
    original_done = MaskDialogBase.done

    @wraps(original_show)
    def show(self, event):
        original_show(self, event)
        if not motion.animations_enabled():
            _settle_dialog(self)

    @wraps(original_done)
    def done(self, code):
        original_done(self, code)
        if not motion.animations_enabled():
            _settle_dialog(self)

    MaskDialogBase.showEvent = show
    MaskDialogBase.done = done
    MaskDialogBase.settle_motion = _settle_dialog
