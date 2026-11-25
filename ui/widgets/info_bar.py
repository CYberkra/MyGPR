"""Fluent notifications with the workbench motion preference."""
from PyQt6 import sip
from PyQt6.QtCore import Qt
from qfluentwidgets import InfoBar as FluentInfoBar, InfoBarPosition
from qfluentwidgets.components.widgets.info_bar import InfoBarManager

from ui import motion
from ui.motion import finish_animation


class InfoBar(FluentInfoBar):
    @classmethod
    def new(cls, icon, title, content, orient=Qt.Orientation.Horizontal, isClosable=True,
            duration=1000, position=InfoBarPosition.TOP_RIGHT, parent=None):
        # Fluent's factory constructs its base class explicitly.
        bar = cls(icon, title, content, orient, isClosable, duration, position, parent)
        bar.show()
        return bar

    def showEvent(self, event):
        super().showEvent(event)
        if not motion.animations_enabled():
            self.settle_motion()

    def _InfoBar__fadeOut(self):
        # Fluent's existing readability timer calls this slot after the full
        # display duration. Consult the preference again in case it changed.
        if sip.isdeleted(self):
            return
        if motion.animations_enabled():
            super()._InfoBar__fadeOut()
        else:
            self.close()

    def _settle_stack(self):
        if self.position != InfoBarPosition.NONE and self.parent() is not None:
            manager = InfoBarManager.make(self.position)
            for bar in manager.infoBars.get(self.parent(), ()):
                finish_animation(bar.property('slideAni'))
            finish_animation(manager.aniGroups.get(self.parent()))

    def settle_motion(self):
        self._settle_stack()
        finish_animation(self.opacityAni)

    def closeEvent(self, event):
        super().closeEvent(event)
        if not motion.animations_enabled():
            self._settle_stack()
