"""Keep Fluent's private Windows styles alive through native widget teardown."""
from functools import wraps

from PyQt6 import sip
from PyQt6.QtWidgets import QApplication
from qfluentwidgets.common import style_sheet
from qfluentwidgets.components.widgets import command_bar, line_edit, table_view, tree_view


def patch_dynamic_style_lifetime() -> None:
    """Preserve Fluent's platform choice while giving private styles an owner.

    Fluent 1.11.2 supplies a parentless WindowsVista style on Windows/Qt >=6.8.
    SIP keeps a Python reference, but that does not guarantee the native base
    survives widget finalization and queued style events. Parent the actual
    base before setStyle installs a QSS proxy; never reparent the app style.
    """
    original = style_sheet.updateDynamicStyle
    if getattr(original, '_mygpr_owned_dynamic_style', False):
        return

    @wraps(original)
    def update(widget):
        missing = object()
        previous_setter = widget.__dict__.get('setStyle', missing)
        set_style = widget.setStyle

        def set_owned_style(style):
            previous_style = widget.__dict__.get('_mygpr_dynamic_style')
            app = QApplication.instance()
            private = (style is not None and style.parent() is None
                       and (app is None or style is not app.style()))
            if private:
                style.setParent(widget)
            set_style(style)
            widget._mygpr_dynamic_style = style if private or style is previous_style else None
            # Fluent also refreshes on border/popup changes. Release replaced
            # bases after installing the new one instead of accumulating them.
            if (previous_style is not None and previous_style is not style
                    and not sip.isdeleted(previous_style) and previous_style.parent() is widget):
                previous_style.deleteLater()

        widget.setStyle = set_owned_style
        try:
            original(widget)
        finally:
            if previous_setter is missing:
                del widget.setStyle
            else:
                widget.setStyle = previous_setter

    update._mygpr_owned_dynamic_style = True
    style_sheet.updateDynamicStyle = update
    # These pinned Fluent modules import the function directly. Updating only
    # style_sheet would leave their already-bound aliases calling the old code.
    for module in (command_bar, line_edit, table_view, tree_view):
        if module.updateDynamicStyle is original:
            module.updateDynamicStyle = update
