"""Current fold widgets respect the existing reduced-motion preference."""
import pytest

pytest.importorskip('PyQt6')
from PyQt6.QtCore import QAbstractAnimation, QCoreApplication, QEvent, QVariantAnimation, Qt
from PyQt6.QtWidgets import (QCheckBox, QComboBox, QHBoxLayout, QLabel,
                             QPushButton, QSpinBox, QVBoxLayout, QWidget)

from ui import motion
from ui.widgets.collapsible_card import CollapsibleCard
from ui.widgets.collapsible_panel import CollapsiblePanel
from ui.widgets.dock_panel import DockPanel


@pytest.fixture(autouse=True)
def drain_fold_deletions(qapp):
    yield
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    qapp.processEvents()


@pytest.mark.parametrize('kind', ['panel', 'card'])
def test_reduced_motion_folding_reaches_final_state_synchronously(qapp, monkeypatch, kind):
    widget = CollapsiblePanel(expand_width=320) if kind == 'panel' else CollapsibleCard('test')
    if kind == 'panel':
        widget.set_content_widget(QLabel('content'))
    else:
        widget.body_layout().addWidget(QLabel('content'))
    widget.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen)
    widget.show()
    qapp.processEvents()
    monkeypatch.setattr(motion, 'animations_enabled', lambda: False)
    states = []
    widget.sig_collapsed.connect(states.append)
    widget.set_collapsed(True)
    assert widget._animation.state() is QAbstractAnimation.State.Stopped
    assert widget._animating is False
    assert states == [True]
    widget.set_collapsed(False)
    assert widget._animation.state() is QAbstractAnimation.State.Stopped
    assert widget._animating is False
    assert states == [True, False]
    assert not (widget._content if kind == 'panel' else widget._body).isHidden()
    widget.deleteLater()


@pytest.mark.parametrize('immediate', ['no_animation', 'reduced_motion'])
def test_dock_page_restore_stops_previous_width_animation(qapp, monkeypatch, immediate):
    import ui.widgets.dock_panel as dock_module
    widget = DockPanel('test', expanded_width=320)
    monkeypatch.setattr(dock_module, 'animations_enabled', lambda: True)
    widget.set_collapsed(True)
    animation = widget._width_anim
    animation.setCurrentTime(70)
    if immediate == 'reduced_motion':
        monkeypatch.setattr(dock_module, 'animations_enabled', lambda: False)
    widget.set_collapsed(False, animate=immediate != 'no_animation')
    assert animation.state() is QAbstractAnimation.State.Stopped
    assert widget.width() == 320
    assert not widget.is_collapsed()
    widget.deleteLater()


def test_dock_reuses_width_animation_across_quick_reversals(qapp, monkeypatch):
    import ui.widgets.dock_panel as dock_module
    widget = DockPanel('test', expanded_width=320)
    monkeypatch.setattr(dock_module, 'animations_enabled', lambda: True)
    for index in range(12):
        widget.set_collapsed(index % 2 == 0)
    assert len(widget.findChildren(QVariantAnimation)) == 1
    widget._width_anim.setCurrentTime(widget._width_anim.duration())
    assert widget.width() == 320
    widget.deleteLater()


def test_card_animates_visible_body_from_actual_height(qapp, monkeypatch):
    monkeypatch.setattr(motion, 'animations_enabled', lambda: True)
    host = QWidget()
    host.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen)
    layout = QVBoxLayout(host)
    widget = CollapsibleCard('test')
    label = QLabel('content')
    label.setFixedHeight(120)
    widget.body_layout().addWidget(label)
    layout.addWidget(widget)
    layout.addStretch()
    host.resize(400, 360)
    host.show()
    qapp.processEvents()
    start_height = widget._body.height()
    widget.set_collapsed(True)
    assert widget._body.isVisibleTo(host)
    assert widget._animation.startValue() == start_height
    widget._animation.setCurrentTime(110)
    qapp.processEvents()
    assert 0 < widget._body.height() < start_height
    widget._animation.setCurrentTime(220)
    assert widget._body.isHidden()
    widget.set_collapsed(False)
    assert widget._body.isVisibleTo(host)
    widget._animation.setCurrentTime(110)
    qapp.processEvents()
    assert 0 < widget._body.height() < start_height
    widget._animation.setCurrentTime(220)
    assert not widget._animating
    assert widget._body.maximumHeight() == 16777215
    host.close()
    host.deleteLater()


def test_card_mid_animation_clips_without_compressing_control_rows(qapp, monkeypatch):
    monkeypatch.setattr(motion, 'animations_enabled', lambda: True)
    host = QWidget()
    host.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen)
    layout = QVBoxLayout(host)
    card = CollapsibleCard('3D controls')
    controls = [QSpinBox(), QCheckBox('Ground line'), QComboBox(), QPushButton('Import')]
    controls[2].addItems(['Elevation', 'Altitude'])
    rows = []
    for index, control in enumerate(controls):
        row = QWidget()
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.addWidget(QLabel(f'Option {index}'))
        row_layout.addWidget(control)
        card.body_layout().addWidget(row)
        rows.append(row)
    layout.addWidget(card)
    layout.addStretch()
    host.resize(400, 420)
    host.show()
    qapp.processEvents()
    expected = [row.geometry() for row in rows]
    natural_height = card.body_height()
    for collapsed in (True, False):
        card.set_collapsed(collapsed)
        card._animation.setCurrentTime(110)
        qapp.processEvents()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.LayoutRequest)
        qapp.processEvents()
        assert 0 < card._body.height() < natural_height
        assert [row.geometry() for row in rows] == expected
        assert all(not left.geometry().intersects(right.geometry())
                   for left, right in zip(rows, rows[1:]))
        assert all(control.height() >= control.minimumSizeHint().height()
                   for control in controls)
        card._animation.setCurrentTime(220)
        qapp.processEvents()
    host.close()
    host.deleteLater()


def test_panel_keeps_content_visible_and_clips_intermediate_width(qapp, monkeypatch):
    monkeypatch.setattr(motion, 'animations_enabled', lambda: True)
    host = QWidget()
    host.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen)
    layout = QHBoxLayout(host)
    widget = CollapsiblePanel(expand_width=320)
    child = QLabel('content')
    child.setFixedWidth(320)
    widget.set_content_widget(child)
    layout.addWidget(widget)
    layout.addWidget(QWidget(), 1)
    host.resize(700, 300)
    host.show()
    qapp.processEvents()
    policy = widget._content.sizePolicy()
    minimum = widget.minimumWidth()
    widget.set_collapsed(True)
    assert widget._content.isVisibleTo(host)
    widget._animation.setCurrentTime(110)
    qapp.processEvents()
    assert 40 < widget.width() < 320
    assert child.width() == 320
    widget._animation.setCurrentTime(220)
    assert widget._content.isHidden()
    assert widget._content.sizePolicy() == policy
    assert widget.minimumWidth() == minimum
    widget.set_collapsed(False)
    assert widget._content.isVisibleTo(host)
    widget._animation.setCurrentTime(110)
    qapp.processEvents()
    assert 40 < widget.width() < 320
    widget.set_collapsed(True, animate=False)
    assert widget._content.isHidden()
    assert widget._content.sizePolicy() == policy
    assert widget.minimumWidth() == minimum
    assert not widget._animating
    host.close()
    host.deleteLater()


@pytest.mark.parametrize('kind', ['panel', 'card'])
@pytest.mark.parametrize('finish', ['reverse', 'repeat_target', 'hide'])
def test_interrupted_fold_restores_constraints_and_requested_state(qapp, monkeypatch, kind, finish):
    monkeypatch.setattr(motion, 'animations_enabled', lambda: True)
    host = QWidget()
    host.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen)
    layout = QVBoxLayout(host)
    widget = CollapsiblePanel(expand_width=320) if kind == 'panel' else CollapsibleCard('test')
    child = QLabel('content')
    child.setFixedSize(280, 120)
    if kind == 'panel':
        widget.set_content_widget(child)
    else:
        widget.body_layout().addWidget(child)
    layout.addWidget(widget)
    layout.addStretch()
    host.resize(400, 350)
    host.show()
    qapp.processEvents()
    content = widget._content if kind == 'panel' else widget._body
    policy = content.sizePolicy()
    minimum = widget.minimumWidth()
    widget.set_collapsed(True)
    widget._animation.setCurrentTime(70)
    qapp.processEvents()
    if finish == 'hide':
        host.hide()
    elif finish == 'reverse':
        widget.set_collapsed(False)
        widget._animation.setCurrentTime(220)
    else:
        widget.set_collapsed(True, animate=False)
    assert widget._animation.state() is QAbstractAnimation.State.Stopped
    assert not widget._animating
    assert widget.is_collapsed() is (finish != 'reverse')
    assert content.sizePolicy() == policy
    assert widget.minimumWidth() == minimum
    assert content.isHidden() is (finish != 'reverse')
    assert child.size().width() == 280
    host.close()
    host.deleteLater()
