# -*- coding: utf-8 -*-
"""OutputPanel（底部输出面板）测试。

覆盖：默认状态、页签切换/收展语义、设置持久化、日志级别解析与着色、
自动滚动锁定、取消信号透传。
收展为瞬时切换（面板高度由主窗口竖向 QSplitter 管理），断言即时终态。
"""
import pytest

pytest.importorskip('PyQt6')

from ui.widgets.output_panel import OutputPanel  # noqa: E402


@pytest.fixture
def panel(qapp):
    p = OutputPanel()
    yield p
    p.deleteLater()


# ============================================================ 页签 / 收展
def test_default_open_on_log_tab(qapp, panel):
    assert panel._open is True
    assert panel._current_tab == 'log'
    assert panel._content.currentIndex() == 0
    assert panel._content.isVisibleTo(panel)


def test_click_active_tab_collapses_content(qapp, panel):
    panel._on_tab_clicked('log')  # 点已激活页签 → 收起
    assert panel._open is False
    assert panel._content.isHidden()

    panel._on_tab_clicked('log')
    assert panel._open is True
    assert panel._content.isVisibleTo(panel)


def test_click_inactive_tab_switches_and_expands(qapp, panel):
    panel._set_open(False, animate=False)
    panel._on_tab_clicked('jobs')  # 点未激活页签 → 切换并展开
    assert panel._open is True
    assert panel._current_tab == 'jobs'
    assert panel._content.currentIndex() == 1
    assert panel._tool_stacked.currentIndex() == 1


def test_toggle_panel_shortcut_path(qapp, panel):
    panel.toggle_panel()
    assert panel._open is False
    panel.toggle_panel()
    assert panel._open is True


# ============================================================ 设置持久化
def test_settings_roundtrip(qapp, tmp_path):
    from ui.settings_manager import SettingsManager
    settings = SettingsManager(str(tmp_path / 's.json'))

    p = OutputPanel()
    p.set_settings_manager(settings)
    p._on_tab_clicked('jobs')
    p._set_open(False, animate=False)
    assert settings.get('output_panel_active_tab') == 'jobs'
    assert settings.get('output_panel_open') is False

    p2 = OutputPanel()
    p2.set_settings_manager(SettingsManager(str(tmp_path / 's.json')))
    assert p2._current_tab == 'jobs'
    assert p2._open is False
    assert p2._content.isHidden()
    p.deleteLater()
    p2.deleteLater()


# ============================================================ 日志：级别 / 搜索
def test_append_log_renders_with_timestamp(qapp, panel):
    panel.append_log('INFO 后端初始化中…')
    text = panel._log_edit.toPlainText()
    assert 'INFO 后端初始化中…' in text
    assert len(panel._entries) == 1
    assert panel._entries[0][0] == 'info'


def test_level_parse_rules(qapp, panel):
    cases = {
        'ERROR 后端初始化失败': 'error',
        'WARNING 磁盘空间不足': 'warning',
        'SUCCESS 任务完成': 'success',
        '普通消息': 'default',
    }
    for msg, level in cases.items():
        panel.clear_log()
        panel.append_log(msg)
        assert panel._entries[-1][0] == level


def test_clear_log_empties_store_and_view(qapp, panel):
    panel.append_log('INFO 一些日志')
    panel.clear_log()
    assert len(panel._entries) == 0
    assert panel._log_edit.toPlainText() == ''


def test_apply_theme_recolors_and_keeps_entries(qapp, panel):
    panel.append_log('ERROR 出错了')
    light_qss = panel._log_edit.styleSheet()
    panel.apply_theme(True)   # 深色
    dark_qss = panel._log_edit.styleSheet()
    assert dark_qss != light_qss
    assert len(panel._entries) == 1
    assert '出错了' in panel._log_edit.toPlainText()
    panel.apply_theme(False)  # 回到浅色
    assert panel._log_edit.styleSheet() == light_qss


# ============================================================ 自动滚动锁定
def _show_panel(qapp, panel):
    """离屏环境下做真实布局，QTextEdit 滚动条才有非零行程。

    用 WA_DontShowOnScreen 而非原生 show：布局照常生效，但不创建原生
    窗口——offscreen 平台下 qfluentwidgets 部分控件（如 TreeWidget）
    反复创建原生窗口会触发库级 access violation，本测试无需原生窗口。
    """
    from PyQt6.QtCore import Qt
    panel.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, True)
    panel.resize(800, 300)
    panel.show()
    qapp.processEvents()


def test_auto_scroll_pauses_when_user_scrolls_up(qapp, panel):
    _show_panel(qapp, panel)
    for i in range(100):
        panel.append_log(f'INFO line {i}')
    qapp.processEvents()
    scrollbar = panel._log_edit.verticalScrollBar()
    assert panel._auto_scroll is True

    scrollbar.setValue(0)  # 用户上翻 → 暂停
    assert panel._auto_scroll is False
    assert panel._auto_scroll_btn.isChecked() is False

    scrollbar.setValue(scrollbar.maximum())  # 滚回底部 → 恢复
    assert panel._auto_scroll is True
    assert panel._auto_scroll_btn.isChecked() is True


def test_auto_scroll_toggle_button(qapp, panel):
    _show_panel(qapp, panel)
    for i in range(100):
        panel.append_log(f'INFO line {i}')
    qapp.processEvents()
    panel._auto_scroll_btn.setChecked(False)
    assert panel._auto_scroll is False
    scrollbar = panel._log_edit.verticalScrollBar()
    value_before = scrollbar.value()
    panel.append_log('INFO 新日志不打断阅读')
    qapp.processEvents()
    assert scrollbar.value() == value_before

    panel._auto_scroll_btn.setChecked(True)
    assert panel._auto_scroll is True
    assert scrollbar.value() == scrollbar.maximum()


# ============================================================ 任务信号透传
def test_cancel_job_signal_wiring(qapp, panel):
    got = []
    panel.cancel_job_requested.connect(got.append)
    panel.mini_jobs().cancel_requested.emit('J1')
    assert got == ['J1']
