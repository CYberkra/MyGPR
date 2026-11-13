"""Project commands for reports and backups."""
from qfluentwidgets import FluentIcon as FIF

from ui.widgets.context_menus import add_action


class ProjectDeliveryMixin:
    def set_delivery_tools(self, tools):
        self._delivery_tools = tools
        tools.set_project_owner(self)
        self.startup.restore_requested.connect(tools.restore_from_backup)

    def _add_delivery_actions(self, menu):
        tools = self._delivery_tools
        if tools is None:
            return
        menu.addSeparator()
        add_action(menu, FIF.DOCUMENT, '生成报告包…', tools.show_report_config, enabled=self._can_write())
        add_action(menu, FIF.FOLDER, '报告历史…', tools.show_report_history)
        add_action(menu, FIF.SAVE, '备份与恢复…', tools.show_backup_tools, enabled=not self._busy)
        menu.addSeparator()
