"""Project-name and path changes without resetting unrelated viewing state."""

import os

from ui import dialogs


class ProjectIdentityChain:
    def __init__(self, chain, page, tree):
        self.chain, self.co, self.page, self.tree = chain, chain._co, page, tree
        self.pc = self.co.project_controller
        self.controller = self.pc.identity
        self.transfer = None
        helper = page._identity
        helper.has_active_tasks = lambda: not self._idle()
        helper.rename_project_requested.connect(self.rename_project)
        helper.rename_line_requested.connect(self.rename_line)
        helper.relocation_requested.connect(self.relocate)
        if tree is not None:
            tree.rename_allowed = lambda: page._can_write() and self._idle()
            tree.line_rename_requested.connect(helper.rename_line)
        self.controller.started.connect(self._started)
        self.controller.progress.connect(self._progress)
        self.controller.finished.connect(self._finished)
        self.controller.error.connect(lambda message: self.co.infobar("warning", "项目操作", message))

    def _idle(self):
        return not (
            self.chain._busy_sources
            or self.chain._active_job_ids
            or self.co.processing.processing_job_id
            or self.co.jobs.import_job_ids
            or self.co.jobs.spatial_job_ids
        )

    def _allowed(self, project_id):
        if str(project_id) != str(self.co.current_project_id() or ""):
            return False
        if not self._idle():
            self.co.infobar("warning", "项目正在使用中", "请等待当前任务结束或取消任务后再操作。")
            return False
        return True

    def rename_project(self, project_id, name):
        if self._allowed(project_id):
            self.controller.rename_project(project_id, name)

    def rename_line(self, project_id, line_id, name):
        if self._allowed(project_id):
            self.controller.rename_line(project_id, line_id, name)

    def relocate(self, project_id, destination):
        if self._allowed(project_id) and self.chain.can_change_project():
            self.controller.relocate_project(project_id, destination)

    def _started(self, operation):
        if operation == "relocate_project":
            self.transfer = dialogs.show_project_transfer(self.page, self.controller.cancel)

    def _progress(self, percent, message):
        if self.transfer is not None:
            self.transfer.setLabelText(message)
            self.transfer.setValue(percent)

    def _remember(self, summary, old_root=""):
        settings = self.co.settings
        if settings is None:
            return
        root = str(summary.root_path)
        names = settings.get("project_display_names", {})
        names = dict(names) if isinstance(names, dict) else {}
        if old_root and old_root != root:
            names.pop(old_root, None)
            for key in ("recent_projects", "pinned_recent_projects"):
                values = settings.get(key, []) or []
                replaced = [
                    root if os.path.normcase(os.path.abspath(p)) == os.path.normcase(os.path.abspath(old_root)) else p
                    for p in values
                ]
                settings.set(key, list(dict.fromkeys(replaced)))
        names[root] = str(summary.name)
        settings.set("project_display_names", names)
        settings.add_recent_project(root)
        settings.save()
        home = self.co.page("homeInterface")
        if home is not None:
            home.refresh_recents()

    def _finished(self, operation, success, error, result, old_root, line_id):
        if self.transfer is not None:
            self.transfer.close()
            self.transfer.deleteLater()
            self.transfer = None
        processing = self.co.page("processingInterface")
        if success and operation == "rename_project":
            self.page.set_project_info(result)
            home = self.co.page("homeInterface")
            if home is not None:
                home.set_current_project(result)
            if self.tree is not None:
                self.tree.set_project_info(result)
            self._remember(result)
        elif success and operation == "rename_line":
            self.page._identity.apply_line_name(result)
            lines = self.page._lines
            if self.tree is not None:
                self.tree.set_lines(lines)
            if processing is not None:
                processing.set_lines(lines)
        elif success and operation == "relocate_project":
            self._remember(result, old_root)
            self.pc._current = None
            self.chain.on_project_closed()
            self.pc._current = result
            self.chain.on_project_opened(result)
            self.pc.refresh_lines()
        if operation == "relocate_project" and not success:
            self.pc.refresh_lines()
            self.pc.refresh_all_artifacts()
            if self.chain.current_line_id:
                self.pc.refresh_artifacts(self.chain.current_line_id)
            self.chain._spatial_reload_pending = True
            if processing is not None and hasattr(processing, "_comparison"):
                processing._preview_requested.clear()
                processing._workspace._pending.clear()
                processing._comparison.restart_pending()
        title = {"rename_project": "项目改名", "rename_line": "测线改名", "relocate_project": "项目目录迁移"}[operation]
        message = (
            ("新位置已打开，旧目录保留：" + old_root if operation == "relocate_project" else "名称已保存。")
            if success
            else error
        )
        self.co.infobar("success" if success else "error", title, message)
