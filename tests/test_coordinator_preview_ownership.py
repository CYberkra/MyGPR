"""Preview ownership and the context change after deleting the active line."""
from types import SimpleNamespace as NS

import pytest

from ui.page_coordinator import PageCoordinator


class Page:
    def __init__(self):
        self.calls = []
        self.bundle = None
        self.lines = []

    def set_bundle(self, bundle):
        self.bundle = bundle
        self.calls.append(('set_bundle', (bundle,)))

    def set_lines(self, lines):
        self.lines = list(lines)

    def select_line(self, line_id):
        return any(line.line_id == line_id for line in self.lines)

    def __getattr__(self, name):
        def record(*args, **kwargs):
            self.calls.append((name, args))
        return record


@pytest.fixture
def context():
    pages = {name + 'Interface': Page() for name in
             ['home', 'project', 'processing', 'interpretation', 'delivery', 'spatial']}
    pc = Page()
    state = NS(project_id='P', live_session='S', dirty_checks=0)

    def dirty():
        state.dirty_checks += 1
        return True

    window = NS(_page=pages.get, _current_project_id=lambda: state.project_id,
                _require_project=lambda: True, _infobar=lambda *a: None,
                project_controller=pc, settings=None,
                interpretation_controller=NS(
                    is_current_session=lambda s: s.session_id == state.live_session,
                    close_session=lambda: setattr(state, 'live_session', ''),
                    has_unsaved_changes=dirty))
    co = PageCoordinator(window)
    co.project.current_line_id = 'L01'
    return co, pages, pc, state


def open_artifact(co, artifact_id='A01'):
    co.processing.on_session_opened(NS(
        project_id='P', line_id='L01', session_id='S',
        input_artifact_id=artifact_id, annotation=NS(points=[])))


def test_artifact_session_displays_its_matrix_and_raw_cannot_replace_it(context):
    co, pages, pc, _ = context
    raw, processed = object(), object()
    co.project.on_dataset_preview(raw)
    open_artifact(co)
    assert ('preview_artifact', ('L01', 'A01')) in pc.calls
    assert pages['interpretationInterface'].bundle is None  # loading the new source
    co.project.on_artifact_preview('A01', processed)
    assert pages['interpretationInterface'].bundle is processed
    co.project.on_dataset_preview(raw)
    assert pages['interpretationInterface'].bundle is processed
    assert ('set_original_bundle', (raw,)) in pages['processingInterface'].calls


@pytest.mark.parametrize('changed', ['artifact', 'project', 'line', 'session'])
def test_only_the_live_interpretation_source_receives_artifact_previews(context, changed):
    co, pages, _, state = context
    open_artifact(co)
    pages['interpretationInterface'].bundle = previous = object()
    if changed == 'project':
        state.project_id = 'OTHER'
    elif changed == 'line':
        co.project.current_line_id = 'L02'
    elif changed == 'session':
        state.live_session = 'NEW'
    co.project.on_artifact_preview('OTHER' if changed == 'artifact' else 'A01', object())
    assert pages['interpretationInterface'].bundle is previous


def test_raw_session_still_receives_raw_and_ignores_other_artifacts(context):
    co, pages, _, _ = context
    open_artifact(co, '')
    co.project.on_dataset_preview(raw := object())
    co.project.on_artifact_preview('A01', object())
    assert pages['interpretationInterface'].bundle is raw


def test_deleted_current_line_fully_selects_remaining_line_without_reconfirmation(context):
    co, pages, pc, state = context
    co.project.on_lines_updated([NS(line_id='L02')])
    assert co.project.current_line_id == 'L02'
    assert ('preview_line', ('L02',)) in pc.calls
    assert ('refresh_artifacts', ('L02',)) in pc.calls
    assert ('invalidate_artifact_previews', ()) in pc.calls
    assert state.dirty_checks == 0  # deletion was already confirmed


def test_deleted_last_line_clears_views_and_rejects_late_raw_preview(context):
    co, pages, pc, state = context
    co.project.on_dataset_preview(object())
    co.project.on_lines_updated([])
    for name, setter in [('home', 'set_preview_bundle'),
                         ('processing', 'set_original_bundle'),
                         ('interpretation', 'set_bundle')]:
        assert (setter, (None,)) in pages[name + 'Interface'].calls
    assert ('set_artifacts', ([],)) in pages['processingInterface'].calls
    calls = {key: len(page.calls) for key, page in pages.items()}
    co.project.on_dataset_preview(object())
    assert {key: len(page.calls) for key, page in pages.items()} == calls
    assert co.project.current_line_id == ''
    assert state.dirty_checks == 0
