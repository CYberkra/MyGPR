from types import SimpleNamespace
import numpy as np
import pytest
from PyQt6.QtCore import QMimeData, QUrl


def test_envelope_is_columnwise_and_reversible(qapp):
    from ui.widgets.bscan_view import BScanView
    from scipy.signal import hilbert

    view = BScanView()
    x = np.cos(2 * np.pi * np.arange(128) / 16)[:, None] * np.array([[1.0, 2.0]])
    x = x.astype("float32")
    saved = x.copy()
    try:
        view.set_matrix(x, -2, 2)
        view.set_display_mode("envelope")
        np.testing.assert_allclose(view._image_item.image, np.abs(hilbert(x, axis=0)), atol=1e-6)
        assert view._image_item.levels[0] >= 0
        view.set_contrast(2)
        view.set_display_mode("grayscale")
        np.testing.assert_array_equal(view._image_item.image, x)
        np.testing.assert_array_equal(x, saved)
        assert not (view._wiggle_item and view._wiggle_item.isVisible())
    finally:
        view.close()


def test_contrast_shrinks_levels_preserves_override_and_data(qapp):
    from ui.widgets.bscan_view import BScanView

    view = BScanView()
    a = np.linspace(-2, 2, 80, dtype="float32").reshape(20, 4)
    try:
        view.set_matrix(a, -2, 2)
        lo, hi = view._image_item.levels
        view.set_contrast(2)
        assert np.diff(view._image_item.levels)[0] == pytest.approx((hi - lo) / 2)
        view.set_levels_override((-3, 3))
        view.set_contrast(3)
        np.testing.assert_array_equal(view._image_item.levels, [-3, 3])
        np.testing.assert_array_equal(view._matrix, a)
    finally:
        view.close()


def test_settings_contrast_roundtrip_and_reset(qapp):
    from ui.pages.settings_page import SettingsPage

    page = SettingsPage()
    try:
        page.load_settings({"bscan_contrast": 1.75})
        assert page.bscan_view_settings()["bscan_contrast"] == pytest.approx(1.75)
        page._bscan_contrast_reset.click()
        assert page.bscan_view_settings()["bscan_contrast"] == 1
    finally:
        page.close()


def test_drop_uses_existing_import_signal_and_next_line_id(qapp, tmp_path):
    from ui.pages.project_page import ProjectPage

    p = ProjectPage()
    events = []
    try:
        p._summary = SimpleNamespace(project_id="p", read_only=False)
        p._lines = [SimpleNamespace(line_id="L01")]
        f = tmp_path / "实测.h5"
        f.write_bytes(b"will be preflighted by backend")
        p.import_requested.connect(events.append)
        assert p.import_dropped_file(str(f))
        assert len(events) == 1 and events[0]["line_id"] == "L02"
        assert events[0]["source"] == str(f) and not events[0]["preflight"]
        p._busy = True
        assert not p.import_dropped_file(str(f))
        assert len(events) == 1
    finally:
        p.close()


def test_drop_rejects_remote_directories_multiple_and_unknown(tmp_path):
    from ui.file_drop import dropped_data_file

    f = tmp_path / "数据.H5"
    f.touch()
    mime = QMimeData()
    mime.setUrls([QUrl.fromLocalFile(str(f))])
    assert dropped_data_file(mime) == str(f)
    for urls in [
        [QUrl("https://example.com/data.h5")],
        [QUrl.fromLocalFile(str(tmp_path))],
        [QUrl.fromLocalFile(str(f))] * 2,
    ]:
        mime.setUrls(urls)
        assert dropped_data_file(mime) is None


def test_envelope_missing_column_not_fabricated():
    from core.display_envelope import display_envelope

    x = np.ones((10, 2), dtype="float32")
    x[3, 1] = np.nan
    got = display_envelope(x)
    np.testing.assert_allclose(got[:, 0], 1)
    assert np.isnan(got[:, 1]).all() and np.isnan(x[3, 1])


@pytest.mark.parametrize("samples", [1, 5, 6, 127])
def test_envelope_handles_odd_even_and_single_sample(samples):
    from scipy.signal import hilbert
    from core.display_envelope import display_envelope

    a = np.random.default_rng(samples).normal(size=(samples, 3)).astype("float32")
    np.testing.assert_allclose(display_envelope(a), abs(hilbert(a, axis=0)), atol=5e-7)


def test_external_file_reaches_drop_filter_over_child(qapp, tmp_path):
    from PyQt6.QtCore import QPointF, Qt
    from PyQt6.QtGui import QDropEvent
    from PyQt6.QtWidgets import QWidget
    from ui.file_drop import FileDropFilter

    owner = QWidget()
    child = QWidget(owner)
    other = QWidget()
    got = []
    f = tmp_path / "input.h5"
    f.touch()
    mime = QMimeData()
    mime.setUrls([QUrl.fromLocalFile(str(f))])
    router = FileDropFilter(owner, got.append)
    event = QDropEvent(
        QPointF(4, 4), Qt.DropAction.CopyAction, mime, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier
    )
    try:
        assert not router.eventFilter(other, event)
        assert router.eventFilter(child, event) and event.isAccepted()
        assert got == [str(f)]
    finally:
        other.close()
        owner.close()


def test_contrast_slider_sync_and_settings_no_load_writeback(qapp):
    from ui.pages.settings_page import SettingsPage

    page = SettingsPage()
    signals = []
    page.bscan_view_changed.connect(lambda: signals.append(True))
    try:
        page.load_settings({"bscan_contrast": 2.3})
        assert not signals and page._bscan_contrast_slider.value() == 230
        page._bscan_contrast_slider.setValue(150)
        assert page.bscan_view_settings()["bscan_contrast"] == 1.5 and signals
        signals.clear()
        page.sync_bscan_view_settings({"bscan_contrast": 1.8})
        assert not signals and page._bscan_contrast_slider.value() == 180
    finally:
        page.close()


@pytest.mark.parametrize("valid", [True, False])
def test_drop_payload_runs_real_backend_preflight(qapp, tmp_path, valid):
    from ui.pages.project_page import ProjectPage
    from mygpr.interfaces.backend import MyGPRBackend

    backend = MyGPRBackend.create_default(max_workers=1)
    page = ProjectPage()
    try:
        project = backend.projects.create_project(tmp_path / "project", name="Drop import")
        page._summary = project
        array = np.arange(120, dtype="float32").reshape(30, 4)
        source = tmp_path / ("radar.npy" if valid else "broken.h5")
        if valid:
            np.save(source, array)
        else:
            source.write_bytes(b"invalid HDF5 must be rejected")
        jobs = []
        page.import_requested.connect(
            lambda p: jobs.append(
                backend.submit_line_import(
                    project.project_id,
                    p["source"],
                    line_id=p["line_id"],
                    name=p["name"],
                    dielectric_constant=p["dielectric"],
                )
            )
        )
        assert page.import_dropped_file(str(source))
        result = backend.jobs.wait(jobs[0], timeout=30)
        assert result.status.value == ("completed" if valid else "failed")
        if valid:
            np.testing.assert_array_equal(backend.projects.read_dataset(project.project_id, "L01").data, array)
        else:
            assert not backend.projects.list_lines(project.project_id)
    finally:
        page.close()
        backend.shutdown(wait=True)
