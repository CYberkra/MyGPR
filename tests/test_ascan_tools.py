import pytest
pytest.importorskip('PyQt6')

import numpy as np
from ui.widgets.ascan_popup import AScanPopup


def test_physical_axis_cursors_and_trace_number(qapp):
    popup = AScanPopup()
    popup.show_trace(np.arange(10), trace_index=0, axis=np.linspace(100, 190, 10), axis_label='时间 (ns)')
    view = popup._ascan_view
    x, _ = view._curve.getData()
    assert (x[0], x[-1]) == (100, 190)
    assert '道 1' in view._plot.plotItem.titleLabel.text
    view._cursors[0].setValue(110)
    view._cursors[1].setValue(150)
    assert '40' in view._cursor_label.text() and 'ns' in view._cursor_label.text()


def test_missing_axis_uses_sample_numbers(qapp):
    popup = AScanPopup()
    popup.show_trace(np.arange(4), trace_index=0)
    view = popup._ascan_view
    assert list(view._curve.getData()[0]) == [0, 1, 2, 3]
    assert '采样点' in view._plot.getAxis('bottom').labelText


def test_spectrum_frequency_and_selection():
    from ui.widgets.ascan_spectrum import amplitude_spectrum
    axis = np.arange(1000, dtype=float) * 0.1 + 100
    signal = np.sin(2 * np.pi * 100e-3 * axis)
    frequency, amplitude = amplitude_spectrum(signal, axis, 'ns', (120, 169.9))
    assert abs(frequency[np.argmax(amplitude)] - 100) < 1


def test_spectrum_requires_uniform_time_axis():
    from ui.widgets.ascan_spectrum import amplitude_spectrum
    import pytest
    with pytest.raises(ValueError):
        amplitude_spectrum([1, 2, 3], [0, 1, 3], 'ns')
    with pytest.raises(ValueError):
        amplitude_spectrum([1, 2, 3], [0, 1, 2], '采样点')


def test_float32_time_axis_and_normalized_amplitude():
    from ui.widgets.ascan_spectrum import amplitude_spectrum
    axis = np.linspace(0, 900, 501, dtype=np.float32)
    signal = np.sin(2 * np.pi * (10 / 901.8) * axis)
    frequency, amplitude = amplitude_spectrum(signal, axis, 'ns')
    assert abs(amplitude[np.argmax(amplitude)] - 1) < 1e-4


def test_clear_disables_spectrum_and_clears_data(qapp):
    popup = AScanPopup()
    popup.show_trace(np.arange(10), trace_index=0, axis=np.arange(10), axis_label='时间 (ns)')
    view = popup._ascan_view
    view.clear()
    assert view._samples is None and view._axis is None
    assert not view._spectrum_button.isEnabled()


def test_bundle_to_popup_float32_axis_remains_usable(qapp):
    from core.gui_rendering import PreviewBundle
    from ui.widgets.bscan_view import BScanView
    axis = np.linspace(0, 900, 501, dtype=np.float32)
    data = np.sin(2 * np.pi * (10 / 901.8) * axis)
    bundle = PreviewBundle(data[:, None], -1, 1, 501, 1, sample_axis=axis.tolist(), sample_axis_label='时间 (ns)')
    view = BScanView()
    view.set_bundle(bundle)
    popup = AScanPopup()
    popup.show_trace(view._matrix[:, 0], trace_index=0, axis=view._sample_axis, axis_label=view._sample_axis_label)
    from ui.widgets.ascan_spectrum import amplitude_spectrum
    _, amplitude = amplitude_spectrum(popup._ascan_view._samples, popup._ascan_view._axis, 'ns')
    assert abs(max(amplitude) - 1) < 1e-4


def test_weak_signal_uses_relative_vertical_range(qapp):
    from ui.widgets.ascan_view import AScanView
    view = AScanView()
    signal = np.sin(np.linspace(0, 2 * np.pi, 101)) * 1e-6
    view.set_trace(signal)
    np.testing.assert_array_equal(view._curve.getData()[1], signal)
    lower, upper = view._plot.getViewBox().viewRange()[1]
    assert lower < signal.min() and upper > signal.max()
    assert upper - lower < 3e-6
    view.deleteLater()
    qapp.processEvents()


def test_constant_and_nonfinite_trace_are_safe(qapp):
    from ui.widgets.ascan_view import AScanView
    view = AScanView()
    view.set_trace(np.full(10, 1e-6))
    lower, upper = view._plot.getViewBox().viewRange()[1]
    assert lower < 1e-6 < upper
    view.set_trace([np.nan, np.inf, -np.inf])
    assert view._samples is None
    assert not view._empty_overlay.isHidden()
    view.deleteLater()
    qapp.processEvents()
