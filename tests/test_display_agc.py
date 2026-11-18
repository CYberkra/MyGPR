"""Numerical contract for the display-only limited gain."""

import numpy as np

from mygpr.domain.processing.display_agc import apply_display_agc


def test_display_agc_matches_centered_window_and_keeps_input():
    raw = np.array([[8., 0.04], [-4., 0.02], [1., -0.04], [0., 0.02], [2., 0.04]])
    raw.setflags(write=False)
    ceiling = 4.
    padded = np.pad(raw**2, ((1, 1), (0, 0)), mode="edge")
    local_rms = np.array([np.sqrt(padded[index:index + 3].mean(axis=0)) for index in range(len(raw))])
    reference = np.sqrt(np.mean(raw**2))
    expected = raw * np.clip(reference / local_rms, 1., ceiling)

    shown = apply_display_agc(raw, 3, 20 * np.log10(ceiling))
    np.testing.assert_allclose(shown, expected)
    np.testing.assert_allclose(shown[:, 1], raw[:, 1] * ceiling)
    assert shown[0, 0] == raw[0, 0] and shown[3, 0] == 0
    assert not raw.flags.writeable


def test_display_agc_handles_zero_or_missing_power_and_short_traces():
    zero = np.zeros((5, 2))
    assert apply_display_agc(zero, 21, 18.) is zero
    short = np.array([[2., np.nan, np.inf]])
    assert apply_display_agc(short, 21, 0.) is short
    np.testing.assert_array_equal(apply_display_agc(short, 21, 18.), short)
