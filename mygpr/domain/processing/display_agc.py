"""Bounded, boost-only AGC for display of sampled traces."""

import numpy as np
from scipy.ndimage import uniform_filter1d


def apply_display_agc(matrix: np.ndarray, window: int, max_db: float) -> np.ndarray:
    """Boost quiet windows toward the section RMS with no gain above max_db.

    The window is measured in samples along each trace. A single working array
    holds power, gain and finally the display pixels. Input amplitudes are kept.
    """
    if not matrix.size or max_db == 0:
        return matrix

    power = np.array(matrix, dtype=np.float64, copy=True)
    low, high = power.min(), power.max()
    if not (np.isfinite(low) and np.isfinite(high)):
        np.nan_to_num(power, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        low, high = power.min(), power.max()
    peak = max(abs(low), abs(high))
    if peak == 0:
        return matrix

    # A common scale cancels in the RMS ratio and prevents overflow on squaring.
    np.divide(power, peak, out=power)
    np.square(power, out=power)
    reference = float(np.sqrt(power.mean()))
    uniform_filter1d(power, size=min(window, matrix.shape[0]), axis=0,
                     output=power, mode="nearest")
    ceiling = 10.0 ** (max_db / 20.0)
    np.maximum(power, (reference / ceiling) ** 2, out=power)
    np.sqrt(power, out=power)
    np.divide(reference, power, out=power)
    np.clip(power, 1.0, ceiling, out=power)
    np.multiply(matrix, power, out=power)
    return power
