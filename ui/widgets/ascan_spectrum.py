"""Amplitude spectrum of the currently displayed single trace."""
import numpy as np


def amplitude_spectrum(samples, axis, unit, bounds=None):
    if samples is None or axis is None or unit != 'ns':
        raise ValueError('需要真实且等间距的纳秒时间轴。')
    data = np.asarray(samples, dtype=float).ravel()
    original = np.asarray(axis).ravel()
    time = original.astype(float)
    if data.size != time.size or data.size < 2 or not np.all(np.isfinite(time)):
        raise ValueError('时间轴无效。')
    step = (time[-1] - time[0]) / (time.size - 1)
    grid = time[0] + np.arange(time.size) * step
    precision = np.finfo(original.dtype).eps if original.dtype.kind == 'f' else np.finfo(float).eps
    if original.dtype == np.float64 and np.array_equal(time, time.astype(np.float32).astype(np.float64)):
        precision = np.finfo(np.float32).eps
    tolerance = max(1e-10, 2 * precision * max(1.0, np.max(np.abs(time))))
    if step <= 0 or np.any(np.diff(time) <= 0) or np.max(np.abs(time - grid)) > tolerance:
        raise ValueError('频谱需要等间距递增的时间轴。')
    if bounds is not None:
        mask = (time >= bounds[0]) & (time <= bounds[1])
        data = data[mask]
    if data.size < 2 or not np.all(np.isfinite(data)):
        raise ValueError('选区必须至少包含两个有限采样值。')
    amplitude = np.abs(np.fft.rfft(data)) / data.size
    amplitude[1:-1 if data.size % 2 == 0 else None] *= 2
    return np.fft.rfftfreq(data.size, d=step) * 1000, amplitude
