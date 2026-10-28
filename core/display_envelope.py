"""Display-only analytic envelope of bounded sample-by-trace preview arrays."""
import numpy as np

def display_envelope(matrix):
    data = np.asarray(matrix)
    if data.ndim != 2:
        raise ValueError('Envelope preview requires a 2D matrix')
    out = np.full(data.shape, np.nan, dtype=np.float32)
    n = data.shape[0]
    if n == 0 or data.shape[1] == 0:
        return out
    valid = np.isfinite(data).all(axis=0)
    if not valid.any():
        return out
    multiplier = np.zeros(n)
    multiplier[0] = 1
    if n % 2 == 0:
        multiplier[n // 2] = 1
        multiplier[1:n // 2] = 2
    else:
        multiplier[1:(n + 1) // 2] = 2
    spectrum = np.fft.fft(data[:, valid], axis=0)
    out[:, valid] = np.abs(np.fft.ifft(spectrum * multiplier[:, None], axis=0))
    return out
