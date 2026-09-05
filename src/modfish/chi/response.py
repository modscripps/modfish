"""Power transfer functions applied to the microconductivity PSD.

Each factor is applied exactly once, by name (spec decision 1). The
preemphasis network follows `remove_sbe_preemphasisMHA.m` with pi in place
of its `PI = 3.14159`; only the time constant `(R24 + R25) C19` = 1.577 s
reaches the chi band.
"""

import numpy as np

from modfish.chi.config import ANTIALIAS_KINDS


def preemphasis_response(f, R24, R25, R22, C19):
    """Complex response H_pre(f) of the SBE 7 preemphasis network.

    Parameters
    ----------
    f : array_like
        Frequency, Hz.
    R24, R25, R22 : float
        Ohm.
    C19 : float
        Farad.

    Returns
    -------
    numpy.ndarray
        Complex response, 1 at f = 0.
    """
    f = np.asarray(f, dtype=float)
    Rf = R24 + R25
    H = np.ones(f.shape, dtype=complex)
    nz = f != 0
    w1 = 1.0 / (2 * np.pi * f[nz] * C19)
    denom = R22 * R22 + w1 * w1
    H[nz] = (1 + Rf * R22 / denom) + 1j * (Rf * w1 / denom)
    return H


def preemphasis_inverse(f, R24, R25, R22, C19):
    """1 / |H_pre(f)|^2, the PSD factor that removes the preemphasis."""
    return 1.0 / np.abs(preemphasis_response(f, R24, R25, R22, C19)) ** 2


def antialias(f, fs, kind):
    """Power transfer of the antialias filter at frequency `f`.

    Parameters
    ----------
    f : array_like
        Frequency, Hz.
    fs : float
        Sampling rate, Hz.
    kind : str
        "som_sinc4": the SOM ADC sinc^4 amplitude filter of
        `get_filters_SOM.m`, power `sinc^8(pi f / fs)`.
        "ap00_sinc2": Alford and Pinkel (2000) `sinc^2(pi k / k_N)`, which
        in frequency is `sinc^2(2 pi f / fs)`. `sinc(x) = sin(x)/x`.

    Returns
    -------
    numpy.ndarray
        Power transfer, 1 at f = 0.
    """
    f = np.asarray(f, dtype=float)
    if kind == "som_sinc4":
        return np.sinc(f / fs) ** 8  # numpy sinc(x) = sin(pi x)/(pi x)
    if kind == "ap00_sinc2":
        return np.sinc(2 * f / fs) ** 2
    raise ValueError(f"antialias must be one of {ANTIALIAS_KINDS}, got {kind!r}")


def derivative(k):
    """(2 pi k)^2, the spectral derivative for `k` in cpm."""
    k = np.asarray(k, dtype=float)
    return (2 * np.pi * k) ** 2
