"""Per-window gradient spectra, corrections, the noise cut and the band
integral (spec "Estimator per window")."""

import gsw
import numpy as np
from scipy.signal import welch

from modfish.chi.config import (
    FLAG_EMPTY,
    FLAG_NOENV,
    FLAG_NOISE,
    FLAG_RAIL,
    FLAG_SLOW,
    ChiParams,
)
from modfish.chi.response import antialias, derivative, preemphasis_inverse


def dtdc(SP, t, p, dt=0.01):
    """dT/dC at fixed practical salinity and pressure, K per S/m.

    Central difference of `gsw.C_from_SP` (mS/cm, divided by 10) over
    `t +- dt`.

    Parameters
    ----------
    SP : array_like
        Practical salinity.
    t : array_like
        In-situ temperature, deg C.
    p : array_like
        Sea pressure, dbar.
    dt : float, optional
        Half-width of the central difference in temperature, K. Default
        0.01.

    Returns
    -------
    numpy.ndarray
        dT/dC, K per S/m.
    """
    SP = np.asarray(SP, dtype=float)
    t = np.asarray(t, dtype=float)
    p = np.asarray(p, dtype=float)
    dcdt = (gsw.C_from_SP(SP, t + dt, p) - gsw.C_from_SP(SP, t - dt, p)) / (2 * dt) / 10.0
    return 1.0 / dcdt


def window_spectrum(x, fs, nsec):
    """Welch PSD of raw volts.

    Hamming segments of `nsec` s, 50 % overlap, per-segment linear
    detrend, one-sided power spectral density.

    Parameters
    ----------
    x : array_like
        Raw conductivity-channel volts.
    fs : float
        Sampling rate, Hz.
    nsec : float
        Welch segment length, s.

    Returns
    -------
    f : numpy.ndarray
        Frequency, Hz.
    Pf : numpy.ndarray
        One-sided power spectral density, V^2/Hz.
    """
    nperseg = int(round(nsec * fs))
    return welch(np.asarray(x, dtype=float), fs=fs, window="hamming", nperseg=nperseg,
                 noverlap=nperseg // 2, detrend="linear", scaling="density",
                 return_onesided=True)


def correct_spectrum(f, Pf, fs, spd, params: ChiParams):
    """Corrected conductivity-gradient spectrum on the wavenumber axis.

    `Phi_dCdz(k) = Phi_raw(f) spd gain^2 (2 pi k)^2 / (|H_pre|^2 A(f))`,
    with `k = f / spd` in cpm. Each named correction (preemphasis
    inversion, antialias compensation, the frequency-to-wavenumber
    spectral derivative) is applied exactly once.

    Parameters
    ----------
    f : array_like
        Frequency, Hz.
    Pf : array_like
        Raw one-sided power spectral density, V^2/Hz.
    fs : float
        Sampling rate, Hz.
    spd : float
        Fall rate, m/s.
    params : ChiParams
        Chi parameters (gain and the preemphasis/antialias constants).

    Returns
    -------
    k : numpy.ndarray
        Wavenumber, cpm.
    Pk : numpy.ndarray
        Corrected conductivity-gradient spectrum, (S/m)^2 m^-2 per cpm.
    """
    f = np.asarray(f, dtype=float)
    k = f / spd
    factor = (spd * params.gain**2 * derivative(k)
              * preemphasis_inverse(f, params.R24, params.R25, params.R22, params.C19)
              / antialias(f, fs, params.antialias))
    return k, np.asarray(Pf, dtype=float) * factor


def noise_kmax(f, Pf, spd, params: ChiParams):
    """First wavenumber above `kmin` where the raw PSD drops below the
    noise floor.

    Parameters
    ----------
    f : array_like
        Frequency, Hz.
    Pf : array_like
        Raw one-sided power spectral density, V^2/Hz.
    spd : float
        Fall rate, m/s.
    params : ChiParams
        Chi parameters (`kmin`, `snr`, `noise_floor`).

    Returns
    -------
    float
        Wavenumber, cpm, of the first bin above `kmin` where `Pf` drops
        below `snr * noise_floor`. `inf` when it never does or when
        `params.snr` is 0 (the cut disabled).
    """
    if params.snr <= 0:
        return np.inf
    k = np.asarray(f, dtype=float) / spd
    below = (k > params.kmin) & (np.asarray(Pf) < params.snr * params.noise_floor)
    idx = np.flatnonzero(below)
    return float(k[idx[0]]) if idx.size else np.inf


def integrate(k, Pk, kmin, kmax, dtdc_val, D):
    """Rectangle-rule band integral of the corrected spectrum.

    `chi = 6 D dTdC^2 sum_{kmin < k < kmax} Pk dk`, over the bins strictly
    inside the band.

    Parameters
    ----------
    k : array_like
        Wavenumber, cpm, uniformly spaced.
    Pk : array_like
        Corrected conductivity-gradient spectrum, (S/m)^2 m^-2 per cpm.
    kmin, kmax : float
        Band limits, cpm.
    dtdc_val : float
        dT/dC, K per S/m.
    D : float
        Thermal diffusivity, m^2/s.

    Returns
    -------
    chi : float
        Temperature variance dissipation rate, K^2/s. NaN when no bin
        survives.
    n_bins : int
        Number of bins strictly inside `(kmin, kmax)`.
    k_hi : float
        Upper edge of the band actually summed, `k_last + dk / 2`, where
        `k_last` is the wavenumber of the last surviving bin. NaN when no
        bin survives.

    Notes
    -----
    The rectangle rule over the interior bins stops half a bin below
    `kmax`; the returned edge is what the closure's resolved fraction must
    be evaluated over, so the two bands agree.
    """
    k = np.asarray(k, dtype=float)
    sel = (k > kmin) & (k < kmax)
    n = int(sel.sum())
    if n == 0:
        return np.nan, 0, np.nan
    dk = k[1] - k[0]
    chi = float(6 * D * dtdc_val**2 * np.nansum(np.asarray(Pk)[sel]) * dk)
    return chi, n, float(k[sel][-1] + dk / 2)


def window_slices(n, fs, params: ChiParams):
    """Start indices and center times of every full window in a range.

    Parameters
    ----------
    n : int
        Number of samples in the range.
    fs : float
        Sampling rate, Hz.
    params : ChiParams
        Chi parameters (`window`, `step`).

    Returns
    -------
    starts : numpy.ndarray of int
        Start sample index of each window, 0-based.
    centers_s : numpy.ndarray of float
        Center time of each window, s from the start of the range.
    """
    nw = int(round(params.window * fs))
    ns = int(round(params.step * fs))
    if n < nw:
        return np.zeros(0, dtype=int), np.zeros(0)
    starts = np.arange(0, n - nw + 1, ns, dtype=int)
    return starts, (starts + nw / 2) / fs


def run_range(c1, fs, spd, dtdc_val, params: ChiParams):
    """Chi of every full window in one gap-free range.

    Parameters
    ----------
    c1 : array_like
        Raw conductivity-channel volts of the range.
    fs : float
        Sampling rate, Hz.
    spd : array_like
        Fall rate, m/s, one value per window in `window_slices` ordering.
        NaN marks a window without environment data.
    dtdc_val : array_like
        dT/dC, K per S/m, one value per window in `window_slices`
        ordering. NaN marks a window without environment data.
    params : ChiParams
        Chi parameters.

    Returns
    -------
    dict
        `chi` : numpy.ndarray of float
            Temperature variance dissipation rate, K^2/s, NaN where the
            window did not yield a value.
        `kmax` : numpy.ndarray of float
            Upper edge of the wavenumber band summed, cpm, NaN where no
            bin survived.
        `n_bins` : numpy.ndarray of int
            Number of wavenumber bins summed.
        `flag` : numpy.ndarray of uint8
            Per-window flag bits (see `modfish.chi.config`).
        One entry per window.

    Raises
    ------
    ValueError
        When `spd` or `dtdc_val` does not have one entry per window.
    """
    # the range stays in its input dtype (float32 from load_c1); each
    # window is cast on its own, so a 12 h single-range deployment does not
    # carry a 680 MB float64 copy beside the column
    c1 = np.asarray(c1)
    starts, _ = window_slices(c1.size, fs, params)
    nw = int(round(params.window * fs))
    nwin = starts.size
    if len(spd) != nwin or len(dtdc_val) != nwin:
        raise ValueError(
            f"spd ({len(spd)}) and dtdc_val ({len(dtdc_val)}) each need one "
            f"entry per window ({nwin})")
    chi = np.full(nwin, np.nan)
    kmax_out = np.full(nwin, np.nan)
    n_bins = np.zeros(nwin, dtype=int)
    flag = np.zeros(nwin, dtype=np.uint8)
    for j, i0 in enumerate(starts):
        x = np.asarray(c1[i0:i0 + nw], dtype=float)
        if np.any(x <= params.rail_lo) or np.any(x >= params.rail_hi):
            flag[j] |= FLAG_RAIL
        s = spd[j]
        if not np.isfinite(s) or not np.isfinite(dtdc_val[j]):
            flag[j] |= FLAG_NOENV
            continue
        if s < params.min_spd:
            flag[j] |= FLAG_SLOW
            continue
        if np.isnan(x).any():
            flag[j] |= FLAG_NOENV
            continue
        f, Pf = window_spectrum(x, fs, params.nsec)
        k, Pk = correct_spectrum(f, Pf, fs, s, params)
        k_noise = noise_kmax(f, Pf, s, params)
        caps = min(params.kmax_cap, params.fmax_cap / s)
        kmax = min(caps, k_noise)
        if k_noise < caps:
            flag[j] |= FLAG_NOISE
        value, nb, k_hi = integrate(k, Pk, params.kmin, kmax, dtdc_val[j], params.D)
        n_bins[j] = nb
        kmax_out[j] = k_hi  # the band actually summed, for the closure's r
        if nb < params.min_bins:
            flag[j] |= FLAG_EMPTY
            continue
        chi[j] = value
    return dict(chi=chi, kmax=kmax_out, n_bins=n_bins, flag=flag)
