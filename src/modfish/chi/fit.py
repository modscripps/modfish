"""Fit the gain times network product against the SBE 49 (spec "Gain fit").

The estimator is the review's stage 3: the square root of the median ratio
of the two conductivity-gradient spectra over 0.05 to 0.5 cpm, where both
sensors resolve the signal.
"""

import dataclasses
import logging

import numpy as np
import pandas as pd
from scipy.signal import welch

from modfish.chi.config import ChiParams
from modfish.chi.load import range_time
from modfish.chi.response import derivative
from modfish.chi.spectra import correct_spectrum

logger = logging.getLogger(__name__)


def fit_gain(c_ctd, fs_ctd, c1, fs_c1, spd, params: ChiParams, band=(0.05, 0.5), nsec=16.0,
             window="blackmanharris"):
    """Gain (S/m per V) of one span from the two gradient spectra.

    Both spectra use the same `window`. The default is Blackman-Harris:
    the SBE 49 conductivity spectrum is red (about f^-3) and a Hamming
    window leaks it into the fit band, biasing the gain by +2.6 % on a
    synthetic pair (+0.6 % with Blackman-Harris); lengthening `nsec`
    makes the Hamming bias worse.

    Parameters
    ----------
    c_ctd : array_like
        SBE 49 conductivity, S/m, at `fs_ctd`.
    fs_ctd : float
        SBE 49 sampling rate, Hz.
    c1 : array_like
        Microconductivity volts at `fs_c1`, same span.
    fs_c1 : float
        Microconductivity sampling rate, Hz.
    spd : float
        Fall rate, m/s.
    params : ChiParams
        Network and antialias parameters; `params.gain` is ignored.
    band : tuple of float
        cpm, fit band.
    nsec : float
        s, Welch segment length for both spectra.
    window : str
        scipy window name for both spectra.

    Returns
    -------
    float
        Gain, S/m per V, the square root of the median ratio of the two
        corrected conductivity-gradient spectra over `band`.

    Raises
    ------
    ValueError
        When fewer than 3 wavenumber bins of `c1` fall inside `band`.
    """
    unit = dataclasses.replace(params, enabled=True, gain=1.0)
    c_ctd = np.asarray(c_ctd, dtype=float)
    c1 = np.asarray(c1, dtype=float)
    n16 = int(round(nsec * fs_ctd))
    f16, P16 = welch(c_ctd, fs=fs_ctd, window=window, nperseg=n16, noverlap=n16 // 2,
                     detrend="linear", scaling="density")
    k16 = f16 / spd
    Pk16 = P16 * spd * derivative(k16)
    n1 = int(round(nsec * fs_c1))
    f1, P1 = welch(c1, fs=fs_c1, window=window, nperseg=n1, noverlap=n1 // 2,
                   detrend="linear", scaling="density")
    k1, Pk1 = correct_spectrum(f1, P1, fs_c1, spd, unit)
    sel = (k1 >= band[0]) & (k1 <= band[1])
    if sel.sum() < 3:
        raise ValueError(f"fewer than 3 wavenumber bins in {band} cpm; lengthen nsec")
    ratio = np.interp(k1[sel], k16, Pk16) / Pk1[sel]
    return float(np.sqrt(np.median(ratio)))


def fit_gain_casts(ctd, c1, ranges: pd.DataFrame, casts, params: ChiParams, band=(0.05, 0.5)):
    """`fit_gain` per down cast of a deployment.

    Parameters
    ----------
    ctd : xr.Dataset
        L1 `ctd` group (`c`, `depth`, `time`, `cast`).
    c1 : numpy.ndarray
        Microconductivity volts, from `load_c1`.
    ranges : pandas.DataFrame
        Range table (`i0`, `n`, `start`, `fs`), from `load_c1`.
    casts : xr.Dataset
        L1 `casts` group.
    params : ChiParams
        Network and antialias parameters; `params.gain` is ignored.
    band : tuple of float
        cpm, fit band.

    Returns
    -------
    pandas.DataFrame
        Columns `cast`, `gain`, `spd`, `n_ctd`, `n_c1`; one row per down
        cast that yielded a fit. A cast is skipped (and, past the
        non-down and not-covered cases, a warning is logged) when: its
        `direction` is not "down"; no single range in `ranges` covers its
        span; the CTD `c` values over the cast span or the selected `c1`
        segment contain a NaN; or `fit_gain` raises because fewer than 3
        wavenumber bins fall inside `band`.
    """
    rows = []
    t_ctd = ctd["time"].values.astype("datetime64[ns]")
    fs_ctd = 1.0 / (np.median(np.diff(t_ctd)).astype("int64") / 1e9)
    for cid, t0, t1, direction in zip(casts["cast"].values, casts["start_time"].values,
                                      casts["end_time"].values, casts["direction"].values):
        if str(direction) != "down":
            continue
        m = (t_ctd >= t0) & (t_ctd <= t1)
        if m.sum() < 10 * fs_ctd:
            continue
        depth = ctd["depth"].values[m]
        seconds = (t_ctd[m][-1] - t_ctd[m][0]).astype("int64") / 1e9
        spd = float(abs(depth[-1] - depth[0]) / seconds)
        for _, r in ranges.iterrows():
            if not np.isfinite(r.fs):
                continue
            tr = range_time(r.start, r.n, r.fs)
            if tr[0] <= t0 and tr[-1] >= t1:
                sel = (tr >= t0) & (tr <= t1)
                seg = c1[int(r.i0):int(r.i0) + int(r.n)][sel]
                c_ctd_vals = ctd["c"].values[m]
                if np.isnan(c_ctd_vals).any() or np.isnan(seg).any():
                    logger.warning(
                        "fit_gain_casts: skipping cast %s, NaN in ctd c or c1 span",
                        cid,
                    )
                    break
                try:
                    g = fit_gain(c_ctd_vals, fs_ctd, seg, float(r.fs), spd, params, band)
                except ValueError as exc:
                    logger.warning("fit_gain_casts: skipping cast %s, %s", cid, exc)
                    break
                rows.append(dict(cast=int(cid), gain=g, spd=spd, n_ctd=int(m.sum()), n_c1=int(sel.sum())))
                break
    return pd.DataFrame(rows, columns=["cast", "gain", "spd", "n_ctd", "n_c1"])
