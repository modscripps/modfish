"""
T-C sensor response corrections for SBE49-carrying MOD profilers.

Fish-agnostic: the sampling interval is read from the data and every
rate- or fall-speed-dependent parameter is an explicit argument. Default
parameter values are documented placeholders; choosing them is the
subject of the T-C correction analysis (FCTD reprocessing sub-project 3).

Lineage: consolidated from gvpy.mod (which extended ctdproc's dual-sensor
implementation to the single-sensor FCTD). ctdproc reference for the
NumPy 2 reshape fix: commit 5e75198.
"""

import logging

import numpy as np
import xarray as xr
from scipy import fft, optimize, signal, stats
from scipy.signal import butter, filtfilt

logger = logging.getLogger(__name__)


def _butter_lowpass(lowcut, fs, order=3):
    """Design a butterworth low-pass filter.

    Parameters
    ----------
    lowcut : float
        Cut-off frequency in units of `fs`.
    fs : float
        Sampling frequency.
    order : int, optional
        Filter order. Defaults to 3.

    Returns
    -------
    b, a : numpy.ndarray
        Numerator and denominator polynomials of the IIR filter.
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    b, a = butter(order, low, btype="lowpass")
    return b, a


def lowpassfilter(x, lowcut, fs, order=3, axis=-1):
    """Low-pass filter a signal using a butterworth filter.

    The filter is applied forward and backward (`scipy.signal.filtfilt`) so
    the output has zero phase shift relative to the input.

    Parameters
    ----------
    x : array-like
        Time series.
    lowcut : float
        Cut-off frequency in units of `fs`.
    fs : float
        Sampling frequency.
    order : int, optional
        Filter order. Defaults to 3.
    axis : int, optional
        Axis of `x` along which the filter is applied. Defaults to -1.

    Returns
    -------
    lpx : numpy.ndarray
        Low-pass filtered time series.

    Notes
    -----
    For example, if sampling four times per hour, `fs=4`. A cut-off period of
    24 hours is then expressed as `lowcut=1/24`.
    """
    b, a = _butter_lowpass(lowcut, fs, order=order)
    lpx = filtfilt(b, a, x, axis=axis)
    return lpx


def _fill_gaps(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fill NaN in a 1-D array so it can pass through a filter or an FFT.

    Interior NaN are filled by linear interpolation against the sample
    index; NaN at either edge are filled with the nearest finite value
    (constant extrapolation), since there is nothing to interpolate
    between. A single `numpy.interp` call gives both: values outside the
    range of its `xp` (the finite indices) are clamped to the boundary
    `fp` values, which is exactly edge-hold.

    Parameters
    ----------
    x : numpy.ndarray
        One-dimensional array, possibly containing NaN.

    Returns
    -------
    filled : numpy.ndarray
        Copy of `x` with every NaN replaced.
    was_nan : numpy.ndarray
        Boolean mask, same shape as `x`, True where `x` was NaN. Callers
        use this to restore NaN after the finite-input step (filtering,
        FFT) that required this function.

    Notes
    -----
    An all-NaN input has no finite value to interpolate from or clamp
    to; `numpy.interp` raises in that case, so this function does too.
    """
    x = np.asarray(x, dtype=float)
    mask = np.isnan(x)
    filled = x.copy()
    if mask.any():
        idx = np.arange(x.size)
        filled[mask] = np.interp(idx[mask], idx[~mask], x[~mask])
    return filled, mask


def add_tcfit_default(ds):
    """Set a default pressure range for the T-C phase fit.

    The range over which the thermistor time constant and the sensor lag are
    fit is chosen from the maximum pressure of the cast. Deep casts start the
    fit deeper to avoid the strongly stratified and often spiky near-surface
    layer.

    ==================  ===============================
    maximum pressure    fit range
    ==================  ===============================
    > 1000 dbar         500 dbar to maximum pressure
    > 300 dbar          200 dbar to maximum pressure
    otherwise           50 dbar to maximum pressure
    ==================  ===============================

    Parameters
    ----------
    ds : xarray.Dataset
        CTD time series with pressure in variable `p` [dbar].

    Returns
    -------
    ds : xarray.Dataset
        Same Dataset with `ds.attrs["tcfit"]` set to a two-element list
        holding the upper and lower pressure limit for the fit in
        `phase_correct`.

    Notes
    -----
    The attribute is set in place on `ds`; the returned object is the input
    object, not a copy.
    """
    if ds.p.max() > 1000:
        tcfit = [500, ds.p.max().data]
    elif ds.p.max() > 300:
        tcfit = [200, ds.p.max().data]
    else:
        tcfit = [50, ds.p.max().data]
    ds.attrs["tcfit"] = tcfit
    return ds


def atanfit(x, f, Phi, W):
    """Cost function for the thermistor time constant and sensor lag fit.

    The model for the cross-spectral phase between temperature and
    conductivity is

    .. math::

        \\Phi(f) = -\\arctan(2 \\pi f \\tau) - 2 \\pi f L

    with thermistor time constant :math:`\\tau` and lag :math:`L` of
    temperature behind conductivity. The cost is the coherence-weighted sum of
    squared residuals of this model, minimized over `x = (tau, L)`.

    Parameters
    ----------
    x : array-like
        Two-element parameter vector `(tau, L)` in seconds.
    f : numpy.ndarray
        Frequency vector [Hz].
    Phi : numpy.ndarray
        Observed cross-spectral phase [rad], same shape as `f`.
    W : numpy.ndarray
        Square weight matrix, `numpy.diag` of the squared coherence. It enters
        as `W**4`, so the weights act as coherence to the fourth power.

    Returns
    -------
    float
        Weighted sum of squared phase residuals.
    """
    f = np.arctan(2 * np.pi * f * x[0]) + 2 * np.pi * f * x[1] + Phi
    f = np.matmul(np.matmul(f.transpose(), W**4), f)
    return f


def phase_correct(ds, N=128, f0=6.0, tcfit=None, return_spectra=False):
    """Bring temperature and conductivity in phase.

    The thermistor lags conductivity both because of its own finite response
    time and because of the physical separation of the two sensors. Both are
    estimated from the cross-spectral phase between temperature and
    conductivity over a pressure range chosen to be well below the surface
    layer, then removed from temperature with a transfer function applied in
    the frequency domain. Temperature, conductivity and pressure are low-pass
    filtered in the same step.

    Parameters
    ----------
    ds : xarray.Dataset
        CTD time series with variables `t`, `c`, `p`, `lon`, `lat` and `dPdt`
        on an evenly sampled `time` coordinate of dtype datetime64. The
        sampling interval is read from `time`, not assumed.
    N : int, optional
        Number of points per fit segment. Defaults to 128 (2**7), the value
        used for 16 Hz SBE49 data in the cruise-1 processing notebook. gvpy
        used 2**6 for 16 Hz and 2**9 for 24 Hz SBE9/11 data.
    f0 : float, optional
        Cut-off frequency [Hz] of the low-pass filter `1 / (1 + (f/f0)**6)`
        applied to `t`, `c` and `p`. Defaults to 6.0, the gvpy value, which
        was picked for 24 Hz data; increasing it filters less. The orphaned
        copy of this code in `modfish.utils` used 9 with a stale comment. The
        right value for 16 Hz FCTD data is an open question for the T-C
        correction analysis (FCTD reprocessing sub-project 3).
    tcfit : tuple or None, optional
        Upper and lower pressure limit [dbar] of the range used for the phase
        fit. Overrides `ds.attrs["tcfit"]`. If neither is given, defaults from
        `add_tcfit_default` are used.
    return_spectra : bool, optional
        If True, also return a Dataset with the spectra, coherence and phase
        before and after the correction. Defaults to False.

    Returns
    -------
    out : xarray.Dataset
        `t`, `c`, `p`, `lon`, `lat` and `dPdt` on the trimmed segment time
        axis, carrying the input attributes plus the fitted `tau1` and `L1`.
    spectra : xarray.Dataset
        Only if `return_spectra` is True. Autospectra `Et`, `Ec` and their
        post-correction counterparts `Et_corrected`, `Ec_corrected`, squared
        coherence `coh`, `coh_corrected` and cross-spectral phase `phase`,
        `phase_corrected`, all on frequency dimension `f` [Hz]. Attributes
        carry the fit results `tau1`, `L1`, the degrees of freedom `dof` and
        the 95% significance level for the coherence, `beta`.

    Notes
    -----
    Sign convention: the correction multiplies the segment spectra of `t` by

    .. math::

        H_1(f) = (1 + i 2 \\pi f \\tau_1) \\exp(+ i 2 \\pi f L_1)

    With the numpy FFT convention a factor `exp(+i 2 pi f L)` advances the
    signal by `L`, so a temperature record that physically lags conductivity
    by `L` fits a positive `L1`. `tau1` is the thermistor time constant in
    seconds.

    The cross-spectral phase is folded by 2 pi wherever it comes out positive
    before the fit. Without that fold the fit is pulled apart by the phase
    wrapping past -pi at high frequency. The fold undoes exactly one turn, so
    it only works while the total phase stays inside 2 pi over the resolved
    band. At 16 Hz, where the highest resolved frequency is just under 8 Hz,
    that caps the lag at roughly 0.1 s. Beyond it the frequencies past the
    -2 pi crossing come out one turn short and bias the fit, unless the
    coherence has already collapsed there, which is the usual case for real
    records and the reason this has not bitten the algorithm in practice.
    `tests/test_tc.py` pins this limit.

    The `tcfit` pressure range is turned into the contiguous index span from
    the first to the last sample inside the pressure window. That assumes a
    record without large pressure reversals inside the fit range; a yo-yo
    profile spanning the window several times would pull all the intervening
    samples into the fit. The correction itself is applied to the span from
    the first to the last sample deeper than 1 dbar.

    Segments overlap by 50%; the reconstruction keeps the middle half of each
    segment, so the output time axis is the fit-application span trimmed by
    `N/4` samples at each end.

    Data must not contain NaNs. Despiking and gap filling belong upstream of
    this function.
    """
    # Sampling interval read from the time axis rather than assumed. 16 Hz for
    # the SBE49 on the FCTD, 24 Hz for the SBE9/11 the algorithm came from.
    dt = float(np.median(np.diff(ds.time.data)) / np.timedelta64(1, "s"))

    # Fit range. Explicit argument wins over the dataset attribute.
    if tcfit is None:
        if "tcfit" in ds.attrs:
            tcfit = ds.attrs["tcfit"]
        else:
            # Work on a copy so the caller's Dataset is left alone.
            ds = add_tcfit_default(ds.copy())
            tcfit = ds.attrs["tcfit"]
            logger.info("no tcfit given, using default %s", tcfit)

    # ---Spectral analysis of raw data---
    # Only data within the tcfit pressure range. Note that this takes the
    # contiguous index span between the first and the last sample inside the
    # window, see Notes.
    ii = np.squeeze(np.argwhere((ds.p.data > tcfit[0]) & (ds.p.data < tcfit[1])))
    i1 = ii[0]
    i2 = ii[-1]
    n = i2 - i1 + 1
    n = (np.floor(n / N) * N).astype("int64")
    # Truncate to be multiple of N elements long
    i2 = (i1 + n).astype("int64")
    # number of segments = dof/2
    m = (n / N).astype("int64")
    logger.info("%d segments", m)
    # Frequency resolution at 2*m degrees of freedom.
    df = 1 / (N * dt)

    # fft of each segment (row). Data are detrended, then windowed.
    window = signal.windows.triang(N) * np.ones((m, N))
    At1 = fft.fft(signal.detrend(np.reshape(ds.t.data[i1:i2], shape=(m, N))) * window)
    Ac1 = fft.fft(signal.detrend(np.reshape(ds.c.data[i1:i2], shape=(m, N))) * window)

    # Positive frequencies only
    At1 = At1[:, 0 : int(N / 2)]
    Ac1 = Ac1[:, 0 : int(N / 2)]

    # Frequency
    f = fft.ifftshift(np.linspace(-N / 2, N / 2 - 1, N) / N / dt)
    f = f[: int(N / 2)]

    # Spectral estimates. Note: In Matlab, At1*conj(At1) is not complex
    # anymore. Here, it is still a complex number but the imaginary part is
    # zero. We keep only the real part to stay consistent.
    Et1 = 2 * np.real(np.nanmean(At1 * np.conj(At1) / df / N**2, axis=0))
    Ec1 = 2 * np.real(np.nanmean(Ac1 * np.conj(Ac1) / df / N**2, axis=0))

    # Cross spectral estimates
    Ct1c1 = 2 * np.nanmean(At1 * np.conj(Ac1) / df / N**2, axis=0)

    # Squared coherence estimates
    Coht1c1 = np.real(Ct1c1 * np.conj(Ct1c1) / (Et1 * Ec1))

    # Cross-spectral phase estimates
    Phit1c1 = np.arctan2(np.imag(Ct1c1), np.real(Ct1c1))

    # ---Determine tau and L---
    # tau is the thermistor time constant (sec)
    # L is the lag of t behind c due to sensor separation (sec)
    # Matrix of weights based on squared coherence.
    W1 = np.diag(Coht1c1)
    # Shift phase by 2*pi to undo the wrap past -pi at high frequency. This is
    # not being done in the ctdproc package, however, the fit looks funky if
    # not folding over by 2*pi.
    Phit1c1[Phit1c1 > 0] = Phit1c1[Phit1c1 > 0] - 2 * np.pi
    # Fit
    x1 = optimize.fmin(func=atanfit, x0=[0, 0], args=(f, Phit1c1, W1), disp=False)

    tau1 = x1[0]
    L1 = x1[1]

    logger.info("tau = %1.4fs, lag = %1.4fs", tau1, L1)

    # ---Apply phase correction and low-pass filter---
    ii = np.squeeze(np.argwhere(ds.p.data > 1))
    i1 = ii[0]
    i2 = ii[-1]
    n = i2 - i1 + 1
    n = (np.floor(n / N) * N).astype("int64")
    # Truncate to be multiple of N elements long
    i2 = (i1 + n).astype("int64")
    # number of segments = dof/2
    m = (n / N).astype("int64")

    # Transfer function
    f = fft.ifftshift(np.linspace(-N / 2, N / 2 - 1, N) / N / dt)
    H1 = (1 + 1j * 2 * np.pi * f * tau1) * np.exp(1j * 2 * np.pi * f * L1)

    # Low pass filter. The exponent 6 goes with f0=6 for 24 Hz data;
    # decreasing it to 3 leads to lots of noise.
    LP = 1 / (1 + (f / f0) ** 6)

    # Restructure data with overlapping segments.
    # Staggered segments
    variables = ["t", "c", "p"]
    vard = {}
    for v in variables:
        if v in ds:
            vard[v] = np.zeros((2 * m - 1, N))
            vard[v][: 2 * m - 1 : 2, :] = np.reshape(ds[v].data[i1:i2], shape=(m, N))
            vard[v][1::2, :] = np.reshape(
                ds[v].data[i1 + int(N / 2) : i2 - int(N / 2)],
                shape=(m - 1, N),
            )

    time = ds.time[i1:i2]
    lon = ds.lon[i1:i2]
    lat = ds.lat[i1:i2]

    # FFTs of staggered segments (each row)
    Ad = {}
    for v in variables:
        if v in ds:
            Ad[v] = fft.fft(vard[v])

    # Corrected Fourier transforms of temperature.
    Ad["t"] = Ad["t"] * ((H1 * LP) * np.ones((2 * m - 1, 1)))

    # Low pass filter the remaining variables
    for v in ["c", "p"]:
        if v in ds:
            Ad[v] = Ad[v] * (LP * np.ones((2 * m - 1, 1)))

    # Inverse transforms of corrected temperature and low passed other
    # variables. Only the middle half of each segment is kept.
    Adi = {}
    for v in variables:
        if v in ds:
            Adi[v] = np.real(fft.ifft(Ad[v]))
            Adi[v] = np.squeeze(
                np.reshape(Adi[v][:, int(N / 4) : (3 * int(N / 4))], shape=(1, -1))
            )

    time = time[int(N / 4) : -int(N / 4)]
    lon = lon[int(N / 4) : -int(N / 4)]
    lat = lat[int(N / 4) : -int(N / 4)]

    # Generate output structure. Copy attributes over.
    out = xr.Dataset(coords={"time": time})
    out.attrs = dict(ds.attrs)
    out["lon"] = lon
    out["lat"] = lat
    out["dPdt"] = ds.dPdt
    for v in variables:
        if v in ds:
            out[v] = xr.DataArray(Adi[v], coords=(out.time,))
            out[v].attrs = ds[v].attrs
    out = out.assign_attrs(dict(tau1=tau1, L1=L1))

    # ---Recalculate spectra, coherence and phase---
    t1 = Adi["t"][int(N / 4) : -int(N / 4)]  # Now N elements shorter
    c1 = Adi["c"][int(N / 4) : -int(N / 4)]

    # Number of segments = dof/2. gvpy computed this as floor((i2 - N) / N),
    # which only equals the number of segments that fit into t1 when the
    # correction span starts within the first N samples of the record. Taking
    # it from the length of t1 gives the same answer in that case and does not
    # blow up the reshape below otherwise.
    m = t1.size // N
    dof = 2 * m  # Number of degrees of freedom (power of 2)
    df = 1 / (N * dt)  # Frequency resolution at dof degrees of freedom.

    window = signal.windows.triang(N) * np.ones((m, N))
    At1 = fft.fft(signal.detrend(np.reshape(t1, shape=(m, N))) * window)
    Ac1 = fft.fft(signal.detrend(np.reshape(c1, shape=(m, N))) * window)

    # Positive frequencies only
    At1 = At1[:, 0 : int(N / 2)]
    Ac1 = Ac1[:, 0 : int(N / 2)]
    fn = f[0 : int(N / 2)]

    Et1n = 2 * np.nanmean(np.absolute(At1[:, : int(N / 2)]) ** 2, 0) / df / N**2
    Ec1n = 2 * np.nanmean(np.absolute(Ac1[:, : int(N / 2)]) ** 2, 0) / df / N**2

    # Cross spectral estimates
    Ct1c1n = 2 * np.nanmean(At1 * np.conj(Ac1) / df / N**2, axis=0)

    # Squared coherence estimates
    Coht1c1n = np.real(Ct1c1n * np.conj(Ct1c1n) / (Et1n * Ec1n))
    # 95% significance level for coherence from Gille notes
    betan = 1 - 0.05 ** (1 / (m - 1))

    # Cross-spectral phase estimates
    Phit1c1n = np.arctan2(np.imag(Ct1c1n), np.real(Ct1c1n))

    if not return_spectra:
        return out

    spectra = xr.Dataset(
        coords=dict(f=("f", fn, dict(long_name="frequency", units="Hz"))),
        data_vars=dict(
            Et=("f", Et1, dict(long_name="temperature spectral density")),
            Ec=("f", Ec1, dict(long_name="conductivity spectral density")),
            Et_corrected=(
                "f",
                Et1n,
                dict(long_name="corrected temperature spectral density"),
            ),
            Ec_corrected=(
                "f",
                Ec1n,
                dict(long_name="corrected conductivity spectral density"),
            ),
            coh=("f", Coht1c1, dict(long_name="squared coherence")),
            coh_corrected=(
                "f",
                Coht1c1n,
                dict(long_name="corrected squared coherence"),
            ),
            phase=("f", Phit1c1, dict(long_name="phase", units="rad")),
            phase_corrected=(
                "f",
                Phit1c1n,
                dict(long_name="corrected phase", units="rad"),
            ),
        ),
        attrs=dict(tau1=tau1, L1=L1, dof=dof, beta=betan),
    )

    return out, spectra


def plot_spectra(spectra):
    """Plot the T-C diagnostics returned by `phase_correct`.

    Draws the gvpy 2x2 diagnostic figure: temperature and conductivity
    spectra, squared coherence and cross-spectral phase, each before and after
    the correction. The fitted phase model and the 95% coherence significance
    level are drawn on top.

    Parameters
    ----------
    spectra : xarray.Dataset
        Second return value of `phase_correct` called with
        `return_spectra=True`.

    Returns
    -------
    ax : numpy.ndarray
        Array of the four matplotlib axes.
    """
    import matplotlib.pyplot as plt

    f = spectra.f.data
    dof = spectra.attrs["dof"]
    tau1 = spectra.attrs["tau1"]
    L1 = spectra.attrs["L1"]

    _, ax = plt.subplots(nrows=2, ncols=2, figsize=(9, 7), constrained_layout=True)
    ax0, ax1, ax2, ax3 = ax.flatten()

    ax0.plot(f, spectra.Et, label="uncorrected", color="0.5")
    ax0.plot(f, spectra.Et_corrected, label="corrected")
    ax0.set(
        yscale="log",
        xscale="log",
        xlabel="frequency [Hz]",
        ylabel=r"spectral density [$^{\circ}$C$^2$/Hz]",
        title="temperature spectra",
    )
    # 95% confidence interval on the corrected temperature spectrum
    et = dof * spectra.Et_corrected.data[10]
    ax0.plot(
        [f[10], f[10]],
        [
            et / stats.distributions.chi2.ppf(0.025, dof),
            et / stats.distributions.chi2.ppf(0.975, dof),
        ],
        "k",
    )
    ax0.legend()

    ax1.plot(f, spectra.Ec, label="uncorrected", color="0.5")
    ax1.plot(f, spectra.Ec_corrected, label="corrected")
    ax1.set(
        yscale="log",
        xscale="log",
        xlabel="frequency [Hz]",
        ylabel=r"spectral density [mmho$^2$/cm$^2$/Hz]",
        title="conductivity spectra",
    )

    ax2.plot(f, spectra.coh, color="0.5")
    ax2.plot(f, spectra.coh_corrected)
    ax2.plot(f, spectra.attrs["beta"] * np.ones(f.size), "k--")
    ax2.set(
        xlabel="frequency [Hz]",
        ylabel="squared coherence",
        ylim=(-0.1, 1.1),
        title="t/c coherence",
    )

    ax3.plot(f, spectra.phase, color="0.5", marker=".", linestyle="")
    ax3.plot(f, spectra.phase_corrected, marker=".", linestyle="")
    ax3.set(
        xlabel="frequency [Hz]",
        ylabel="phase [rad]",
        ylim=[-4, 4],
        title="t/c phase",
    )
    ax3.plot(f, -np.arctan(2 * np.pi * f * tau1) - 2 * np.pi * f * L1, "k--")

    return ax


def response_correction(
    ds: xr.Dataset, lag: float, tau_t: float, var: str = "t"
) -> xr.Dataset:
    r"""Apply the sensor response model to a whole record by FFT.

    The thermistor lags the conductivity cell because of its own finite
    response time and because of the physical separation of the two
    sensors. Both are removed in one step by multiplying the real FFT of
    the whole (gap-filled) record by the transfer function

    .. math::

        H(f) = (1 + i 2 \pi f \tau_t) \exp(i 2 \pi f \, \mathrm{lag})

    and inverting. This is the model `phase_correct` fits per segment in
    the frequency domain (its `H1`), applied here to the whole record at
    once and without `phase_correct`'s extra low-pass.

    Parameters
    ----------
    ds : xarray.Dataset
        CTD time series with variable `var` on an evenly sampled `time`
        coordinate of dtype datetime64. The sampling interval is read
        from `time`, not assumed.
    lag : float
        Sensor lag [s]. `exp(+i 2 pi f lag)` advances the record by
        `lag`, so a positive `lag` means `var` (temperature) physically
        lags conductivity, matching the sign convention of `find_lags`
        and `phase_correct`. Negative `lag` raises `ValueError`: an
        advance in the other direction is not expected for a FastCAT.
    tau_t : float
        Thermistor time constant [s]. Zero disables the `(1 + i 2 pi f
        tau_t)` amplitude term.
    var : str, optional
        Name of the variable to correct. Defaults to `"t"`.

    Returns
    -------
    out : xarray.Dataset
        Deep copy of `ds` with `var` replaced by the response-corrected
        record. Interior NaN present in the input are restored as NaN.
        The last `ceil(lag * fs)` samples are additionally set to NaN,
        since advancing the record by `lag` wraps that many samples in
        from the record start (the FFT treats the record as periodic).
        When `lag == 0` and `tau_t == 0` this is a copy of `ds` with `var`
        unchanged.

    Raises
    ------
    ValueError
        If `lag` is negative.

    Notes
    -----
    `fs` is `1 / median(diff(time))`, the same convention `phase_correct`
    and `thermal_mass_correction` use, so a time gap in a concatenated
    record perturbs at most the samples adjacent to it, not the sampling
    rate.
    """
    if lag < 0:
        raise ValueError(
            "negative lag is not expected for a FastCAT (temperature "
            f"advancing ahead of conductivity); got lag={lag}"
        )

    ds = ds.copy(deep=True)
    if lag == 0 and tau_t == 0:
        return ds

    dt = float(np.median(np.diff(ds.time.data)) / np.timedelta64(1, "s"))
    fs = 1 / dt

    x_raw = ds[var].data
    x, mask = _fill_gaps(x_raw)
    n = x.size

    f = np.fft.rfftfreq(n, d=1 / fs)
    H = (1 + 1j * 2 * np.pi * f * tau_t) * np.exp(1j * 2 * np.pi * f * lag)
    y = np.fft.irfft(np.fft.rfft(x) * H, n)

    if lag > 0:
        ntrail = int(np.ceil(lag * fs))
        y[-ntrail:] = np.nan
    y[mask] = np.nan

    ds[var] = (ds[var].dims, y, ds[var].attrs)
    return ds


def thermal_mass_correction(
    ds: xr.Dataset, alpha: float = 0.03, beta: float = 1 / 7, dcdt: str = "sbe"
) -> xr.Dataset:
    """Correct conductivity for the thermal mass of the conductivity cell.

    A temperature change advecting past the conductivity cell heats or cools
    the cell wall with a lag, which perturbs the conductivity reading. The
    correction is a recursive filter (Lueck & Picklo, 1990) applied to the
    temperature record and added back to conductivity.

    Parameters
    ----------
    ds : xarray.Dataset
        CTD time series with variables `t` [degC] and `c` on an evenly
        sampled `time` coordinate of dtype datetime64. The sampling interval
        is read from `time`, not assumed.
    alpha : float, optional
        Amplitude of the thermal anomaly. Defaults to 0.03, the SBE Data
        Processing manual value for the SBE49. See Notes for alternatives.
    beta : float, optional
        Inverse relaxation time constant [1/s] of the thermal anomaly.
        Defaults to 1/7 (SBE Data Processing manual). See Notes for
        alternatives.
    dcdt : {"sbe", "constant"}, optional
        Conductivity sensitivity `dc/dT` [S/m/degC] used to scale the
        temperature-difference input to the recursion. `"sbe"` (default)
        uses the SBE Data Processing manual's temperature-dependent form
        `0.1 * (1 + 0.006 * (t - 20))`, evaluated per sample on the
        (gap-filled) input `t`. `"constant"` uses the fixed `0.1` this
        function used before this parameter existed, and the value Lueck
        & Picklo (1990) and the dead MATLAB toolbox branch used
        throughout. The two agree exactly at 20 degC; over 3 to 27 degC
        they differ by up to about 10% (see the table below).

    Returns
    -------
    out : xarray.Dataset
        Deep copy of `ds` with `c` replaced by the thermal-mass-corrected
        conductivity. `ds` itself is not modified. NaN in the input `t`
        do not propagate into `c`: both `t` and `c` are gap-filled for the
        recursion, and the interior-NaN mask captured from the input `c`
        is restored on the output.

    Notes
    -----
    The discrete filter coefficients follow Lueck & Picklo (1990), not the
    SBE Data Processing manual formula (both are quoted below for
    reference); the two differ in how the sample rate enters the
    coefficients. `alpha` and `beta` are picked from a Nyquist frequency
    `fn = 1 / (2 * dt)` computed from the data's own time axis rather than a
    fish-specific hardcoded value, which is what makes this function
    fish-agnostic (for a 16 Hz record, `fn = 8`, matching the value gvpy
    hardcoded for the SBE49-on-FCTD case this was ported from).

    Choosing `alpha` and `beta` is the subject of the T-C correction analysis
    (FCTD reprocessing sub-project 3). Values seen in the wild:

    ======================================  =========  =========
    source                                  alpha      1/beta
    ======================================  =========  =========
    SBE Data Processing manual (SBE49)      0.03       7.0
    Lueck & Picklo (1990)                   0.02       0.10
    dead MATLAB toolbox branch              0.02       0.10
    Andriatis/Pinkel TFO 2021, serial 537   0.134      3.95
    ======================================  =========  =========

    The SBE Data Processing manual formula, given for reference and not
    used here:

    .. math::

        a &= \\frac{2 \\alpha}{\\Delta t \\, \\beta + 2} \\\\
        b &= 1 - \\frac{2a}{\\alpha} \\\\
        dc/dT &= 0.1 (1 + 0.006 (T - 20)) \\\\
        \\mathrm{ctm}_i &= -b \\, \\mathrm{ctm}_{i-1} + a \\, (dc/dT)_i \\, dT_i

    with `dT` the sample-to-sample temperature difference and `ctm` in S/m.
    """
    if dcdt not in ("sbe", "constant"):
        raise ValueError(f"dcdt must be 'sbe' or 'constant', got {dcdt!r}")

    ds = ds.copy(deep=True)

    dt = float(np.median(np.diff(ds.time.data)) / np.timedelta64(1, "s"))
    fn = 1 / (2 * dt)

    T, _ = _fill_gaps(ds.t.data)
    C, c_mask = _fill_gaps(ds.c.data)

    if dcdt == "sbe":
        gamma = 0.1 * (1 + 0.006 * (T - 20))
    else:
        gamma = np.full_like(T, 0.1)

    dTp = np.diff(T, prepend=T[0])
    dTp[0] = dTp[1]
    ctm = np.zeros_like(dTp)

    aa = 4 * fn * alpha / beta / (1 + 4 * fn / beta)
    bb = 1 - 2 * aa / alpha
    for ii in range(1, len(ctm)):
        ctm[ii] = -bb * ctm[ii - 1] + aa * gamma[ii] * dTp[ii]

    c_out = C + ctm
    c_out[c_mask] = np.nan
    ds["c"] = (ds.c.dims, c_out, ds.c.attrs)
    return ds


def viscous_heating_temperature_correction(v, Pr: float = 12.4) -> np.ndarray:
    r"""Temperature error from viscous heating of an unpumped sensor.

    Flow past an unpumped thermistor dissipates kinetic energy at the sensor
    surface, warming the reading. Ullman & Hebert (2014) give

    .. math::

        \Delta T = 0.8 \times 10^{-4} \, \mathrm{Pr}^{0.5} \, v^2

    with flow speed `v` past the sensor [m/s] and Prandtl number `Pr`, the
    ratio of momentum to thermal diffusivity, :math:`\mathcal{O}(10)` for
    seawater.

    Parameters
    ----------
    v : array-like
        Flow speed past the sensor [m/s].
    Pr : float, optional
        Prandtl number. Defaults to 12.4, seawater's value at 2 degC
        (Larson & Pedersen, 1996), the design temperature for the T-C
        correction analysis's viscous-heating bound (FCTD reprocessing
        sub-project 3).

    Returns
    -------
    dT : numpy.ndarray
        Temperature correction [degC], same shape as `v`.

    Notes
    -----
    The Ullman & Hebert derivation is for an unpumped sensor. It is not
    applicable to the pumped SBE49 and is off by default in the FCTD
    pipeline; `correct` runs it only when `viscous_heating=True`, to bound
    the effect rather than to apply it routinely.

    An earlier version of this function multiplied the formula by an
    undocumented `scale=2.0` factor of unknown origin. It has been
    dropped: the formula above is Ullman & Hebert's as published, with no
    empirical adjustment.
    """
    v = np.asarray(v)
    return 0.8e-4 * Pr**0.5 * v**2


def find_lags(ds: xr.Dataset, window: int = 80) -> tuple[np.ndarray, np.ndarray]:
    """Estimate the time lag between temperature and conductivity by segment.

    Temperature and conductivity are each differenced and cross-correlated
    within short overlapping windows; the correlation peak within each
    window is refined to sub-sample resolution with a quadratic fit. This
    gives a lag time series that can reveal drift over a cast, unlike the
    single lag `phase_correct` fits over the whole fit range.

    Parameters
    ----------
    ds : xarray.Dataset
        CTD time series with variables `t`, `c`, `dPdt` on an evenly sampled
        `time` coordinate of dtype datetime64. The sampling interval is read
        from `time`, not assumed.
    window : int, optional
        Number of samples per correlation window. Windows overlap by half.
        Defaults to 80.

    Returns
    -------
    lags : numpy.ndarray
        Lag [s] for each window, positive meaning `t` lags `c`.
    w : numpy.ndarray
        Mean `dPdt` over each window, used as a proxy for fall speed.

    Notes
    -----
    `correlate(ci, ti)` on its own (`ci`, `ti` the differenced, demeaned `c`
    and `t` segments, gvpy `mod.py:1087-1092`) peaks at the negative of the
    delay of `t` behind `c`: correlating a signal `a` against a copy `b`
    delayed by `d` samples peaks at lag `-d`, confirmed independently with a
    plain `numpy.roll` delay test. The sign is flipped here so the returned
    lag matches the documented convention, positive meaning `t` lags `c`,
    which is what `tests/test_tc.py::test_find_lags_recovers_known_lag` pins.
    """
    c = ds.c.data
    t = ds.t.data
    dpdt = ds.dPdt.data

    dt = float(np.median(np.diff(ds.time.data)) / np.timedelta64(1, "s"))
    freq = 1 / dt

    def fit_2d_poly(lags, corrs):
        # Fit the quadratic curve
        coefficients = np.polyfit(lags, corrs, 2)
        # Vertex of the parabola
        vertex_x = -coefficients[1] / (2 * coefficients[0])
        return vertex_x

    def find_corrs(t, c, tr):
        ci = np.diff(c[tr] - np.mean(c[tr]))
        ti = np.diff(t[tr] - np.mean(t[tr]))
        correlation = signal.correlate(ci - np.mean(ci), ti - np.mean(ti), mode="full")
        lags = signal.correlation_lags(len(ci), len(ti), mode="full") * 1 / freq
        lag = np.argmax(np.abs(correlation))
        inds = range(lag - 1, lag + 2)
        return lags[inds], correlation[inds]

    t_lp = lowpassfilter(t, lowcut=1 / 4, fs=1)
    c_lp = lowpassfilter(c, lowcut=1 / 4, fs=1)

    n = len(t)
    m = window
    logger.info("%d scans, %d segments", n, n // (m // 2))

    start_index = np.arange(0, n - m, m // 2)
    lagi = []
    wi = []
    for si in start_index:
        tr = range(si, si + window)
        lags, corrs = find_corrs(t_lp, c_lp, tr)
        # Sign flipped relative to the raw correlate() convention; see Notes.
        lag = -fit_2d_poly(lags, corrs)
        lagi.append(lag)
        wi.append(np.mean(dpdt[tr]))

    return np.array(lagi), np.array(wi)


def correct(
    ds: xr.Dataset,
    lag: float = 0.0,
    tau_t: float = 0.0,
    lowpass: float | None = None,
    thermal_mass: bool = False,
    alpha: float = 0.03,
    beta: float = 1 / 7,
    viscous_heating: bool = False,
    pr: float = 12.4,
) -> xr.Dataset:
    """Apply the T-C sensor response correction chain to a CTD record.

    Every parameter is explicit; the library default (every argument at
    its default) is a no-op, so a cruise config opts into each step. The
    steps run in this fixed order, each skippable independently:

    0. Gap fill. Interior NaN in `t` and `c` are linearly interpolated in
       time for the duration of each step that needs finite input and
       restored to NaN afterwards (a mask captured fresh at each step,
       equivalent to one mask carried through since no step removes an
       existing NaN; `response_correction` also adds new trailing NaN,
       see step 2). Handled internally by `_fill_gaps`, `response_correction`
       and `thermal_mass_correction`; nothing extra is needed here beyond
       the low-pass step, which needs finite input too.
    1. Low-pass. `t` and `c` are zero-phase Butterworth filtered
       (`lowpassfilter`, order 3) at `lowpass` Hz. Skipped when `lowpass`
       is None.
    2. Sensor response, on `t`. `response_correction(ds, lag, tau_t)`
       multiplies the whole record's real FFT by
       `H(f) = (1 + i 2 pi f tau_t) exp(i 2 pi f lag)` and inverts.
       `exp(+i 2 pi f lag)` advances the record by `lag`, so a positive
       `lag` means `t` physically lags `c`, matching `find_lags` and
       `phase_correct`. Skipped when both `lag` and `tau_t` are 0.

       This step runs on the low-passed record, not the raw one, because
       the derivative-like `tau_t` term amplifies noise without bound
       towards Nyquist; the low-pass in step 1 is what keeps that finite.
       It runs as a whole-record FFT rather than the time-domain form
       `t_corrected = t + tau_t * dt/dt_sample` a central difference
       would suggest, because that finite-difference derivative
       underestimates the true derivative near the cutoff at 16 Hz. A
       numerical check during design review (`tests/test_tc.py`, true
       lag 0.08 s, tau_t 0.06 s, 4 Hz low-pass) found central differences
       bias the fitted tau_t high by nearly a factor of two (roughness
       minimum at tau_t = 0.11 s against the true 0.06 s), while the FFT
       transfer function recovers the clean signal to within 1.3% of the
       uncorrected residual against 8.9% for the central-difference
       version; `test_correct_true_parameters_recover_clean_signal` pins
       the 10x gap between them. The correction is therefore applied in
       the frequency domain; time-domain estimators (`find_lags`, the
       roughness cost map) are unaffected, since they only measure `lag`
       and `tau_t`, they do not differentiate to apply them.
    3. Thermal mass, on `c`. `thermal_mass_correction(ds, alpha, beta)`,
       with `dc/dT = 0.1 * (1 + 0.006 * (t - 20))` evaluated on the
       (already response-corrected) `t`. Skipped when `thermal_mass` is
       False.
    4. Viscous heating, on `t`. `t <- t - 0.8e-4 * pr**0.5 * v**2` with
       `v = |dPdt|` (dbar/s taken as m/s). Off by default; kept to bound
       the effect on an unpumped-sensor assumption that does not strictly
       hold for the pumped SBE49.

    `p` is never filtered or otherwise touched.

    Parameters
    ----------
    ds : xarray.Dataset
        CTD time series with `t`, `c`, `p`, `dPdt` on an evenly sampled
        `time` coordinate of dtype datetime64. The sampling interval is
        read from `time`, not assumed.
    lag : float, optional
        Sensor lag [s] for the response step, `lag > 0` meaning `t`
        lags `c`. Defaults to 0.0 (no advance).
    tau_t : float, optional
        Thermistor time constant [s] for the response step. Defaults to
        0.0 (amplitude term disabled).
    lowpass : float or None, optional
        Cut-off frequency [Hz] of the zero-phase low-pass on `t` and `c`.
        Defaults to None (no low-pass).
    thermal_mass : bool, optional
        Whether to run the thermal-mass correction on `c`. Defaults to
        False.
    alpha : float, optional
        Thermal-mass amplitude, passed to `thermal_mass_correction`.
        Defaults to 0.03 (SBE Data Processing manual).
    beta : float, optional
        Thermal-mass inverse relaxation time [1/s], passed to
        `thermal_mass_correction`. Defaults to 1/7 (SBE Data Processing
        manual).
    viscous_heating : bool, optional
        Whether to run the viscous-heating correction on `t`. Defaults
        to False.
    pr : float, optional
        Prandtl number, passed to
        `viscous_heating_temperature_correction`. Defaults to 12.4.

    Returns
    -------
    out : xarray.Dataset
        Deep copy of `ds` on the same `time` axis (no reindexing), with
        `t` and `c` carrying the requested corrections. `t.attrs["processing"]`
        and `c.attrs["processing"]` each list the steps that touched that
        variable as `"; "`-joined strings (`"none"` if none did), and
        `ds.attrs["corrections"]` is a function-call-style summary of the
        whole chain with every parameter used, in the sub-project 2
        style (`"none"` if nothing was applied).

    Notes
    -----
    With every argument at its default, this function returns an
    unmodified copy of `ds`: `t`, `c`, `p` and `dPdt` are byte-for-byte
    identical to the input, and both `processing` attrs and
    `attrs["corrections"]` read `"none"`.
    """
    ds = ds.copy(deep=True)

    dt = float(np.median(np.diff(ds.time.data)) / np.timedelta64(1, "s"))
    fs = 1 / dt

    t_steps: list[str] = []
    c_steps: list[str] = []
    corrections: list[str] = []

    if lowpass is not None:
        t_attrs = dict(ds["t"].attrs)
        c_attrs = dict(ds["c"].attrs)

        t_raw, t_mask = _fill_gaps(ds["t"].data)
        c_raw, c_mask = _fill_gaps(ds["c"].data)
        t_lp = lowpassfilter(t_raw, lowcut=lowpass, fs=fs)
        c_lp = lowpassfilter(c_raw, lowcut=lowpass, fs=fs)
        t_lp[t_mask] = np.nan
        c_lp[c_mask] = np.nan

        ds["t"] = (ds["t"].dims, t_lp, t_attrs)
        ds["c"] = (ds["c"].dims, c_lp, c_attrs)

        t_steps.append(f"lowpass {lowpass} Hz")
        c_steps.append(f"lowpass {lowpass} Hz")
        corrections.append(f"lowpassfilter(lowcut={lowpass}, fs={fs})")

    if lag != 0 or tau_t != 0:
        ds = response_correction(ds, lag=lag, tau_t=tau_t, var="t")
        t_steps.append(f"response lag {lag:.3f} s tau {tau_t:.3f} s")
        corrections.append(f"response_correction(lag={lag}, tau_t={tau_t})")

    if thermal_mass:
        ds = thermal_mass_correction(ds, alpha=alpha, beta=beta)
        c_steps.append(f"thermal mass alpha={alpha} beta={beta}")
        corrections.append(f"thermal_mass_correction(alpha={alpha}, beta={beta})")

    if viscous_heating:
        v = np.abs(ds["dPdt"].data)
        dT = viscous_heating_temperature_correction(v, Pr=pr)
        t_attrs = dict(ds["t"].attrs)
        ds["t"] = ds["t"] - dT
        ds["t"].attrs = t_attrs
        t_steps.append(f"viscous heating pr={pr}")
        corrections.append(f"viscous_heating_temperature_correction(pr={pr})")

    ds["t"].attrs["processing"] = "; ".join(t_steps) if t_steps else "none"
    ds["c"].attrs["processing"] = "; ".join(c_steps) if c_steps else "none"
    ds.attrs["corrections"] = "; ".join(corrections) if corrections else "none"

    return ds
