"""The production T-C correction chain, `correct`, and its steps."""

import numpy as np
import xarray as xr
from scipy import signal
from scipy.signal import butter, filtfilt

from modfish.tc._grid import _fill_gaps, _on_uniform_grid
from modfish.utils import sampling_interval


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
        CTD time series with variable `var` on a `time` coordinate of
        dtype datetime64. The sampling interval is read from `time`, not
        assumed. Time gaps are allowed, see Notes.
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
        If `lag` is negative, or, when the correction runs at all (`lag`
        or `tau_t` nonzero), if `ds.time` has a step of zero or less,
        which `_uniform_slots` refuses because two samples would fall in
        the same slot of the uniform grid.

    Notes
    -----
    `fs` is the inverse of `modfish.utils.sampling_interval`, the mean
    step over the gap-free stretches of `time`, the convention every
    function in this module shares, so a time gap in a concatenated
    record perturbs at most the samples adjacent to it and leaves the
    sampling rate alone.

    The whole-record `rfft` treats the record as periodic, so a mismatch
    between the first and last sample wraps into a discontinuity that the
    fractional-delay kernel of `H` (a sinc) rings on, all the way through
    the record. This was found on a real d09 cast whose record starts at
    4.6 dbar and ends at 436 dbar (a 15.7 degC step between its last and
    first sample): the salinity-roughness cost along a lag scan showed a
    sawtooth with minima at exactly 0, 0.0625 and 0.125 s (whole samples
    at 16 Hz) and a max/min ratio of 34, against a smooth single minimum
    and max/min ratio of 3.4 on a stretch whose ends differ by only 0.04
    degC. To avoid this, the straight line through the first and last
    sample of the gap-filled record is subtracted before the FFT, so the
    periodic extension is continuous; `H` is applied to the residual, and
    the line is added back analytically rather than through the FFT. For
    a line `a + b*t`, advancing by `lag` gives `a + b*(t + lag)`, and the
    `(1 + i 2 pi f tau_t)` factor is a time-derivative operator, adding
    `tau_t * b`; the restored line is therefore
    `a + b*(t + lag) + tau_t*b`.

    The record is laid out on a uniform time grid at its own sampling
    interval (`_on_uniform_grid`) before the FFT, so a time gap arrives
    as a run of missing samples that the gap fill bridges with a linear
    ramp, in place of a step for the fractional-delay sinc and the
    derivative term to ring on. A record without a gap is corrected as it
    is, with no re-laying.
    """
    if lag < 0:
        raise ValueError(
            "negative lag is not expected for a FastCAT (temperature "
            f"advancing ahead of conductivity); got lag={lag}"
        )

    if lag == 0 and tau_t == 0:
        return ds.copy(deep=True)

    def _apply(ds: xr.Dataset) -> xr.Dataset:
        ds = ds.copy(deep=True)

        dt = sampling_interval(ds.time.data)
        fs = 1 / dt

        x_raw = ds[var].data
        x, mask = _fill_gaps(x_raw)
        n = x.size

        t_idx = np.arange(n) * dt
        a = x[0]
        b = (x[-1] - x[0]) / t_idx[-1] if n > 1 else 0.0
        resid = x - (a + b * t_idx)

        f = np.fft.rfftfreq(n, d=1 / fs)
        H = (1 + 1j * 2 * np.pi * f * tau_t) * np.exp(1j * 2 * np.pi * f * lag)
        y = np.fft.irfft(np.fft.rfft(resid) * H, n)
        y += a + b * (t_idx + lag) + tau_t * b

        if lag > 0:
            ntrail = int(np.ceil(lag * fs))
            y[-ntrail:] = np.nan
        y[mask] = np.nan

        ds[var] = (ds[var].dims, y, ds[var].attrs)
        return ds

    return _on_uniform_grid(ds, (var,), _apply)


def thermal_mass_correction(
    ds: xr.Dataset,
    alpha: float = 0.03,
    beta: float = 1 / 7,
    dcdt: str = "sbe",
    profile: dict | None = None,
) -> xr.Dataset:
    """Correct conductivity for the thermal mass of the conductivity cell.

    A temperature change advecting past the conductivity cell heats or cools
    the cell wall with a lag, which perturbs the conductivity reading. The
    correction is a recursive filter (Lueck & Picklo, 1990) applied to the
    temperature record and added back to conductivity.

    Parameters
    ----------
    ds : xarray.Dataset
        CTD time series with variables `t` [degC] and `c` on a `time`
        coordinate of dtype datetime64. The sampling interval is read from
        `time`, not assumed. Time gaps are allowed, see Notes.
    alpha : float, optional
        Amplitude of the thermal anomaly. Must be positive: it divides into
        the filter coefficient `b = 1 - 2a/alpha`, so `alpha = 0` has no
        meaning here and raises `ValueError`. To run no thermal-mass
        correction, pass `thermal_mass=False` to `tc.correct`. Defaults to
        0.03, the SBE Data Processing manual value for the SBE49. See Notes
        for alternatives.
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
        & Picklo (1990) used throughout. The two agree exactly at 20 degC; over 3 to 27 degC
        they differ by up to about 10% (see the table below).
    profile : dict or None, optional
        Pressure profile of the product `alpha / beta` (`alpha * tau`,
        seconds), as `{"p": [...], "product": [...]}` with `p` [dbar]
        strictly increasing and every `product` positive. The product is
        linear in pressure between knots and held at the end values
        beyond them. When given, the recursion runs at a unit product
        (`alpha = beta`) and its output at each sample is multiplied by
        the product at that sample's pressure, `ds` must carry `p`, and
        the scalar `alpha` is not used. Defaults to None (constant
        `alpha`).

    Returns
    -------
    out : xarray.Dataset
        Deep copy of `ds` with `c` replaced by the thermal-mass-corrected
        conductivity. `ds` itself is not modified. NaN in the input `t`
        do not propagate into `c`: both `t` and `c` are gap-filled for the
        recursion, and the interior-NaN mask captured from the input `c`
        is restored on the output.

    Raises
    ------
    ValueError
        If `alpha` is not positive (without a profile), if `dcdt` is
        neither `"sbe"` nor `"constant"`, if `profile` is malformed or
        given for a `ds` without `p`, or if `ds.time` has a step of zero
        or less, which
        `_uniform_slots` refuses because two samples would fall in the
        same slot of the uniform grid.

    Notes
    -----
    The code writes the discrete filter coefficient as
    `a = 4 fn alpha / beta / (1 + 4 fn / beta)` (Lueck & Picklo, 1990) with the
    Nyquist frequency `fn = 1 / (2 * dt)` taken from the data's own time
    axis. With that `fn` they equal the SBE Data Processing manual
    coefficients below (0.029867 for both at 16 Hz with the manual's
    `alpha` and `beta`). The two forms differ only in `dc/dT`, which the
    `dcdt` argument selects. For a 16 Hz record `fn = 8`, the value gvpy
    hardcoded for the SBE49 on the FCTD.

    `alpha` and `beta` are fitted per instrument. For SBE49 serial 0537
    the 2025 MOTIVE fit gives `alpha` 0.010 and `beta` 1/12 per second,
    set in that cruise's processing config. Values seen in the wild:

    ========================================  ===========  ==========
    source                                    alpha        1/beta [s]
    ========================================  ===========  ==========
    SBE Data Processing manual (SBE49)        0.03         7.0
    Lueck & Picklo (1990), 0.85 m/s tow       0.028        10
    Lueck & Picklo (1990), 1.7 to 1.9 m/s     0.020-0.025  10
    SBE fit quoted by Lueck & Picklo (1990)   0.015-0.020  10
    Andriatis/Pinkel TFO 2021, serial 537     0.134        3.95
    ========================================  ===========  ==========

    Lueck & Picklo recommend 0.028 for a cell pumped at 30e-6 m^3/s.

    The SBE Data Processing manual formula:

    .. math::

        a &= \\frac{2 \\alpha}{\\Delta t \\, \\beta + 2} \\\\
        b &= 1 - \\frac{2a}{\\alpha} \\\\
        dc/dT &= 0.1 (1 + 0.006 (T - 20)) \\\\
        \\mathrm{ctm}_i &= -b \\, \\mathrm{ctm}_{i-1} + a \\, (dc/dT)_i \\, dT_i

    with `dT` the sample-to-sample temperature difference and `ctm` in S/m.

    The recursion is evaluated with `scipy.signal.lfilter` (denominator
    `[1, bb]`), which reproduces the sample-by-sample loop to 1e-15. On a
    4 M-sample record this function takes 0.2 s against 1.9 s for the
    previous loop; the `lfilter` call itself accounts for 0.03 s of that,
    the rest being the deep copy, gap fill and gamma computation.

    With a pressure profile the correction is the output of the recursion
    at a unit product, scaled per sample by `product(p)`. The recursion is
    linear in a constant `alpha` (`aa` is linear in `alpha` and `bb` does
    not depend on it), so every sample receives exactly the correction a
    record with one constant product, the product at that sample's
    pressure, would receive. This is the quantity a per-band fit of a
    constant product measures. Scaling the recursion's input instead would
    let the heat the cell carries for about `1 / beta` keep the product of
    the pressure where it was picked up, which is shallower on a down cast
    and deeper on an up cast, and would make the two directions see
    different products wherever the product changes with depth. In steady
    state the correction is `product(p) * dc/dT * dT/dt` either way.

    The record is laid out on a uniform time grid at its own sampling
    interval (`_on_uniform_grid`) before the recursion runs, so a time
    gap arrives as a run of missing samples that the gap fill bridges
    with a linear ramp. A step in `t` between the two samples adjacent to
    a gap would otherwise start a transient the recursion carries for a
    few time constants. A record without a gap is corrected as it is,
    with no re-laying.
    """
    if profile is None and alpha <= 0:
        raise ValueError(
            "alpha must be positive; set thermal_mass=False to disable the "
            "thermal-mass step"
        )
    knots = _check_tm_profile(profile) if profile is not None else None
    if knots is not None and "p" not in ds:
        raise ValueError("a thermal-mass profile needs pressure `p` in ds")
    if dcdt not in ("sbe", "constant"):
        raise ValueError(f"dcdt must be 'sbe' or 'constant', got {dcdt!r}")

    def _apply(ds: xr.Dataset) -> xr.Dataset:
        ds = ds.copy(deep=True)

        dt = sampling_interval(ds.time.data)
        fn = 1 / (2 * dt)

        T, _ = _fill_gaps(ds.t.data)
        C, c_mask = _fill_gaps(ds.c.data)

        if dcdt == "sbe":
            gamma = 0.1 * (1 + 0.006 * (T - 20))
        else:
            gamma = np.full_like(T, 0.1)

        dTp = np.diff(T, prepend=T[0])
        dTp[0] = dTp[1]

        # with a profile the recursion runs at a unit product and its
        # output is scaled by the product at each sample's pressure
        a_run = alpha if knots is None else beta
        aa = 4 * fn * a_run / beta / (1 + 4 * fn / beta)
        bb = 1 - 2 * aa / a_run
        x = aa * gamma * dTp
        x[0] = 0.0  # the loop starts at index 1 with ctm[0] = 0
        ctm = signal.lfilter([1.0], [1.0, bb], x)
        if knots is not None:
            # p carries NaN at gap slots when this runs inside `correct`'s
            # uniform grid, so it is filled like t before the lookup
            P, _ = _fill_gaps(ds.p.data)
            ctm = np.interp(P, knots[0], knots[1]) * ctm

        c_out = C + ctm
        c_out[c_mask] = np.nan
        ds["c"] = (ds.c.dims, c_out, ds.c.attrs)
        return ds

    variables = ("t", "c") if knots is None else ("t", "c", "p")
    return _on_uniform_grid(ds, variables, _apply)


def _check_tm_profile(profile) -> tuple[np.ndarray, np.ndarray]:
    """Validate a thermal-mass profile and return its knots as arrays.

    Parameters
    ----------
    profile : mapping
        `{"p": [...], "product": [...]}`, see `thermal_mass_correction`.

    Returns
    -------
    p, product : numpy.ndarray
        Knot pressures [dbar] and products [s], float, 1-D.

    Raises
    ------
    ValueError
        If a key is missing, the arrays are not 1-D of equal length at
        least 1, `p` is not strictly increasing, or a product is not
        positive and finite.
    """
    try:
        p = np.asarray(profile["p"], dtype=float)
        prod = np.asarray(profile["product"], dtype=float)
    except (KeyError, TypeError) as err:
        raise ValueError(
            'thermal-mass profile must be {"p": [...], "product": [...]}'
        ) from err
    if p.ndim != 1 or prod.ndim != 1 or p.size == 0 or p.size != prod.size:
        raise ValueError("thermal-mass profile p and product must be 1-D of equal length")
    if np.any(np.diff(p) <= 0) or not np.isfinite(p).all():
        raise ValueError("thermal-mass profile p must be finite and strictly increasing")
    if not (np.isfinite(prod).all() and np.all(prod > 0)):
        raise ValueError("thermal-mass profile products must be positive and finite")
    return p, prod


def _format_tm_profile(profile) -> str:
    """Compact text form of a thermal-mass profile for the processing attrs."""
    p, prod = _check_tm_profile(profile)
    ps = ", ".join(f"{v:g}" for v in p)
    qs = ", ".join(f"{v:g}" for v in prod)
    return f"p=[{ps}] product=[{qs}]"


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
        (Larson & Pedersen, 1996), the temperature at which the FCTD T-C
        analysis bounded the viscous-heating effect.

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
    thermal_mass_profile: dict | None = None,
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
        CTD time series with `t`, `c`, `p`, `dPdt` on a `time` coordinate
        of dtype datetime64. The sampling interval is read from `time`, not
        assumed. Time gaps are allowed: the chain runs on a uniform grid,
        see Notes.
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
    thermal_mass_profile : dict or None, optional
        Pressure profile of the thermal-mass product `alpha * tau`,
        passed to `thermal_mass_correction` as `profile`. When given, the
        scalar `alpha` is not used. Defaults to None.

    Returns
    -------
    out : xarray.Dataset
        Deep copy of `ds` on the same `time` axis (no reindexing), with
        `t` and `c` carrying the requested corrections. `t.attrs["processing"]`
        and `c.attrs["processing"]` each list the steps that touched that
        variable as `"; "`-joined strings (`"none"` if none did), and
        `ds.attrs["corrections"]` is a function-call-style summary of the
        whole chain with every parameter used (`"none"` if nothing was
        applied).

    Raises
    ------
    ValueError
        If `ds.time` has a step of zero or less, which `_uniform_slots`
        refuses because two samples would fall in the same slot of the
        uniform grid. This is checked before any step runs, so it is
        raised even with every argument at its default.

    Notes
    -----
    With every argument at its default, this function returns an
    unmodified copy of `ds`, provided the time axis is strictly
    increasing: `t`, `c`, `p` and `dPdt` are byte-for-byte identical to
    the input, and both `processing` attrs and `attrs["corrections"]`
    read `"none"`.

    The whole chain runs on the record laid out on a uniform time grid at
    its own sampling interval (`_on_uniform_grid`), and the result comes
    back on the input time axis. A time gap in a concatenated deployment
    record then reaches each step as a run of missing samples that the
    gap fill bridges with a linear ramp. Without the grid the two samples
    adjacent to a gap present a step of several degrees: the zero-phase
    low-pass smears it, the `(1 + i 2 pi f tau_t)` term differentiates it
    into a spike, and the thermal-mass recursion carries the transient
    for a few time constants. On the 2025 MOTIVE d09 record, where a
    276 s gap has pressure jumping from 385 to 61 dbar, the whole-record
    chain differed from a per-stretch one by up to 3.5 degC in `t` and
    4.3 in practical salinity, with 631 samples above 0.01 in salinity
    within about a minute after the gap. A record without a gap is handed
    to the steps as it is, so its output is unchanged to the last bit.
    """

    def _chain(ds: xr.Dataset) -> xr.Dataset:
        ds = ds.copy(deep=True)

        dt = sampling_interval(ds.time.data)
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

        if thermal_mass and thermal_mass_profile is None:
            ds = thermal_mass_correction(ds, alpha=alpha, beta=beta)
            c_steps.append(f"thermal mass alpha={alpha} beta={beta}")
            corrections.append(f"thermal_mass_correction(alpha={alpha}, beta={beta})")
        elif thermal_mass:
            ds = thermal_mass_correction(ds, beta=beta, profile=thermal_mass_profile)
            text = _format_tm_profile(thermal_mass_profile)
            c_steps.append(f"thermal mass beta={beta} {text}")
            corrections.append(f"thermal_mass_correction(beta={beta}, profile {text})")

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

    return _on_uniform_grid(ds, ("t", "c", "p", "dPdt"), _chain)
