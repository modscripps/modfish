"""Estimators for the T-C correction parameters: lag, time constant, thermal mass."""

import logging

import gsw
import numpy as np
import xarray as xr
from scipy import signal

from modfish.tc._grid import _fill_gaps
from modfish.tc.chain import correct, lowpassfilter, thermal_mass_correction
from modfish.tc.phase import add_tcfit_default
from modfish.utils import sampling_interval

logger = logging.getLogger(__name__)


def _fit_range_default(ds: xr.Dataset, pmin, pmax):
    """Fill a missing pressure-range bound from `add_tcfit_default`.

    Parameters
    ----------
    ds : xarray.Dataset
        CTD time series with pressure in variable `p` [dbar], passed to
        `add_tcfit_default` when either bound is missing.
    pmin, pmax : float or None
        Lower/upper pressure bound [dbar]. `None` is replaced by the
        corresponding `add_tcfit_default` bound; a given value is kept
        as-is.

    Returns
    -------
    pmin, pmax : float
        Resolved bounds.
    """
    if pmin is None or pmax is None:
        tcfit = add_tcfit_default(ds.copy()).attrs["tcfit"]
        if pmin is None:
            pmin = tcfit[0]
        if pmax is None:
            pmax = tcfit[1]
    return pmin, pmax


def find_lags(ds: xr.Dataset, window: int = 80, lowpass: float = 4.0) -> xr.Dataset:
    """Estimate the time lag between temperature and conductivity by segment.

    Temperature and conductivity are each low-pass filtered, differenced and
    cross-correlated within short overlapping windows; the correlation peak
    within each window is refined to sub-sample resolution with a quadratic
    fit. This gives a lag time series that can reveal drift over a cast,
    unlike the single lag `phase_correct` fits over the whole fit range.

    Parameters
    ----------
    ds : xarray.Dataset
        CTD time series with variables `t`, `c`, `p`, `dPdt` on an evenly
        sampled `time` coordinate of dtype datetime64. The sampling interval
        is read from `time`, not assumed.
    window : int, optional
        Number of samples per correlation window. Windows overlap by half.
        Defaults to 80.
    lowpass : float, optional
        Cut-off frequency [Hz] of the zero-phase low-pass (`lowpassfilter`)
        applied to `t` and `c` before differencing and correlating. Defaults
        to 4.0.

    Returns
    -------
    out : xarray.Dataset
        Dim `segment`. Data variables: `lag` [s] per window, positive
        meaning `t` lags `c`, NaN for a window that holds a NaN in `t` or
        `c` or whose correlation is not finite; `dPdt`, the mean pressure
        rate of change over
        each window, used as a proxy for fall speed; `p`, the mean pressure
        over each window.

    Notes
    -----
    `correlate(ci, ti)` on its own (`ci`, `ti` the differenced, demeaned `c`
    and `t` segments, gvpy `mod.py:1087-1092`) peaks at the negative of the
    delay of `t` behind `c`: correlating a signal `a` against a copy `b`
    delayed by `d` samples peaks at lag `-d`, confirmed independently with a
    plain `numpy.roll` delay test. The sign is flipped here so the returned
    lag matches the documented convention, positive meaning `t` lags `c`,
    which is what `tests/test_tc.py::test_find_lags_recovers_known_lag` pins.

    The raw correlation peak is refined to sub-sample resolution with a
    quadratic fit through the peak and its two neighbors. When the peak
    sits at either end of the correlation array (window index 0 or the
    last), there is no neighbor on one side to fit through; that window's
    lag is returned as the raw (unrefined) peak lag instead of raising
    `IndexError`.

    NaN in `t` or `c` are filled by `_fill_gaps` before the low-pass, since
    `filtfilt` would otherwise spread a single NaN over the whole record.
    The fill only keeps the filter finite; a window holding a filled
    sample gets a NaN lag.
    """
    c = ds.c.data
    t = ds.t.data
    dpdt = ds.dPdt.data
    p = ds.p.data

    dt = sampling_interval(ds.time.data)
    freq = 1 / dt

    def fit_2d_poly(lags, corrs):
        # Fit the quadratic curve
        coefficients = np.polyfit(lags, corrs, 2)
        # Vertex of the parabola
        vertex_x = -coefficients[1] / (2 * coefficients[0])
        return vertex_x

    def find_corrs(t, c, tr):
        # Returns the final, sign-flipped lag estimate for one window (see
        # Notes for the sign flip and the edge fallback).
        ci = np.diff(c[tr] - np.mean(c[tr]))
        ti = np.diff(t[tr] - np.mean(t[tr]))
        correlation = signal.correlate(ci - np.mean(ci), ti - np.mean(ti), mode="full")
        lags = signal.correlation_lags(len(ci), len(ti), mode="full") * 1 / freq
        if not np.isfinite(correlation).all():
            return np.nan
        lag = np.argmax(np.abs(correlation))
        if lag == 0 or lag == len(correlation) - 1:
            return -lags[lag]
        inds = range(lag - 1, lag + 2)
        return -fit_2d_poly(lags[inds], correlation[inds])

    t_filled, t_mask = _fill_gaps(t)
    c_filled, c_mask = _fill_gaps(c)
    gap = t_mask | c_mask
    t_lp = lowpassfilter(t_filled, lowcut=lowpass, fs=freq)
    c_lp = lowpassfilter(c_filled, lowcut=lowpass, fs=freq)

    n = len(t)
    m = window
    logger.info("%d scans, %d segments", n, n // (m // 2))

    start_index = np.arange(0, n - m, m // 2)
    lagi = []
    dpdti = []
    pi = []
    for si in start_index:
        tr = range(si, si + window)
        lag = np.nan if gap[tr].any() else find_corrs(t_lp, c_lp, tr)
        lagi.append(lag)
        dpdti.append(np.mean(dpdt[tr]))
        pi.append(np.mean(p[tr]))

    out = xr.Dataset(
        data_vars=dict(
            lag=("segment", np.array(lagi), dict(long_name="t-c lag", units="s")),
            dPdt=(
                "segment",
                np.array(dpdti),
                dict(long_name="mean pressure rate of change", units="dbar/s"),
            ),
            p=("segment", np.array(pi), dict(long_name="mean pressure", units="dbar")),
        ),
        coords=dict(segment=("segment", np.arange(len(lagi)))),
    )
    return out


def salinity_roughness(
    ds: xr.Dataset, pmin: float | None = None, pmax: float | None = None, edge: float = 2.0
) -> float:
    """RMS second difference of practical salinity, a T-C mismatch cost.

    A lag or thermistor time constant that does not match the true sensor
    response leaves spiky, unphysical structure in salinity computed from
    the mismatched `t`/`c` pair; the rms of the second difference of `SP`
    is small when the pair is well matched and grows with the mismatch.
    Used as the cost function for `lag_tau_cost_map`.

    Parameters
    ----------
    ds : xarray.Dataset
        CTD time series with `SP` and `p` on an evenly sampled `time`
        coordinate of dtype datetime64. The sampling interval is read from
        `time`, not assumed. When a `cast` coordinate is present (integer,
        0 outside casts, as `modfish.fctd.casts.label_casts` produces), the
        second difference is taken separately within each nonzero cast and
        the results pooled; otherwise the whole record is treated as one
        cast.
    pmin, pmax : float or None, optional
        Pressure range [dbar] a sample must be strictly inside to enter the
        cost. `None` (default) for either takes the corresponding bound
        from `add_tcfit_default`.
    edge : float, optional
        Seconds excluded from the start and end of the record (of each cast
        when labeled) before differencing. Defaults to 2.0. Keeps the
        wrapped samples from `response_correction`'s FFT advance and
        low-pass filter transients out of the cost; without it they
        dominate the rms.

    Returns
    -------
    float
        RMS of `np.diff(SP, 2)` over finite samples with
        `pmin < p < pmax`, or `nan` if no such sample exists.
    """
    pmin, pmax = _fit_range_default(ds, pmin, pmax)

    SP = ds["SP"].data
    p = ds["p"].data
    dt = sampling_interval(ds.time.data)
    nedge = int(round(edge / dt))

    if "cast" in ds.coords:
        cast = ds["cast"].data
        segments = [np.flatnonzero(cast == c) for c in np.unique(cast[cast != 0])]
    else:
        segments = [np.arange(SP.size)]

    d2_all = []
    p_all = []
    for seg in segments:
        core = seg[nedge : seg.size - nedge] if nedge > 0 else seg
        if core.size < 3:
            continue
        d2_all.append(np.diff(SP[core], 2))
        p_all.append(p[core][1:-1])

    if not d2_all:
        return float("nan")
    d2_all = np.concatenate(d2_all)
    p_all = np.concatenate(p_all)
    mask = np.isfinite(d2_all) & np.isfinite(p_all) & (p_all > pmin) & (p_all < pmax)
    if not mask.any():
        return float("nan")
    return float(np.sqrt(np.mean(d2_all[mask] ** 2)))


def _grid_eval(grid_a, grid_b, fn) -> np.ndarray:
    """Evaluate `fn(a, b)` over the outer product of two grids.

    Shared by `lag_tau_cost_map` and `thermal_mass_cost_map`.

    Parameters
    ----------
    grid_a, grid_b : array-like
        Grid values along each axis.
    fn : callable
        `fn(a, b) -> float`, called once per grid pair.

    Returns
    -------
    numpy.ndarray
        Shape `(len(grid_a), len(grid_b))`; entry `[i, j]` is
        `fn(grid_a[i], grid_b[j])`.
    """
    grid_a = np.asarray(grid_a)
    grid_b = np.asarray(grid_b)
    cost = np.empty((grid_a.size, grid_b.size))
    for i, a in enumerate(grid_a):
        for j, b in enumerate(grid_b):
            cost[i, j] = fn(a, b)
    return cost


def lag_tau_cost_map(
    ds: xr.Dataset,
    lags,
    taus,
    lowpass: float,
    pmin: float | None = None,
    pmax: float | None = None,
    **correct_kw,
) -> xr.Dataset:
    """Map `salinity_roughness` over a `(lag, tau_t)` grid.

    For each grid pair, `correct(ds, lag=lag, tau_t=tau_t, lowpass=lowpass,
    **correct_kw)` is applied, `SP` is recomputed from the corrected `t`,
    `c`, `p`, and `salinity_roughness` is evaluated over `pmin`/`pmax`. The
    minimum of the returned map is the joint estimate of sensor lag and
    thermistor time constant; `find_lags`'s median lag is drawn on top of
    it as an independent constraint.

    Parameters
    ----------
    ds : xarray.Dataset
        CTD time series with `t`, `c`, `p`, `dPdt` on a `time` coordinate
        of dtype datetime64, as `correct` takes it.
    lags : array-like
        Sensor lag grid [s], `lag > 0` meaning `t` lags `c`.
    taus : array-like
        Thermistor time constant grid [s].
    lowpass : float
        Cut-off frequency [Hz] passed to `correct`.
    pmin, pmax : float or None, optional
        Pressure range [dbar] passed to `salinity_roughness`. `None`
        (default) for either takes the corresponding bound from
        `add_tcfit_default`.
    **correct_kw
        Extra keyword arguments passed through to `correct` (e.g.
        `thermal_mass`, `alpha`, `beta`).

    Returns
    -------
    out : xarray.Dataset
        Dims `(lag, tau_t)`, coordinates `lag` [s] and `tau_t` [s] from
        `lags`/`taus`, data variable `cost` holding the `salinity_roughness`
        at each grid pair.
    """
    pmin, pmax = _fit_range_default(ds, pmin, pmax)

    def fn(lag, tau_t):
        out = correct(ds, lag=lag, tau_t=tau_t, lowpass=lowpass, **correct_kw)
        SP = gsw.SP_from_C(10 * out["c"].data, out["t"].data, out["p"].data)
        out = out.assign(SP=("time", SP))
        return salinity_roughness(out, pmin, pmax)

    cost = _grid_eval(lags, taus, fn)
    return xr.Dataset(
        data_vars=dict(cost=(("lag", "tau_t"), cost)),
        coords=dict(lag=("lag", np.asarray(lags)), tau_t=("tau_t", np.asarray(taus))),
    )


def downup_separation(
    ds: xr.Dataset,
    casts: xr.Dataset,
    tbins,
    pmin: float | None = None,
    pmax: float | None = None,
) -> xr.Dataset:
    """Mean absolute salinity separation between consecutive down/up casts.

    Morison's cost: a down cast and the up cast immediately following it
    sample close to the same water column, so a well-corrected `SP(T)`
    relation should agree between them. `t` and `SP` are binned into
    `tbins`; the cost per pair is the mean absolute difference of the
    binned `SP` over bins where both casts have data.

    Parameters
    ----------
    ds : xarray.Dataset
        CTD time series with `t`, `SP`, `p` and an integer `cast`
        coordinate on `time` (0 outside casts, as
        `modfish.fctd.casts.label_casts` produces).
    casts : xarray.Dataset
        Dim `cast`, data variables `start_time`, `end_time` and
        `direction` (`"down"`/`"up"`), as produced by
        `modfish.fctd.casts.casts_to_dataset` (the `casts` table from
        `make_l1`).
    tbins : array-like
        Temperature bin edges [degC].
    pmin : float or None, optional
        Only samples with `p > pmin` [dbar] enter the cost. `None`
        (default) takes the lower `add_tcfit_default` bound.
    pmax : float or None, optional
        Only samples with `p < pmax` [dbar] enter the cost. `None`
        (default) sets no upper bound. Together with `pmin` this restricts
        the cost to a pressure band, which is how the fall-rate analysis
        compares deployments of different depth range on the band they
        share.

    Returns
    -------
    out : xarray.Dataset
        Dim `pair`, data variable `sep` (mean absolute `SP` separation per
        down/up pair, indexed by the down cast id), attrs `mean` (the mean
        of `sep` over all pairs).
    """
    # pmax=None means no upper bound here, so only pmin takes the default
    pmin, _ = _fit_range_default(ds, pmin, np.inf)

    cast_id = ds["cast"].data
    t = ds["t"].data
    SP = ds["SP"].data
    p = ds["p"].data

    order = np.argsort(casts["start_time"].data)
    ids = casts["cast"].data[order]
    directions = np.asarray(casts["direction"].data)[order]

    # cast labels are contiguous runs in time; index each id's run once so
    # a pair costs O(cast length) and a 4 M-sample record with 300 pairs
    # stays under a second (the masks over the whole record cost 9 s)
    edges = np.flatnonzero(np.diff(cast_id)) + 1
    starts = np.concatenate([[0], edges])
    stops = np.concatenate([edges, [cast_id.size]])
    runs: dict[int, list[tuple[int, int]]] = {}
    for a, b in zip(starts, stops):
        runs.setdefault(int(cast_id[a]), []).append((int(a), int(b)))

    def cast_index(cid):
        spans = runs.get(int(cid), [])
        if not spans:
            return np.zeros(0, dtype=int)
        return np.concatenate([np.arange(a, b) for a, b in spans])

    finite = np.isfinite(t) & np.isfinite(SP) & (p > pmin)
    if pmax is not None:
        finite &= p < pmax

    def binned_mean(idx):
        tv = t[idx]
        sp = SP[idx]
        bidx = np.digitize(tv, tbins) - 1
        out = np.full(len(tbins) - 1, np.nan)
        for k in range(len(tbins) - 1):
            vals = sp[bidx == k]
            if vals.size:
                out[k] = np.mean(vals)
        return out

    seps = []
    pair_ids = []
    for i in range(len(ids) - 1):
        if directions[i] != "down" or directions[i + 1] != "up":
            continue
        di = cast_index(ids[i]); di = di[finite[di]]
        ui = cast_index(ids[i + 1]); ui = ui[finite[ui]]
        if not di.size or not ui.size:
            continue
        down_binned = binned_mean(di)
        up_binned = binned_mean(ui)
        both = np.isfinite(down_binned) & np.isfinite(up_binned)
        if not both.any():
            continue
        seps.append(float(np.mean(np.abs(down_binned[both] - up_binned[both]))))
        pair_ids.append(int(ids[i]))

    seps = np.array(seps)
    out = xr.Dataset(
        data_vars=dict(sep=("pair", seps)),
        coords=dict(pair=("pair", np.array(pair_ids))),
    )
    out.attrs["mean"] = float(np.mean(seps)) if seps.size else float("nan")
    return out


def rosette_rms(
    fctd: xr.Dataset, ctd: xr.Dataset, pmin: float | None = None, pmax: float | None = None
) -> float:
    """RMS of FCTD salinity against a rosette CTD reference profile.

    `fctd.SP` is bin-averaged onto `ctd`'s `depth` grid (bin width taken
    from the median spacing of `ctd.depth`) and compared to the rosette's
    practical salinity `ctd.s1`.

    Parameters
    ----------
    fctd : xarray.Dataset
        FCTD time series with `SP` and `depth` [m] on `time`.
    ctd : xarray.Dataset
        Rosette product on a `depth` [m] dimension coordinate, with
        practical salinity in `s1` (the 1 m binned down-cast product).
    pmin, pmax : float or None, optional
        Depth range [m] a `ctd.depth` bin must be strictly inside to enter
        the rms; named `pmin`/`pmax` for consistency with the other
        estimators, but gates on depth, not pressure. `None` (default) for
        either takes the corresponding `add_tcfit_default` bound computed
        from `fctd`'s pressure `p` (dbar), used here as a depth proxy.

    Returns
    -------
    float
        RMS of `binned_fctd_SP - ctd.s1` over `ctd.depth` bins where both
        are finite and `pmin < depth < pmax`.
    """
    if (pmin is None or pmax is None) and "p" in fctd:
        pmin, pmax = _fit_range_default(fctd, pmin, pmax)

    depth_grid = ctd["depth"].data
    dz = np.median(np.diff(depth_grid))
    edges = np.concatenate([depth_grid - dz / 2, depth_grid[-1:] + dz / 2])

    fctd_depth = fctd["depth"].data
    fctd_sp = fctd["SP"].data
    idx = np.digitize(fctd_depth, edges) - 1
    binned = np.full(depth_grid.size, np.nan)
    for k in range(depth_grid.size):
        vals = fctd_sp[idx == k]
        vals = vals[np.isfinite(vals)]
        if vals.size:
            binned[k] = np.mean(vals)

    ctd_sp = ctd["s1"].data
    mask = np.isfinite(binned) & np.isfinite(ctd_sp)
    if pmin is not None:
        mask &= depth_grid > pmin
    if pmax is not None:
        mask &= depth_grid < pmax
    diff = binned[mask] - ctd_sp[mask]
    return float(np.sqrt(np.mean(diff**2)))


def thermal_mass_cost_map(ds: xr.Dataset, alphas, taus, objective, **objective_kw) -> xr.Dataset:
    """Map an objective over a `(alpha, tau)` thermal-mass parameter grid.

    For each grid pair, `thermal_mass_correction(ds, alpha=alpha,
    beta=1/tau)` is applied, `SP` is recomputed from the corrected `c`, and
    `objective(ds_corr, **objective_kw)` is evaluated. `objective` is
    typically `downup_separation` or `rosette_rms`.

    Parameters
    ----------
    ds : xarray.Dataset
        CTD time series with `t`, `c`, `p` on a `time` coordinate of dtype
        datetime64, as `thermal_mass_correction` takes it.
    alphas : array-like
        Thermal-mass amplitude grid.
    taus : array-like
        Thermal-mass relaxation time grid [s]; `beta = 1 / tau` is passed
        to `thermal_mass_correction`. Must not contain 0.
    objective : callable
        `objective(ds_corr, **objective_kw) -> float`, evaluated on the
        thermal-mass-corrected Dataset (which carries a recomputed `SP`).
    **objective_kw
        Extra keyword arguments passed through to `objective`.

    Returns
    -------
    out : xarray.Dataset
        Dims `(alpha, tau)`, coordinates `alpha` and `tau` [s] from
        `alphas`/`taus`, data variable `cost` holding `objective` at each
        grid pair.
    """

    def fn(alpha, tau):
        out = thermal_mass_correction(ds, alpha=alpha, beta=1 / tau)
        SP = gsw.SP_from_C(10 * out["c"].data, out["t"].data, out["p"].data)
        out = out.assign(SP=("time", SP))
        return objective(out, **objective_kw)

    cost = _grid_eval(alphas, taus, fn)
    return xr.Dataset(
        data_vars=dict(cost=(("alpha", "tau"), cost)),
        coords=dict(alpha=("alpha", np.asarray(alphas)), tau=("tau", np.asarray(taus))),
    )
