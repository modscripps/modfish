"""Assemble the `/chi` group of an L1 tree from the L0 `efe/c1` files."""

import importlib.metadata
import logging

import numpy as np
import pandas as pd
import xarray as xr
from scipy.ndimage import uniform_filter1d

from modfish.chi.batchelor import FractionTable
from modfish.chi.closure import closure, stratification
from modfish.chi.config import FLAG_MEANINGS, FLAG_NOENV, ChiParams
from modfish.chi.load import load_c1
from modfish.chi.noise import resolve
from modfish.chi.spectra import dtdc, run_range, window_slices
from modfish.utils import sampling_interval

logger = logging.getLogger(__name__)


def _interp_at(centers_ns, time_ns, values):
    """Linear interpolation of one CTD variable onto window centers.

    Parameters
    ----------
    centers_ns : numpy.ndarray of int64
        Window center times, nanoseconds since the epoch.
    time_ns : numpy.ndarray of int64
        CTD timestamps, nanoseconds since the epoch, increasing.
    values : array_like
        CTD variable, same length as `time_ns`.

    Returns
    -------
    numpy.ndarray of float
        `values` linearly interpolated at `centers_ns`. NaN outside the
        span of `time_ns`.
    """
    return np.interp(centers_ns, time_ns, np.asarray(values, dtype=float), left=np.nan, right=np.nan)


def _fraction_table(params: ChiParams) -> FractionTable:
    """Batchelor band-fraction table over the kmax range `run_range` returns.

    `integrate` reports the upper edge of the band it summed,
    `k_last + dk/2`, so the recorded `kmax` exceeds `params.kmax_cap` by up
    to half a Welch bin. The wavenumber bin width is `1 / (nsec * spd)` and
    the widest bin comes from the slowest window the pipeline keeps, so the
    table must reach `kmax_cap + 0.5 / (nsec * min_spd)` (14.5 cpm at the
    defaults) or `r_of` clamps and evaluates `r` over a narrower band than
    the sum, biasing `chi_tot` and the eps solve high by a few percent.

    Parameters
    ----------
    params : ChiParams
        Supplies `kmin`, `kmax_cap`, `nsec`, `min_spd`, `nu`, `D` and `q`.

    Returns
    -------
    FractionTable
        Grid from `kmin + 0.5` to that ceiling at about 0.5 cpm spacing.
    """
    kmax_max = params.kmax_cap + 0.5 / (params.nsec * params.min_spd)
    n_kmax = max(int(round((kmax_max - params.kmin - 0.5) / 0.5)) + 1, 2)
    return FractionTable.build(params.kmin, kmax_max, params.nu, params.D,
                               params.q, n_kmax=n_kmax)


def _label_casts(centers, casts: xr.Dataset):
    """Cast id of each window center, 0 outside every detected cast.

    Parameters
    ----------
    centers : numpy.ndarray
        datetime64, window center times.
    casts : xarray.Dataset
        `casts` group with dim `cast` and variables `start_time`,
        `end_time`.

    Returns
    -------
    numpy.ndarray of int
        Cast id at each center, 0 where the center falls outside every
        cast interval.
    """
    label = np.zeros(centers.size, dtype=int)
    for cid, t0, t1 in zip(casts["cast"].values, casts["start_time"].values, casts["end_time"].values):
        label[(centers >= t0) & (centers <= t1)] = int(cid)
    return label


def _record_floor(diag, nu, deep_frac=80, qs=(0.02, 0.05, 0.30),
                  ref_f=(4.0, 10.0, 20.0, 37.5)):
    """The record's own floor estimate and its lower-tail flatness.

    Pools the subsampled raw PSDs kept by `run_range`, takes the deepest
    `100 - deep_frac` percent of them and inverts the chi-square quantile,
    `N_hat(q) = P_q / Q(q)`. The result is an upper bound on this record's
    floor. Flatness is `N_hat(0.30) / N_hat(0.02)` at 20 Hz; near 1 means a
    stationary additive level dominates the tail.

    Returns `(ref_f, N_hat at q=0.05, flatness)`, all NaN when too few
    spectra were kept.
    """
    from scipy.stats import chi2

    ref = np.asarray(ref_f, dtype=float)
    if not diag:
        return ref, np.full(ref.size, np.nan), np.nan
    # Each range carries its own fs, and run_range's Welch grid depends on
    # fs, so two ranges of one deployment can produce frequency grids of
    # different length (or the same length from a different fs). Ranges in
    # one deployment do not always share a grid, so pool only the entries
    # that share one, keeping the largest group.
    groups = {}
    for Pf, dep, fg in diag:
        groups.setdefault(np.asarray(fg, dtype=float).tobytes(), []).append((Pf, dep, fg))
    best = max(groups.values(), key=lambda g: sum(int(d[0].shape[0]) for d in g))
    P = np.concatenate([d[0] for d in best])
    dep = np.concatenate([d[1] for d in best])
    f = np.asarray(best[0][2], dtype=float)
    good = np.isfinite(dep)
    if good.sum() < 200:
        return ref, np.full(ref.size, np.nan), np.nan
    P, dep = P[good], dep[good]
    deep = dep >= np.nanpercentile(dep, deep_frac)
    if deep.sum() < 200:
        return ref, np.full(ref.size, np.nan), np.nan
    Q = chi2.ppf(np.asarray(qs), nu) / nu
    emp = np.percentile(P[deep], 100 * np.asarray(qs), axis=0)
    nhat = emp / Q[:, None]
    j = int(np.argmin(np.abs(f - 20.0)))
    flat = float(nhat[2, j] / nhat[0, j])
    at_ref = 10 ** np.interp(ref, f, np.log10(nhat[1]))
    return ref, at_ref, flat


def chi_dataset(ctd: xr.Dataset, casts: xr.Dataset, c1, ranges: pd.DataFrame,
                params: ChiParams) -> xr.Dataset:
    """Build the window-level chi Dataset from an L1 `ctd` group, its
    `casts` table and the loaded `efe/c1` column.

    Every gap-free range in `ranges` is windowed with `window_slices`, its
    environment (`depth`, `p`, `lon`, `lat`, `t`, `SP`, fall rate) is
    interpolated from `ctd` onto the window centers, and `run_range`
    computes chi over the window. Ranges are concatenated along `time`.
    When `params.closure` is set, `stratification` and `closure` add
    `chi_tot`, `eps_chi`, `r`, `n2`, `Tz`, `Sz`, `Rrho` and their flag bits
    are folded into `chi_flag`.

    Parameters
    ----------
    ctd : xarray.Dataset
        L1 `ctd` group (16 Hz), with `time`, `depth`, `p`, `lon`, `lat`,
        `t`, `SP`, `sgth0`.
    casts : xarray.Dataset
        L1 `casts` group, dim `cast`, with `start_time`, `end_time`.
    c1 : numpy.ndarray
        float32 volts, the concatenated `efe/c1` stream from `load_c1`.
    ranges : pandas.DataFrame
        Gap-free range table from `load_c1` (columns `i0`, `n`, `start`,
        `fs`).
    params : ChiParams
        `enabled` must be True and `gain` set.

    Returns
    -------
    xarray.Dataset
        On dim `time` (window centers), with data variables `depth`,
        `p`, `lon`, `lat`, `spd`, `chi`, `phi`, `kmax`, `n_bins`,
        `range_id`, `chi_flag`, coordinate `cast`, and, when
        `params.closure` is True, `chi_tot`, `eps_chi`, `r`, `n2`, `Tz`,
        `Sz`, `Rrho`. Group attrs carry every `ChiParams` field,
        `flag_meanings`, `range_fs`, `n_ranges`, `n_windows` and
        `modfish_version`, and, when `params.noise` resolves to a floor,
        that floor's provenance (`noise_source`, `noise_records`,
        `noise_measured`, `noise_nu`) and the record's own floor estimate
        (`record_floor_f`, `record_floor_n`, `record_flatness_20hz`) for
        comparison; the estimate never feeds back into the subtraction.

    Raises
    ------
    ValueError
        When `params.enabled` is False or `params.gain` is None, or when
        no range yields a full window.
    """
    if not params.enabled or params.gain is None:
        raise ValueError("add_chi needs ChiParams with enabled=True and a gain")
    floor = resolve(params.noise)
    fs16 = 1.0 / sampling_interval(ctd["time"].values)
    time_ns = ctd["time"].values.astype("datetime64[ns]").astype("int64")
    spd16 = np.gradient(ctd["depth"].values.astype(float)) * fs16
    spd16 = np.abs(uniform_filter1d(spd16, max(int(round(params.spd_smooth * fs16)), 1), mode="nearest"))

    # _record_floor pools across ranges, so size the stride from the
    # deployment's total window count. Targeting about 5000 retained
    # spectra leaves roughly 1000 in the deepest 20 percent whatever way
    # the record is split into ranges. 5000 x 82 bins x 8 bytes is 3.3 MB.
    total_win = sum(int(r.n) / (params.step * float(r.fs))
                    for _, r in ranges.iterrows()
                    if np.isfinite(r.fs) and r.n >= 2)
    stride = 0 if floor is None else max(int(round(total_win / 5000)), 1)

    pieces = []
    diag = []
    for rid, r in ranges.iterrows():
        if not np.isfinite(r.fs) or r.n < 2:
            continue
        starts, centers_s = window_slices(int(r.n), float(r.fs), params)
        if starts.size == 0:
            continue
        # Only the range start is needed here (element 0 of range_time's
        # full per-sample grid, by construction); building that whole
        # grid just to read one scalar is a multi-hundred-MB transient
        # on a single-range 12 h deployment at 320 Hz.
        start_ns = np.datetime64(r.start, "ns").astype("int64")
        centers_ns = start_ns + np.round(centers_s * 1e9).astype("int64")
        env = {name: _interp_at(centers_ns, time_ns, ctd[name].values)
               for name in ("depth", "p", "lon", "lat", "t", "SP")}
        spd = _interp_at(centers_ns, time_ns, spd16)
        dt_dc = dtdc(env["SP"], env["t"], env["p"])
        seg = c1[int(r.i0):int(r.i0) + int(r.n)]
        out = run_range(seg, float(r.fs), spd, dt_dc, params,
                        noise=floor, diag_stride=stride)
        pieces.append(xr.Dataset(
            dict(depth=("time", env["depth"]), p=("time", env["p"]), lon=("time", env["lon"]),
                 lat=("time", env["lat"]), spd=("time", spd), chi=("time", out["chi"]),
                 phi=("time", out["phi"]),
                 kmax=("time", out["kmax"]), n_bins=("time", out["n_bins"]),
                 range_id=("time", np.full(starts.size, int(rid), dtype=int)),
                 chi_flag=("time", out["flag"])),
            coords=dict(time=centers_ns.astype("datetime64[ns]"))))
        if out["diag_idx"].size:
            diag.append((out["diag_Pf"], env["depth"][out["diag_idx"]], out["diag_f"]))
    if not pieces:
        raise ValueError("no full chi window in any range")
    ds = xr.concat(pieces, dim="time")
    ds = ds.assign_coords(cast=("time", _label_casts(ds["time"].values, casts)))

    if params.closure:
        # the table reaches half a Welch bin above kmax_cap because that is
        # where run_range's band edge can land; see _fraction_table
        table = _fraction_table(params)
        strat = stratification(ctd, ds["time"].values, params)
        clo = closure(ds["chi"].values, ds["kmax"].values, strat, params, table)
        ds = ds.assign(**{k: strat[k] for k in ("n2", "Tz", "Sz", "Rrho")},
                       chi_tot=clo["chi_tot"], eps_chi=clo["eps_chi"], r=clo["r"])
        flag = ds["chi_flag"].values | clo["flag"].values
        # a window with a spectrum but no closure input is missing
        # environment (bit 64), not inverted stratification (bit 16)
        noenv = np.isfinite(ds["chi"].values) & ~(
            np.isfinite(strat["n2"].values)
            & np.isfinite(strat["alpha"].values)
            & np.isfinite(strat["Rrho"].values))
        flag[noenv] |= FLAG_NOENV
        ds["chi_flag"] = ("time", flag.astype(np.uint8))

    ds["chi"].attrs = dict(long_name="temperature-gradient variance dissipation, resolved band", units="K^2/s")
    ds["phi"].attrs = dict(long_name="noise fraction of the band integral", units="1")
    ds["kmax"].attrs = dict(long_name="upper integration limit", units="cpm")
    ds["spd"].attrs = dict(long_name="fall rate", units="m/s")
    for name in ("depth", "p", "lon", "lat"):
        ds[name].attrs = dict(ctd[name].attrs)
    ds["chi_flag"].attrs = dict(long_name="quality flag bitmask", flag_meanings=FLAG_MEANINGS)
    attrs = {k: (v if v is not None else "") for k, v in vars(params).items()}
    attrs["flag_meanings"] = FLAG_MEANINGS
    attrs["range_fs"] = [float(x) for x in ranges.fs.values]
    attrs["n_ranges"] = int(len(ranges))
    attrs["n_windows"] = int(ds.sizes["time"])
    attrs["modfish_version"] = importlib.metadata.version("modfish")
    attrs["enabled"] = int(params.enabled)
    attrs["closure"] = int(params.closure)
    if floor is not None:
        attrs["noise_source"] = floor.source
        attrs["noise_records"] = floor.records
        attrs["noise_measured"] = floor.measured
        attrs["noise_nu"] = float(floor.nu)
        rf, rn, flat = _record_floor(diag, floor.nu)
        attrs["record_floor_f"] = [float(v) for v in rf]
        attrs["record_floor_n"] = [float(v) for v in rn]
        attrs["record_flatness_20hz"] = float(flat)
        ref = floor.at(rf)
        ratio_20 = float(rn[2] / ref[2])
        if np.all(np.isfinite(rn)) and ratio_20 > 10 and flat < 1.6:
            logger.warning(
                "record floor estimate is %.1fx the shipped floor at 20 Hz "
                "with a flat lower tail (%.2f). The shipped floor may not "
                "suit this record; see chi.noise per-entry override.",
                ratio_20, flat)
    ds.attrs = attrs
    return ds


def add_chi(l1: xr.DataTree, l0_files, params: ChiParams) -> xr.DataTree:
    """Return `l1` with a `/chi` group computed from `l0_files`' `efe/c1`.

    Loads and range-splits the raw conductivity-channel stream
    (`load_c1`), builds the window-level product (`chi_dataset`) from the
    L1 `ctd` and `casts` groups, and assembles a new tree carrying every
    existing group over unchanged plus the new `chi` group.

    Parameters
    ----------
    l1 : xr.DataTree
        Output of `modfish.fctd.l1.make_l1` (groups `ctd`, `casts`, ...).
    l0_files : sequence of Path
        Per-file L0 netCDF paths of the same deployment.
    params : ChiParams
        `enabled` must be True and `gain` set.

    Returns
    -------
    xr.DataTree
        A new tree: every existing group carried over, plus `chi`.

    Raises
    ------
    ValueError
        When `params.enabled` is False or `params.gain` is None, or
        propagated from `chi_dataset`/`load_c1` when no data survive.
    """
    if not params.enabled or params.gain is None:
        raise ValueError("add_chi needs ChiParams with enabled=True and a gain")
    c1, ranges = load_c1(l0_files, gap=params.gap)
    ctd = l1["ctd"].to_dataset()
    casts = l1["casts"].to_dataset()
    ds = chi_dataset(ctd, casts, c1, ranges, params)
    groups = {f"/{name}": child.to_dataset() for name, child in l1.children.items()}
    groups["/chi"] = ds
    out = xr.DataTree.from_dict(groups)
    out.attrs = dict(l1.attrs)
    return out
