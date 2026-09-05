"""Assemble the `/chi` group of an L1 tree from the L0 `efe/c1` files."""

import importlib.metadata

import numpy as np
import pandas as pd
import xarray as xr
from scipy.ndimage import uniform_filter1d

from modfish.chi.batchelor import FractionTable
from modfish.chi.closure import closure, stratification
from modfish.chi.config import FLAG_MEANINGS, FLAG_NOENV, ChiParams
from modfish.chi.load import load_c1, range_time
from modfish.chi.spectra import dtdc, run_range, window_slices
from modfish.utils import sampling_interval


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
        `p`, `lon`, `lat`, `spd`, `chi`, `kmax`, `n_bins`, `range_id`,
        `chi_flag`, coordinate `cast`, and, when `params.closure` is
        True, `chi_tot`, `eps_chi`, `r`, `n2`, `Tz`, `Sz`, `Rrho`. Group
        attrs carry every `ChiParams` field, `flag_meanings`, `range_fs`,
        `n_ranges`, `n_windows` and `modfish_version`.

    Raises
    ------
    ValueError
        When `params.enabled` is False or `params.gain` is None, or when
        no range yields a full window.
    """
    if not params.enabled or params.gain is None:
        raise ValueError("add_chi needs ChiParams with enabled=True and a gain")
    fs16 = 1.0 / sampling_interval(ctd["time"].values)
    time_ns = ctd["time"].values.astype("datetime64[ns]").astype("int64")
    spd16 = np.gradient(ctd["depth"].values.astype(float)) * fs16
    spd16 = np.abs(uniform_filter1d(spd16, max(int(round(params.spd_smooth * fs16)), 1), mode="nearest"))

    pieces = []
    for rid, r in ranges.iterrows():
        if not np.isfinite(r.fs) or r.n < 2:
            continue
        starts, centers_s = window_slices(int(r.n), float(r.fs), params)
        if starts.size == 0:
            continue
        t_range = range_time(r.start, r.n, r.fs)
        centers_ns = t_range[0].astype("int64") + np.round(centers_s * 1e9).astype("int64")
        env = {name: _interp_at(centers_ns, time_ns, ctd[name].values)
               for name in ("depth", "p", "lon", "lat", "t", "SP")}
        spd = _interp_at(centers_ns, time_ns, spd16)
        dt_dc = dtdc(env["SP"], env["t"], env["p"])
        seg = c1[int(r.i0):int(r.i0) + int(r.n)]
        out = run_range(seg, float(r.fs), spd, dt_dc, params)
        pieces.append(xr.Dataset(
            dict(depth=("time", env["depth"]), p=("time", env["p"]), lon=("time", env["lon"]),
                 lat=("time", env["lat"]), spd=("time", spd), chi=("time", out["chi"]),
                 kmax=("time", out["kmax"]), n_bins=("time", out["n_bins"]),
                 range_id=("time", np.full(starts.size, int(rid), dtype=int)),
                 chi_flag=("time", out["flag"])),
            coords=dict(time=centers_ns.astype("datetime64[ns]"))))
    if not pieces:
        raise ValueError("no full chi window in any range")
    ds = xr.concat(pieces, dim="time")
    ds = ds.assign_coords(cast=("time", _label_casts(ds["time"].values, casts)))

    if params.closure:
        # Controller ruling: kmax_max = params.kmax_cap, not
        # max(kmax_cap, fmax_cap / min_spd) -- run_range never returns a
        # kmax above kmax_cap, so a wider grid only coarsens the r(eps,
        # kmax) interpolation over the kmax range that is actually used.
        table = FractionTable.build(params.kmin, params.kmax_cap,
                                    params.nu, params.D, params.q)
        strat = stratification(ctd, ds["time"].values, params)
        clo = closure(ds["chi"].values, ds["kmax"].values, strat, params, table)
        ds = ds.assign(**{k: strat[k] for k in ("n2", "Tz", "Sz", "Rrho")},
                       chi_tot=clo["chi_tot"], eps_chi=clo["eps_chi"], r=clo["r"])
        ds["chi_flag"] = ("time", (ds["chi_flag"].values | clo["flag"].values).astype(np.uint8))

    ds["chi"].attrs = dict(long_name="temperature-gradient variance dissipation, resolved band", units="K^2/s")
    ds["kmax"].attrs = dict(long_name="upper integration limit", units="cpm")
    ds["spd"].attrs = dict(long_name="fall rate", units="m/s")
    ds["depth"].attrs = dict(ctd["depth"].attrs)
    ds["chi_flag"].attrs = dict(long_name="quality flag bitmask", flag_meanings=FLAG_MEANINGS)
    attrs = {k: (v if v is not None else "") for k, v in vars(params).items()}
    attrs["flag_meanings"] = FLAG_MEANINGS
    attrs["range_fs"] = [float(x) for x in ranges.fs.values]
    attrs["n_ranges"] = int(len(ranges))
    attrs["n_windows"] = int(ds.sizes["time"])
    attrs["modfish_version"] = importlib.metadata.version("modfish")
    attrs["enabled"] = int(params.enabled)
    attrs["closure"] = int(params.closure)
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
