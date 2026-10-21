"""Grid stage: per-cast depth gridding of the L1 product.

Consumes the `xr.DataTree` `modfish.fctd.l1.make_l1` produces and bins each
detected cast's `ctd` samples onto a fixed-width depth grid by bin-mean,
producing a rectangular `(depth, cast)` product. Two Matlab gridder bugs
are fixed here. Empty bins come out NaN (Matlab wrote 0). Samples outside
the grid range are dropped (Matlab clamped them into the edge bins).
"""

import numpy as np
import xarray as xr

from modfish.chi.config import FLAG_EMPTY, FLAG_RAIL, FLAG_SLOW
from modfish.fctd.config import GridParams

#: `ctd` variables never gridded as data variables: positions (kept
#: instead as per-cast means) and `depth` itself, which defines the
#: grid's coordinate rather than one of its data variables.
_EXCLUDE_VARS = {"lon", "lat", "depth"}


def depth_edges(depth: np.ndarray, params: GridParams) -> tuple[np.ndarray, np.ndarray]:
    """Depth bin centers and edges for the grid.

    Parameters
    ----------
    depth : np.ndarray
        Per-sample depth, m, used to derive `depth_min`/`depth_max` when
        `params` leaves either as None (min/max of the finite depths,
        rounded outward to a multiple of `dz`).
    params : GridParams
        Uses `dz`, `depth_min`, `depth_max`.

    Returns
    -------
    centers : np.ndarray
        Bin center depths, m, `depth_min + dz * arange(n)` with `n` the
        whole number of steps to `depth_max`, both ends included.
    edges : np.ndarray
        Bin edges, m, `centers +/- dz / 2`; length `len(centers) + 1`.

    Raises
    ------
    ValueError
        If a limit must be derived but `depth` has no finite value, or if
        `depth_min` exceeds `depth_max`.
    """
    dz = params.dz
    depth_min = params.depth_min
    depth_max = params.depth_max
    if depth_min is None or depth_max is None:
        finite = np.asarray(depth, dtype=float)
        finite = finite[np.isfinite(finite)]
        if finite.size == 0:
            raise ValueError("no finite depth to derive the grid range from; "
                             "set GridParams.depth_min and depth_max")
        if depth_min is None:
            depth_min = dz * np.floor(finite.min() / dz)
        if depth_max is None:
            depth_max = dz * np.ceil(finite.max() / dz)
    if depth_min > depth_max:
        raise ValueError(f"depth_min ({depth_min}) exceeds depth_max ({depth_max})")

    # an integer count: np.arange with a float step overshoots depth_max
    # by one bin for steps such as 0.1 that binary floats cannot represent
    n = int(round((depth_max - depth_min) / dz)) + 1
    centers = depth_min + dz * np.arange(n)
    edges = np.concatenate([centers - dz / 2, [centers[-1] + dz / 2]])
    return centers, edges


#: Former private name, imported by downstream analysis scripts.
_depth_edges = depth_edges


def _bin_mean(depth: np.ndarray, values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Bin-mean `values` by `depth` into the bins defined by `edges`.

    Samples where either `depth` or `values` is NaN are masked out before
    binning, so a variable's own NaN samples never suppress a bin that
    other variables can still fill.
    Bins with zero valid samples come out NaN (Matlab wrote 0). Samples
    outside `edges` are dropped by `np.histogram` (Matlab clamped them
    into the first/last bin).

    Parameters
    ----------
    depth : np.ndarray
        Per-sample depth, m.
    values : np.ndarray
        Per-sample values to average, same shape as `depth`.
    edges : np.ndarray
        Bin edges, m, length `n_bins + 1`.

    Returns
    -------
    np.ndarray
        Bin means, length `len(edges) - 1`.
    """
    valid = np.isfinite(depth) & np.isfinite(values)
    d = depth[valid]
    v = values[valid]
    counts, _ = np.histogram(d, edges)
    sums, _ = np.histogram(d, edges, weights=v)
    with np.errstate(invalid="ignore"):
        means = sums / counts
    return means


def _bin_geomean(depth: np.ndarray, values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Geometric bin mean of `values` by `depth` into the bins defined by `edges`.

    `_bin_mean` of `log10(values)`, raised back to linear units. Non-positive
    values (undefined log) are masked with NaN before binning, alongside the
    NaN handling `_bin_mean` already applies.

    Parameters
    ----------
    depth : np.ndarray
        Per-sample depth, m.
    values : np.ndarray
        Per-sample values to average geometrically, same shape as `depth`.
    edges : np.ndarray
        Bin edges, m, length `n_bins + 1`.

    Returns
    -------
    np.ndarray
        Geometric bin means, length `len(edges) - 1`.
    """
    v = np.asarray(values, dtype=float)
    logv = np.where(v > 0, np.log10(np.where(v > 0, v, 1.0)), np.nan)
    return 10.0 ** _bin_mean(depth, logv, edges)


def _bin_or(depth: np.ndarray, flags: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Bitwise or of `flags` by `depth` into the bins defined by `edges`.

    Parameters
    ----------
    depth : np.ndarray
        Per-sample depth, m.
    flags : np.ndarray
        Per-sample uint8 flag bitmask, same shape as `depth`.
    edges : np.ndarray
        Bin edges, m, length `n_bins + 1`.

    Returns
    -------
    np.ndarray
        Bitwise-or of `flags` per bin, dtype uint8, length `len(edges) - 1`.
        0 where the bin has no samples.
    """
    idx = np.digitize(depth, edges) - 1
    out = np.zeros(edges.size - 1, dtype=np.uint8)
    # digitize puts NaN past the last edge, so the upper bound drops it
    ok = (idx >= 0) & (idx < out.size)
    np.bitwise_or.at(out, idx[ok], flags[ok].astype(np.uint8))
    return out


def _mean_time(time: np.ndarray) -> np.datetime64:
    """Mean of a datetime64 array.

    Averages via an offset from the first sample rather than the raw
    nanosecond epoch: that epoch (~1.8e18 ns for present-day timestamps)
    exceeds float64's 2^53 exact-integer range, and summing it directly
    over many samples risks overflowing int64. Offsets within one cast
    (seconds to minutes, i.e. well under 1e15 ns) have neither problem.

    Parameters
    ----------
    time : np.ndarray
        Timestamps, dtype datetime64, at least one entry.

    Returns
    -------
    np.datetime64
        Mean timestamp, dtype datetime64[ns].
    """
    time_ns = time.astype("datetime64[ns]").astype("int64")
    ref = time_ns[0]
    offset_mean = round((time_ns - ref).mean())
    return (ref + np.int64(offset_mean)).astype("datetime64[ns]")


def grid_casts(l1: xr.DataTree, params: GridParams | None = None) -> xr.Dataset:
    """Bin the L1 `ctd` group onto a fixed-width depth grid, per cast.

    Parameters
    ----------
    l1 : xr.DataTree
        Output of `modfish.fctd.l1.make_l1`: `ctd` group (`depth` and an
        int `cast` coord on `time`) and `casts` group (`direction` on dim
        `cast`).
    params : GridParams or None, optional
        Gridding parameters. Defaults to `GridParams()`.

    Returns
    -------
    xr.Dataset
        Dims `(depth, cast)`.

        Data variables: every 1-D float `ctd` variable on `time` except
        `lon`, `lat` and `depth`, bin-averaged per cast (`p` included).
        When the tree has a `chi` group, also `chi`, `chi_tot`, `eps_chi`
        (geometric bin means), `r`, `kmax`, `phi` (bin means) and
        `chi_flag` (bitwise or), each present in `chi`.

        Coordinates: `depth` (bin centers, m, from `depth_edges`), `cast`
        (ids from the `casts` group), `direction` (from `casts`), and
        `time`/`lon`/`lat` (per-cast means over that cast's samples; NaT
        or NaN for a cast without labeled samples).

        Attrs: copied from `ctd`, plus `dz` and `closure_exclude_flags`.
        With a `chi` group also `chi_gain`, `chi_gain_source`,
        `chi_antialias`, `chi_modfish_version` and, when present,
        `chi_modfish_commit`.

    Raises
    ------
    ValueError
        From `depth_edges`, if `params.depth_min` or `params.depth_max` is
        None and no cast sample has a finite depth, or if the resulting
        `depth_min` exceeds `depth_max`.

    Notes
    -----
    Only samples with a cast label (`cast > 0`) set a derived depth range,
    so deck and surface-soak samples do not stretch the grid.

    Chi windows flagged `FLAG_SLOW`, `FLAG_EMPTY` or `FLAG_RAIL` are
    excluded from every chi bin mean. Windows carrying a bit of
    `params.closure_exclude_flags` are also excluded from the `chi_tot`,
    `eps_chi` and `r` means. A railed window's chi is set by the
    saturation. On the MOTIVE products railed windows are 0.05 % of
    windows and 93 % of those above 1e-5 K^2/s. `chi_flag`'s bitwise or
    still covers every window, so a bin can have those bits set without
    the flagged windows having contributed to the mean. Bit 2
    (noise-dominated) does not exclude a window from the means, since a
    small signal is a valid estimate.

    The gridded `chi` is censored at zero. `_bin_geomean` masks
    non-positive values to NaN before averaging, so a negative chi never
    reaches the gridded product even though the window-level value keeps
    it. Within a bin, `chi` averages only the non-excluded windows that
    are also positive, while `phi` averages every non-excluded window. The
    gap between those two window sets widens where `phi` is high, so a
    high `phi` in a bin says nothing by itself about the windows behind
    that bin's `chi`.
    """
    if params is None:
        params = GridParams()

    ctd = l1["ctd"].to_dataset()
    casts_ds = l1["casts"].to_dataset()

    cast_ids = casts_ds["cast"].values
    n_casts = cast_ids.size

    depth_all = ctd["depth"].values
    cast_label = ctd["cast"].values
    # only cast samples are gridded, so deck and surface-soak samples must
    # not stretch a derived depth range
    centers, edges = depth_edges(depth_all[cast_label > 0], params)
    n_depth = centers.size

    grid_vars = [
        name
        for name, da in ctd.data_vars.items()
        if name not in _EXCLUDE_VARS
        and da.dims == ("time",)
        and np.issubdtype(da.dtype, np.floating)
    ]

    time_all = ctd["time"].values
    lon_all = ctd["lon"].values
    lat_all = ctd["lat"].values

    gridded = {name: np.full((n_depth, n_casts), np.nan) for name in grid_vars}
    mean_time = np.full(n_casts, np.datetime64("NaT", "ns"))
    mean_lon = np.full(n_casts, np.nan)
    mean_lat = np.full(n_casts, np.nan)

    for j, cid in enumerate(cast_ids):
        mask = cast_label == cid
        if not mask.any():
            # a cast in the table without labeled samples, e.g. a time
            # subset of an L1 tree: leave its column NaN and its time NaT
            continue
        d = depth_all[mask]
        for name in grid_vars:
            gridded[name][:, j] = _bin_mean(d, ctd[name].values[mask], edges)

        mean_time[j] = _mean_time(time_all[mask])
        # lon is all NaN on a fallback-latitude run; skip nanmean's
        # empty-slice warning, once per cast
        for out, vals in ((mean_lon, lon_all[mask]), (mean_lat, lat_all[mask])):
            if np.isfinite(vals).any():
                out[j] = np.nanmean(vals)

    chi_grid = {}
    if "chi" in l1.children:
        chi = l1["chi"].to_dataset()
        geo = [n for n in ("chi", "chi_tot", "eps_chi") if n in chi]
        arith = [n for n in ("r", "kmax", "phi") if n in chi]
        for name in geo + arith:
            chi_grid[name] = np.full((n_depth, n_casts), np.nan)
        chi_grid["chi_flag"] = np.zeros((n_depth, n_casts), dtype=np.uint8)
        chi_cast = chi["cast"].values
        chi_depth = chi["depth"].values
        chi_flag_all = chi["chi_flag"].values
        # flags 1 (FLAG_SLOW), 4 (FLAG_EMPTY) and 32 (FLAG_RAIL) exclude
        # a window from the means, but chi_flag's bitwise-or still covers
        # every window of the cast, so a bin can carry those bits even
        # though the flagged windows did not contribute to
        # chi/chi_tot/eps_chi/r/kmax/phi.
        excluded = (chi_flag_all.astype(np.uint8) & (FLAG_SLOW | FLAG_EMPTY | FLAG_RAIL)) != 0
        # params.closure_exclude_flags adds bits for the closure-derived
        # fields only; chi itself does not pass through the closure.
        closure_excluded = excluded | (
            (chi_flag_all.astype(np.uint8) & np.uint8(params.closure_exclude_flags)) != 0
        )
        closure_fields = {"chi_tot", "eps_chi", "r"}
        for j, cid in enumerate(cast_ids):
            m = chi_cast == cid
            if not m.any():
                continue
            d = chi_depth[m]
            for name in geo:
                excl = (closure_excluded if name in closure_fields else excluded)[m]
                vals = np.where(excl, np.nan, chi[name].values[m])
                chi_grid[name][:, j] = _bin_geomean(d, vals, edges)
            for name in arith:
                excl = (closure_excluded if name in closure_fields else excluded)[m]
                vals = np.where(excl, np.nan, chi[name].values[m])
                chi_grid[name][:, j] = _bin_mean(d, vals, edges)
            chi_grid["chi_flag"][:, j] = _bin_or(d, chi_flag_all[m], edges)

    coords = {
        "depth": ("depth", centers),
        "cast": ("cast", cast_ids),
        "direction": ("cast", casts_ds["direction"].values),
        "time": ("cast", mean_time),
        "lon": ("cast", mean_lon),
        "lat": ("cast", mean_lat),
    }
    data_vars = {name: (("depth", "cast"), gridded[name]) for name in grid_vars}
    for name, arr in chi_grid.items():
        data_vars[name] = (("depth", "cast"), arr)

    grid = xr.Dataset(data_vars=data_vars, coords=coords)
    for name in grid_vars:
        grid[name].attrs = dict(ctd[name].attrs)
    grid["depth"].attrs = dict(ctd["depth"].attrs)
    grid["lon"].attrs = dict(ctd["lon"].attrs)
    grid["lat"].attrs = dict(ctd["lat"].attrs)

    grid.attrs = dict(ctd.attrs)
    grid.attrs["dz"] = params.dz
    grid.attrs["closure_exclude_flags"] = int(params.closure_exclude_flags)

    if chi_grid:
        for name in chi_grid:
            grid[name].attrs = dict(chi[name].attrs)
        for key in ("gain", "gain_source", "antialias", "modfish_version"):
            grid.attrs[f"chi_{key}"] = chi.attrs[key]
        if "modfish_commit" in chi.attrs:
            grid.attrs["chi_modfish_commit"] = chi.attrs["modfish_commit"]

    return grid
