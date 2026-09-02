"""Grid stage: per-cast depth gridding of the L1 product.

Consumes the `xr.DataTree` `modfish.fctd.l1.make_l1` produces and bins each
detected cast's `ctd` samples onto a fixed-width depth grid by bin-mean,
producing a rectangular `(depth, cast)` product. Two Matlab bugs are fixed
here: empty bins come out NaN rather than 0 (bug B4), and samples outside
the configured depth range fall out of the binning naturally rather than
being clamped into the edge bins (bug B3). See
`plans/2026-09-01-fctd-pipeline-design.md` for the stage order and
rationale.
"""

import numpy as np
import xarray as xr

from modfish.fctd.config import GridParams

#: `ctd` variables never gridded as data variables: positions (kept
#: instead as per-cast means) and `depth` itself, which defines the
#: grid's coordinate rather than one of its data variables.
_EXCLUDE_VARS = {"lon", "lat", "depth"}


def _depth_edges(depth: np.ndarray, params: GridParams) -> tuple[np.ndarray, np.ndarray]:
    """Depth bin centers and edges for the grid.

    Parameters
    ----------
    depth : np.ndarray
        Per-sample depth, m, used to derive `depth_min`/`depth_max` when
        `params` leaves either as None (floor/ceil of the 0th/100th
        percentile, rounded to `dz`).
    params : GridParams
        Uses `dz`, `depth_min`, `depth_max`.

    Returns
    -------
    centers : np.ndarray
        Bin center depths, m, `np.arange(depth_min, depth_max + dz, dz)`.
    edges : np.ndarray
        Bin edges, m, `centers +/- dz / 2`; length `len(centers) + 1`.
    """
    dz = params.dz
    depth_min = params.depth_min
    depth_max = params.depth_max
    if depth_min is None:
        depth_min = dz * np.floor(np.nanpercentile(depth, 0) / dz)
    if depth_max is None:
        depth_max = dz * np.ceil(np.nanpercentile(depth, 100) / dz)

    centers = np.arange(depth_min, depth_max + dz, dz)
    edges = np.concatenate([centers - dz / 2, [centers[-1] + dz / 2]])
    return centers, edges


def _bin_mean(depth: np.ndarray, values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Bin-mean `values` by `depth` into the bins defined by `edges`.

    Samples where either `depth` or `values` is NaN are masked out before
    binning, so a variable's own NaN edge samples (e.g. from phase-match
    trimming) never suppress a bin that other variables can still fill.
    Bins with zero valid samples come out NaN rather than 0 (Matlab bug
    B4); samples outside `edges` fall out of `np.histogram` on their own
    rather than being clamped into the first/last bin (Matlab bug B3).

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
        Dims `(depth, cast)`. Data variables: every 1-D float `ctd`
        variable on `time` except `lon`, `lat`, and `depth` itself,
        bin-averaged per cast (`p` included, gridded like any other
        variable). Coordinates: `depth` (bin centers, m), `cast` (ids from
        the `casts` group), `direction` (from `casts`), `time`/`lon`/`lat`
        (per-cast means over that cast's samples). Attrs copied from
        `ctd`, plus `dz`.
    """
    if params is None:
        params = GridParams()

    ctd = l1["ctd"].to_dataset()
    casts_ds = l1["casts"].to_dataset()

    cast_ids = casts_ds["cast"].values
    n_casts = cast_ids.size

    depth_all = ctd["depth"].values
    centers, edges = _depth_edges(depth_all, params)
    n_depth = centers.size

    grid_vars = [
        name
        for name, da in ctd.data_vars.items()
        if name not in _EXCLUDE_VARS
        and da.dims == ("time",)
        and np.issubdtype(da.dtype, np.floating)
    ]

    cast_label = ctd["cast"].values
    time_all = ctd["time"].values
    lon_all = ctd["lon"].values
    lat_all = ctd["lat"].values

    gridded = {name: np.full((n_depth, n_casts), np.nan) for name in grid_vars}
    mean_time = np.empty(n_casts, dtype="datetime64[ns]")
    mean_lon = np.full(n_casts, np.nan)
    mean_lat = np.full(n_casts, np.nan)

    for j, cid in enumerate(cast_ids):
        mask = cast_label == cid
        d = depth_all[mask]
        for name in grid_vars:
            gridded[name][:, j] = _bin_mean(d, ctd[name].values[mask], edges)

        mean_time[j] = _mean_time(time_all[mask])
        mean_lon[j] = np.nanmean(lon_all[mask])
        mean_lat[j] = np.nanmean(lat_all[mask])

    coords = {
        "depth": ("depth", centers),
        "cast": ("cast", cast_ids),
        "direction": ("cast", casts_ds["direction"].values),
        "time": ("cast", mean_time),
        "lon": ("cast", mean_lon),
        "lat": ("cast", mean_lat),
    }
    data_vars = {name: (("depth", "cast"), gridded[name]) for name in grid_vars}

    grid = xr.Dataset(data_vars=data_vars, coords=coords)
    for name in grid_vars:
        grid[name].attrs = dict(ctd[name].attrs)
    grid["depth"].attrs = dict(ctd["depth"].attrs)
    grid["lon"].attrs = dict(ctd["lon"].attrs)
    grid["lat"].attrs = dict(ctd["lat"].attrs)

    grid.attrs = dict(ctd.attrs)
    grid.attrs["dz"] = params.dz

    return grid
