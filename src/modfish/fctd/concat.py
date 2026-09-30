"""Deployment-level concatenation of per-file L0 DataTrees.

Consumes the netCDF layout `modfish.modraw.read()` produces (one group per
decoded stream, plus a root dataset of per-block clock forensics on dim
`block`): per-file `.nc` files as written by `modfish.modraw.convert`.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

logger = logging.getLogger(__name__)

#: Consecutive-file time ranges overlapping by more than this many seconds
#: are logged as a warning.
_OVERLAP_WARN_S = 5.0

#: Consecutive-file time ranges gapping by more than this many seconds are
#: logged as a warning.
_GAP_WARN_S = 60.0

#: A `ctd` time step longer than this many seconds is an acquisition gap for
#: the stale-sample test (`_drop_stale_after_gap`).
_STALE_GAP_S = 0.5

#: The first `ctd` sample after a gap is dropped when its pressure departs
#: from the linear extrapolation of the next two samples by more than this
#: many dbar. On MOTIVE the stale samples sat 30 to 50 dbar off; the other
#: 155 first-after-gap samples stayed within 1.5 dbar (99th percentile
#: 0.6), modscripps/modfish#41.
_STALE_DBAR = 5.0

#: A backward time step this close to a file's start marks every record
#: before it as a misstamped leading block (`_drop_misstamped`). MOTIVE's
#: blocks were two to four records long.
_LEADING_RECORDS = 16


def _load_groups(path: Path, groups=None) -> dict:
    """Open one file's L0 DataTree, load its groups into memory, and close it.

    Loading eagerly and closing immediately keeps at most one file handle
    open at a time; a deployment can span hundreds of files, and holding
    every `xr.open_datatree` handle open until the whole concatenation
    finishes risks exhausting file descriptors.

    Parameters
    ----------
    path : Path
        Per-file L0 netCDF path.
    groups : iterable of str or None, optional
        Group names to load. None (default) loads every child group. A
        name absent from this file is skipped without error.

    Returns
    -------
    dict of str -> xr.Dataset
        In-memory group datasets, keyed by group name. The root (block
        forensics, dim `block`) is excluded: it is never a child of itself
        in `tree.children`.
    """
    wanted = None if groups is None else set(groups)
    with xr.open_datatree(path) as tree:
        return {
            name: node.ds.load()
            for name, node in tree.children.items()
            if wanted is None or name in wanted
        }


def _drop_misstamped(ds: xr.Dataset) -> tuple[xr.Dataset, int]:
    """Drop records stamped later than a record that follows them in the file.

    After an acquisition restart the first two to four SBE49 records of a
    file can carry timestamps 0.2 to 2.3 s late while their values continue
    into the records that follow (modscripps/modfish#41). Sorted by time,
    they would interleave with the real record. A backward time step within
    the first 16 records drops every record before it, since a block can be
    stamped late by less than its own length. Anywhere in the file, a record
    stamped later than any record after it is dropped. Exact duplicate timestamps
    are removed first (first occurrence kept, as `concat_l0` does), so a
    repeated record is not counted here.

    Parameters
    ----------
    ds : xr.Dataset
        One file's `ctd` group, in file order.

    Returns
    -------
    ds : xr.Dataset
        The group without duplicate or misstamped records.
    n : int
        Number of misstamped records dropped.
    """
    t = ds["time"].values
    dup = pd.Index(t).duplicated()
    if dup.any():
        ds = ds.isel(time=~dup)
        t = ds["time"].values
    if t.size < 2:
        return ds, 0
    later_min = np.minimum.accumulate(t[::-1])[::-1]
    bad = np.zeros(t.size, dtype=bool)
    bad[:-1] = t[:-1] > later_min[1:]
    back = np.flatnonzero(t[1:] < t[:-1]) + 1
    back = back[back <= _LEADING_RECORDS]
    if back.size:
        bad[: back.max()] = True
    if bad.any():
        ds = ds.isel(time=~bad)
    return ds, int(bad.sum())


def _drop_stale_after_gap(ds: xr.Dataset) -> tuple[xr.Dataset, int]:
    """Drop a first sample after a gap whose pressure is not in sequence.

    After an acquisition gap the first `ctd` sample can be a reading taken
    15 to 20 s earlier, stamped as if it were current (modscripps/modfish#41).
    Its `t`, `c` and `p` are all stale, so the whole sample is dropped. A
    first-after-gap sample is tested against the linear extrapolation of the
    two samples that follow it, and only where those two are themselves
    gap-free. The first sample of the record is never tested.

    Parameters
    ----------
    ds : xr.Dataset
        Concatenated `ctd` group, sorted by time and deduplicated.

    Returns
    -------
    ds : xr.Dataset
        The group without stale samples.
    n : int
        Number of samples dropped.
    """
    t = ds["time"].values
    if t.size < 4 or "p" not in ds:
        return ds, 0
    p = ds["p"].values
    gap = np.diff(t) / np.timedelta64(1, "s") > _STALE_GAP_S
    k = np.flatnonzero(gap) + 1
    k = k[k + 2 < t.size]
    k = k[~gap[k] & ~gap[k + 1]]
    off = np.abs(p[k] - (2 * p[k + 1] - p[k + 2]))
    bad = k[off > _STALE_DBAR]
    if bad.size:
        keep = np.ones(t.size, dtype=bool)
        keep[bad] = False
        ds = ds.isel(time=keep)
    return ds, int(bad.size)


def _time_range(groups: dict):
    """Overall (min, max) time span covered by a file's groups.

    Union across all groups present, so the check does not depend on which
    stream happens to be present in every file.

    Parameters
    ----------
    groups : dict of str -> xr.Dataset
        One file's group datasets, as returned by `_load_groups`.

    Returns
    -------
    tuple of np.datetime64 or None
        `(tmin, tmax)`, or None if no group has any `time` samples.
    """
    times = [
        ds["time"].values
        for ds in groups.values()
        if "time" in ds.coords and ds.sizes.get("time", 0)
    ]
    if not times:
        return None
    all_t = np.concatenate(times)
    return all_t.min(), all_t.max()


def concat_l0(files: list, keep_counts: bool = False, groups=None) -> xr.DataTree:
    """Concatenate per-file L0 DataTrees into one deployment-level DataTree.

    Parameters
    ----------
    files : list of Path or str
        Per-file L0 netCDF paths. Order does not matter for the result
        (each group is sorted by time), but consecutive entries are checked
        for overlap/gap and named in warnings in the order given.
    keep_counts : bool, optional
        Keep the raw-counts variables (`t_raw`, `c_raw`, `p_raw`, `pt_raw`,
        `bb_raw`, `chla_raw`, `fdom_raw`) in the `ctd` and `ecop` groups.
        Default False (drop them).
    groups : iterable of str or None, optional
        L0 group names to load and concatenate, e.g. ``("ctd", "gps")``.
        None (default) loads every group present. `ctd` must be among the
        selection, or nothing survives and `ValueError` is raised. Selecting
        groups is how a deployment whose `efe` stream would not fit in
        memory is processed for its CTD product alone.

    Returns
    -------
    xr.DataTree
        One group per stream present in any input file (union of groups),
        each concatenated over `time`, sorted, and deduplicated on
        timestamp (first occurrence kept). The L0 root data (per-block
        clock forensics) is not carried into the result. Variable-level
        attrs (`units`, `long_name`, ...) and each group's own dataset
        attrs (e.g. the SBE49 calibration coefficients `modraw.read()`
        stamps onto `ctd`) are taken from whichever file's group sorts
        first among the parts handed to `xr.concat` for that group.

        Root attrs: `files` (input basenames, in the order given), `n_files`.
        Root attrs also carry `groups`: the selection as a list, or the
        string ``"all"`` when none was given.
        Each group dataset's own attrs carry `n_bad_length`, summed across
        the files that contributed to it, where at least one of them
        carried that attr (this overwrites whichever single file's count
        `xr.concat` otherwise would have carried through).
        The `ctd` attrs also carry `n_misstamped` and `n_stale_after_gap`,
        the records dropped by the two acquisition-restart checks below.

    Raises
    ------
    ValueError
        If `files` is empty, or if no `ctd` group survives concatenation.

    Notes
    -----
    `logger.warning` fires once per consecutive file pair whose time ranges
    overlap by more than 5 s or gap by more than 60 s, based on the union of
    each file's group time spans (not `ctd` alone, so a file missing `ctd`
    still participates in the check).

    Two acquisition-restart defects are removed from `ctd`
    (modscripps/modfish#41). Per file, records stamped later than a record
    that follows them in the file are dropped (`_drop_misstamped`). After
    concatenation, a first sample after a gap longer than 0.5 s is dropped
    when its pressure departs from the extrapolation of the next two samples
    by more than 5 dbar (`_drop_stale_after_gap`). Both counts are logged.

    Each input file is opened, loaded into memory, and closed one at a time
    (see `_load_groups`), so at most one netCDF file handle is open at a
    time regardless of how many files are given.
    """
    files = [Path(f) for f in files]
    if not files:
        raise ValueError("concat_l0: no files given")

    groups = None if groups is None else list(groups)
    per_file_groups = [_load_groups(f, groups) for f in files]

    group_names = []
    for file_groups in per_file_groups:
        for name in file_groups:
            if name not in group_names:
                group_names.append(name)

    ranges = [_time_range(groups) for groups in per_file_groups]
    for i in range(len(ranges) - 1):
        r0, r1 = ranges[i], ranges[i + 1]
        if r0 is None or r1 is None:
            continue
        delta_s = (r1[0] - r0[1]) / np.timedelta64(1, "s")
        if delta_s < -_OVERLAP_WARN_S:
            logger.warning(
                "%s and %s overlap by %.1f s",
                files[i].name,
                files[i + 1].name,
                -delta_s,
            )
        elif delta_s > _GAP_WARN_S:
            logger.warning(
                "gap of %.1f s between %s and %s",
                delta_s,
                files[i].name,
                files[i + 1].name,
            )

    n_misstamped = 0
    for file_groups, path in zip(per_file_groups, files):
        if "ctd" in file_groups:
            file_groups["ctd"], n = _drop_misstamped(file_groups["ctd"])
            if n:
                logger.warning("%s: dropped %d misstamped ctd records", path.name, n)
            n_misstamped += n

    result_groups = {}
    for name in group_names:
        parts = [groups[name] for groups in per_file_groups if name in groups]
        if not parts:
            continue

        ds = xr.concat(parts, dim="time", combine_attrs="override")
        ds = ds.sortby("time")
        dup = pd.Index(ds["time"].values).duplicated()
        if dup.any():
            ds = ds.sel(time=~dup)

        if name == "ctd":
            ds, n_stale = _drop_stale_after_gap(ds)
            if n_stale:
                logger.warning("dropped %d stale ctd samples after gaps", n_stale)
            ds.attrs["n_misstamped"] = n_misstamped
            ds.attrs["n_stale_after_gap"] = n_stale

        if not keep_counts and name in ("ctd", "ecop"):
            drop = [v for v in ds.data_vars if str(v).endswith("_raw")]
            ds = ds.drop_vars(drop)

        bad_length_parts = [p.attrs["n_bad_length"] for p in parts if "n_bad_length" in p.attrs]
        if bad_length_parts:
            ds.attrs["n_bad_length"] = int(sum(bad_length_parts))

        result_groups[name] = ds

    if "ctd" not in result_groups:
        raise ValueError("concat_l0: no ctd group survived concatenation")

    tree = xr.DataTree.from_dict({f"/{k}": v for k, v in result_groups.items()})
    tree.attrs = dict(
        files=[f.name for f in files],
        n_files=len(files),
        groups="all" if groups is None else list(groups),
    )
    return tree
