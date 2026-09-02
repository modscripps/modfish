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


def _load_groups(path: Path) -> dict:
    """Open one file's L0 DataTree, load its groups into memory, and close it.

    Loading eagerly and closing immediately keeps at most one file handle
    open at a time; a deployment can span hundreds of files, and holding
    every `xr.open_datatree` handle open until the whole concatenation
    finishes risks exhausting file descriptors.

    Parameters
    ----------
    path : Path
        Per-file L0 netCDF path.

    Returns
    -------
    dict of str -> xr.Dataset
        In-memory group datasets, keyed by group name. The root (block
        forensics, dim `block`) is excluded: it is never a child of itself
        in `tree.children`.
    """
    with xr.open_datatree(path) as tree:
        return {name: node.ds.load() for name, node in tree.children.items()}


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


def concat_l0(files: list, keep_counts: bool = False) -> xr.DataTree:
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
        Each group dataset's own attrs carry `n_bad_length`, summed across
        the files that contributed to it, where at least one of them
        carried that attr (this overwrites whichever single file's count
        `xr.concat` otherwise would have carried through).

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

    Each input file is opened, loaded into memory, and closed one at a time
    (see `_load_groups`), so at most one netCDF file handle is open at a
    time regardless of how many files are given.
    """
    files = [Path(f) for f in files]
    if not files:
        raise ValueError("concat_l0: no files given")

    per_file_groups = [_load_groups(f) for f in files]

    group_names = []
    for groups in per_file_groups:
        for name in groups:
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
    )
    return tree
