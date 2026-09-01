"""Synthetic L0 DataTree builder, for tests exercising pipeline stages that
consume `modfish.modraw.read()`'s output without needing real `.modraw`
files.

Reproduces the group layout `read()` produces: `ctd` (16 Hz by default),
`efe` (320 Hz), `gps` (0.5 Hz), and a root dataset carrying block-forensics
data on dim `block` (unused by downstream stages, present here only so
tests that touch the root do not need a second fixture shape). `ecop` is
not built here; pipeline stages must not assume it is present.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

#: EFE channel names, matching `modfish.modraw.efe.decode_efe4`'s output.
_EFE_CHANNELS = ("t1", "t2", "f1", "c1", "a1", "a2", "a3")


def two_cast_p(seconds):
    """Two-cast sawtooth pressure profile.

    60 s at 5 dbar, then a down/up cast pair spanning 300 dbar at
    1 dbar/s (300 s down, 300 s up), repeating.

    Parameters
    ----------
    seconds : array_like
        Seconds since the start of the profile.

    Returns
    -------
    np.ndarray
        Pressure, dbar, same shape as `seconds`.
    """
    seconds = np.asarray(seconds, dtype=float)
    period = 60.0 + 300.0 + 300.0
    phase = np.mod(seconds, period)

    p = np.full_like(phase, 5.0)
    down = (phase >= 60.0) & (phase < 360.0)
    p = np.where(down, 5.0 + (phase - 60.0), p)
    up = phase >= 360.0
    p = np.where(up, 5.0 + 300.0 - (phase - 360.0), p)
    return p


def make_l0_tree(
    t0: str,
    minutes: float,
    fs: float = 16.0,
    p_fn=None,
    with_efe: bool = True,
    with_gps: bool = True,
    seed: int = 0,
) -> xr.DataTree:
    """Build one synthetic L0 DataTree, matching `modfish.modraw.read()`'s
    group layout.

    Parameters
    ----------
    t0 : str or Timestamp
        Start time of this file's `ctd` stream (and of `efe`/`gps`, which
        share the same start).
    minutes : float
        Duration of the file, minutes.
    fs : float, optional
        `ctd` sampling rate, Hz. Default 16.0.
    p_fn : callable or None
        `p_fn(seconds) -> pressure`, `seconds` counted from `t0`. Default:
        constant 5 dbar.
    with_efe : bool, optional
        Include the `efe` group (320 Hz, channels `t1 t2 f1 c1 a1 a2 a3`,
        volts, generated as noise). Default True.
    with_gps : bool, optional
        Include the `gps` group (0.5 Hz fix stream at (2.0 N, 140.0 W),
        drifting 0.001 deg/min in both lat and lon). Default True.
    seed : int, optional
        Seed for the per-file random generator. Default 0.

    Returns
    -------
    xr.DataTree
        Root (dim `block`, block-forensics placeholder) plus groups `ctd`,
        and optionally `efe` and `gps`.
    """
    rng = np.random.default_rng(seed)
    t0 = pd.Timestamp(t0)

    if p_fn is None:
        def p_fn(seconds):
            return np.full_like(np.asarray(seconds, dtype=float), 5.0)

    n = int(round(minutes * 60 * fs))
    seconds = np.arange(n) / fs
    time = (t0 + pd.to_timedelta(seconds, unit="s")).values

    p = np.asarray(p_fn(seconds), dtype=float) + rng.normal(0, 0.05, n)
    t = 25.0 - 0.02 * p + rng.normal(0, 0.01, n)
    c = 3.5 - 0.001 * p + rng.normal(0, 0.001, n)

    ctd = xr.Dataset(
        coords={"time": ("time", time)},
        data_vars={
            "t": ("time", t),
            "c": ("time", c),
            "p": ("time", p),
            "t_raw": ("time", rng.integers(0, 2**16, n).astype(float)),
            "c_raw": ("time", rng.integers(0, 2**16, n).astype(float)),
            "p_raw": ("time", rng.integers(0, 2**16, n).astype(float)),
            "pt_raw": ("time", rng.integers(0, 2**16, n).astype(float)),
        },
    )
    ctd.attrs["n_bad_length"] = 0

    groups = {"ctd": ctd}

    if with_efe:
        fs_efe = 320.0
        n_efe = int(round(minutes * 60 * fs_efe))
        seconds_efe = np.arange(n_efe) / fs_efe
        time_efe = (t0 + pd.to_timedelta(seconds_efe, unit="s")).values
        efe = xr.Dataset(
            coords={"time": ("time", time_efe)},
            data_vars={
                ch: ("time", rng.normal(0, 1.0, n_efe)) for ch in _EFE_CHANNELS
            },
        )
        efe.attrs["n_bad_length"] = 0
        groups["efe"] = efe

    if with_gps:
        fs_gps = 0.5
        n_gps = max(int(round(minutes * 60 * fs_gps)), 1)
        seconds_gps = np.arange(n_gps) / fs_gps
        time_gps = (t0 + pd.to_timedelta(seconds_gps, unit="s")).values
        drift_per_s = 0.001 / 60.0  # deg/min -> deg/s
        gps = xr.Dataset(
            coords={"time": ("time", time_gps)},
            data_vars={
                "lat": ("time", 2.0 + drift_per_s * seconds_gps),
                "lon": ("time", -140.0 + drift_per_s * seconds_gps),
            },
        )
        groups["gps"] = gps

    root = xr.Dataset(
        data_vars={
            "block_tag": ("block", np.array(["SB49"] * n, dtype="<U8")),
        },
    )

    tree = xr.DataTree.from_dict({"/": root, **{f"/{k}": v for k, v in groups.items()}})
    return tree


def write_l0_files(
    outdir, n_files: int = 3, minutes: float = 10.0, p_fn=None, **kw
) -> list[Path]:
    """Write a sequence of synthetic per-file L0 netCDF files.

    Consecutive files share a 2-sample timestamp overlap at each boundary
    (on the `ctd` sampling grid), matching how the acquisition system rolls
    files over. `p_fn` is called with seconds since the start of the FIRST
    file, so the pressure record is continuous across file boundaries.

    Parameters
    ----------
    outdir : Path or str
        Output directory; created if missing.
    n_files : int, optional
        Number of files to write. Default 3.
    minutes : float, optional
        Duration of each file, minutes. Default 10.0.
    p_fn : callable or None
        `p_fn(seconds) -> pressure`, `seconds` counted from the start of
        the first file. Default: constant 5 dbar.
    **kw
        Passed through to `make_l0_tree` (`fs`, `with_efe`, `with_gps`,
        `seed`).

    Returns
    -------
    list of pathlib.Path
        Paths to the written `.nc` files, in chronological order.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    fs = kw.get("fs", 16.0)
    n_per_file = int(round(minutes * 60 * fs))
    step = n_per_file - 2  # 2-sample overlap at each boundary

    t0 = pd.Timestamp("2026-01-01T00:00:00")
    paths = []
    for i in range(n_files):
        k0 = i * step
        offset = k0 / fs
        file_t0 = t0 + pd.to_timedelta(offset, unit="s")

        if p_fn is not None:
            def file_p_fn(seconds, _offset=offset):
                return p_fn(np.asarray(seconds, dtype=float) + _offset)
        else:
            file_p_fn = None

        tree = make_l0_tree(file_t0, minutes, p_fn=file_p_fn, **kw)
        path = outdir / f"l0_{i:03d}.nc"
        tree.to_netcdf(path)
        paths.append(path)

    return paths
