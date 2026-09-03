"""Cast detection: split a pressure record into down/up casts.

Profiler-agnostic: operates on pressure and time arrays plus `CastParams`
only. No FCTD-specific variable names appear here so the same detection
serves L1 processing and gridding.
"""

import logging

import numpy as np
import pandas as pd
import xarray as xr
from scipy.ndimage import uniform_filter1d
from scipy.signal import medfilt

from modfish.fctd.config import CastParams
from modfish.utils import sampling_interval

logger = logging.getLogger(__name__)


def _odd_window(seconds: float, fs: float, n_max: int) -> int:
    """Convert a window length in seconds to an odd sample count.

    Parameters
    ----------
    seconds : float
        Window length in seconds.
    fs : float
        Sampling rate in Hz.
    n_max : int
        Number of samples in the record; the window is capped to this
        length (rounded down to odd) so short records do not error out.

    Returns
    -------
    int
        Odd window length in samples, at least 1.
    """
    n = round(seconds * fs)
    if n % 2 == 0:
        n += 1
    n = max(n, 1)
    if n > n_max:
        n = n_max if n_max % 2 == 1 else n_max - 1
        n = max(n, 1)
    return n


def _contiguous_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Inclusive index ranges of contiguous True runs in a boolean mask.

    Parameters
    ----------
    mask : np.ndarray
        Boolean array.

    Returns
    -------
    list of tuple of int
        (i0, i1) inclusive start/end index pairs, one per contiguous run.
    """
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return []
    breaks = np.flatnonzero(np.diff(idx) > 1) + 1
    groups = np.split(idx, breaks)
    return [(int(g[0]), int(g[-1])) for g in groups]


def find_casts(
    p: np.ndarray, time: np.ndarray, params: CastParams | None = None
) -> pd.DataFrame:
    """Detect down- and up-casts in a pressure record.

    Smooths pressure with a median filter followed by a centered moving
    average, differentiates to a physical rate of pressure change, and
    keeps contiguous runs that exceed the rate threshold and satisfy
    minimum range and duration criteria.

    Parameters
    ----------
    p : np.ndarray
        Pressure, dbar. Length N.
    time : np.ndarray
        Timestamps, dtype datetime64. Length N, matching `p`.
    params : CastParams or None
        Detection parameters. Defaults to `CastParams()`.

    Returns
    -------
    pd.DataFrame
        Columns:

        cast : int
            Chronological cast id, starting at 1.
        i0 : int
            Start index into `p`/`time` (inclusive).
        i1 : int
            End index into `p`/`time` (inclusive).
        direction : str
            "down" or "up".

        Empty (zero rows) when no cast satisfies the criteria.
    """
    if params is None:
        params = CastParams()

    p = np.asarray(p, dtype=float)
    n_samples = p.size

    fs = 1.0 / sampling_interval(time)
    n_window = _odd_window(params.smooth, fs, n_samples)

    p_med = medfilt(p, n_window)
    p_smooth = uniform_filter1d(p_med, n_window, mode="nearest")

    dpdt = np.gradient(p_smooth) * fs  # dbar/s, length N

    rows = []
    n_rejected = 0
    for mask, direction in ((dpdt > params.wlim, "down"), (dpdt < -params.wlim, "up")):
        for i0, i1 in _contiguous_runs(mask):
            if i1 <= i0:
                continue
            p_range = abs(p[i1] - p[i0])
            duration = (time[i1] - time[i0]) / np.timedelta64(1, "s")
            if p_range >= params.min_range and duration >= params.min_duration:
                rows.append((i0, i1, direction))
            else:
                n_rejected += 1

    if n_rejected > 0:
        logger.info(
            "find_casts: %d candidate run(s) rejected by min_range=%s/min_duration=%s",
            n_rejected,
            params.min_range,
            params.min_duration,
        )

    rows.sort(key=lambda r: r[0])
    casts = pd.DataFrame(rows, columns=["i0", "i1", "direction"])
    casts.insert(0, "cast", np.arange(1, len(casts) + 1))
    casts["i0"] = casts["i0"].astype(int)
    casts["i1"] = casts["i1"].astype(int)
    return casts


def casts_to_dataset(casts: pd.DataFrame, time: np.ndarray) -> xr.Dataset:
    """Summarize detected casts as a per-cast dataset.

    Parameters
    ----------
    casts : pd.DataFrame
        Output of `find_casts`.
    time : np.ndarray
        Timestamps, dtype datetime64, the same array `find_casts` ran on.

    Returns
    -------
    xr.Dataset
        Dim "cast" (coordinate: cast id). Data variables:

        start_time : datetime64
            `time[i0]` for each cast.
        end_time : datetime64
            `time[i1]` for each cast.
        direction : str
            "down" or "up".
    """
    time = np.asarray(time)
    if len(casts) == 0:
        cast_ids = np.array([], dtype=int)
        start_time = np.array([], dtype=time.dtype)
        end_time = np.array([], dtype=time.dtype)
        direction = np.array([], dtype=object)
    else:
        cast_ids = casts["cast"].to_numpy()
        start_time = time[casts["i0"].to_numpy()]
        end_time = time[casts["i1"].to_numpy()]
        direction = casts["direction"].to_numpy()

    return xr.Dataset(
        data_vars={
            "start_time": ("cast", start_time),
            "end_time": ("cast", end_time),
            "direction": ("cast", direction),
        },
        coords={"cast": ("cast", cast_ids)},
    )


def label_casts(
    ds: xr.Dataset, casts: pd.DataFrame, time_ref: np.ndarray | None = None
) -> xr.Dataset:
    """Attach a per-sample cast-id coordinate to a dataset's time dimension.

    Samples outside any detected cast are labelled 0.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset with a "time" dimension/coordinate to label.
    casts : pd.DataFrame
        Output of `find_casts`.
    time_ref : np.ndarray or None
        The time axis `find_casts` ran on (dtype datetime64), i.e. the
        axis that `casts.i0`/`casts.i1` index into. When `ds.time` is a
        different axis (e.g. EFE at 320 Hz), each cast interval's start
        and end timestamps are located in `ds.time` via `searchsorted`.
        None (default) means `ds.time` is itself the detection axis, so
        `i0`/`i1` index into it directly.

    Returns
    -------
    xr.Dataset
        `ds` with an added int coordinate "cast" on dim "time".
    """
    target_time = ds["time"].values
    label = np.zeros(target_time.shape[0], dtype=int)

    if time_ref is None:
        for row in casts.itertuples():
            label[row.i0 : row.i1 + 1] = row.cast
    else:
        time_ref = np.asarray(time_ref)
        for row in casts.itertuples():
            t0 = time_ref[row.i0]
            t1 = time_ref[row.i1]
            j0 = np.searchsorted(target_time, t0, side="left")
            j1 = np.searchsorted(target_time, t1, side="right") - 1
            if j1 >= j0:
                label[j0 : j1 + 1] = row.cast

    return ds.assign_coords(cast=("time", label))
