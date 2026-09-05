"""Load the `efe/c1` stream of a deployment as one float32 column plus a
range table.

The EFE record timestamps are whole milliseconds while the ADC runs at
3.072 ms (325.52 Hz on every file of both cruises), so the per-sample
times are reconstructed per range as `start + n / fs` with `fs` from the
sample count over the range span. The timestamps serve only the range
starts and the gap test.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

RANGE_COLUMNS = ["i0", "n", "start", "fs"]


def load_c1(files, gap=0.01):
    """Read `efe/c1` of every file and split the record into gap-free ranges.

    Parameters
    ----------
    files : sequence of Path or str
        Per-file L0 netCDF paths in chronological order.
    gap : float, optional
        s, a timestamp step above this opens a new range. Default 0.01.

    Returns
    -------
    c1 : numpy.ndarray
        float32 volts, strictly increasing timestamps, overlaps dropped.
    ranges : pandas.DataFrame
        Columns `i0`, `n`, `start` (datetime64[ns]), `fs` (Hz, NaN when
        `n < 2`).

    Raises
    ------
    ValueError
        When no file carries a non-empty `efe/c1`.
    FileNotFoundError
        When a file in `files` does not exist.
    """
    times, values = [], []
    for f in files:
        with xr.open_datatree(Path(f)) as tree:
            if "efe" not in tree.children:
                continue
            ds = tree["efe"].to_dataset()
            if "c1" not in ds or ds.sizes.get("time", 0) == 0:
                continue
            times.append(ds["time"].values.astype("datetime64[ns]").astype("int64"))
            values.append(ds["c1"].values.astype(np.float32))
    if not times:
        raise ValueError("no efe/c1 data in the given files")
    t = np.concatenate(times)
    c1 = np.concatenate(values)

    keep = np.ones(t.size, dtype=bool)
    keep[1:] = t[1:] > np.maximum.accumulate(t)[:-1]
    t, c1 = t[keep], c1[keep]

    step = np.diff(t)
    starts = np.concatenate([[0], np.flatnonzero(step > gap * 1e9) + 1])
    ends = np.concatenate([starts[1:], [t.size]])
    rows = []
    for i0, i1 in zip(starts, ends):
        n = int(i1 - i0)
        span = (t[i1 - 1] - t[i0]) / 1e9
        fs = (n - 1) / span if n > 1 and span > 0 else np.nan
        rows.append(dict(i0=int(i0), n=n, start=t[i0].astype("datetime64[ns]"), fs=fs))
    ranges = pd.DataFrame(rows, columns=RANGE_COLUMNS)
    return c1, ranges


def range_time(start, n, fs):
    """Uniform timestamps of one range, datetime64[ns].

    Parameters
    ----------
    start : datetime64[ns]
        Start time of the range.
    n : int
        Number of samples in the range.
    fs : float
        Sampling frequency in Hz. Must be finite and positive when n > 1.

    Returns
    -------
    numpy.ndarray
        datetime64[ns] array of uniform timestamps.

    Raises
    ------
    ValueError
        When n > 1 and fs is not finite or not positive.
    """
    n = int(n)
    if n <= 1:
        return np.array([np.datetime64(start, "ns")])
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError(
            f"fs must be finite and positive when n > 1, got fs={fs}"
        )
    start_ns = np.datetime64(start, "ns").astype("int64")
    offsets = np.round(np.arange(n) / float(fs) * 1e9).astype("int64")
    return (start_ns + offsets).astype("datetime64[ns]")
