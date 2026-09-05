"""Load the `efe/c1` stream of a deployment as one float32 column plus a
range table.

The EFE record timestamps are whole milliseconds while the ADC runs at
3.072 ms (325.52 Hz on every file of both cruises), so the per-sample
times are reconstructed per range as `start + i / fs` with `fs` from the
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

    One pass per file: the timestamps are used for the overlap test, the
    interior gap search and the range bookkeeping, then dropped. Only the
    `c1` pieces are held, so peak memory is the float32 column plus one
    file's timestamps, not the int64 timestamp column and its copies.

    Files are chronological, so an acquisition-rollover overlap sits at
    the file start: leading samples not strictly later than the last kept
    timestamp are dropped, and the same running-maximum rule drops any
    non-increasing sample inside a file. A range whose last sample is
    within `gap` of the next file's first sample continues across the file
    boundary and keeps one `fs`.

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
    gap_ns = gap * 1e9
    pieces, spans = [], []
    total, prev_max = 0, None
    for f in files:
        with xr.open_datatree(Path(f)) as tree:
            if "efe" not in tree.children:
                continue
            ds = tree["efe"].to_dataset()
            if "c1" not in ds or ds.sizes.get("time", 0) == 0:
                continue
            t = ds["time"].values.astype("datetime64[ns]").astype("int64")
            v = np.asarray(ds["c1"].values, dtype=np.float32)
        acc = np.maximum.accumulate(t)
        keep = np.ones(t.size, dtype=bool)
        keep[1:] = t[1:] > acc[:-1]
        if prev_max is not None:
            keep &= t > prev_max
        prev_max = int(acc[-1]) if prev_max is None else max(prev_max, int(acc[-1]))
        t, v = t[keep], v[keep]
        if t.size == 0:
            continue
        cuts = np.flatnonzero(np.diff(t) > gap_ns) + 1
        seg0 = np.concatenate([[0], cuts])
        seg1 = np.concatenate([cuts, [t.size]])
        for a, b in zip(seg0, seg1):
            n = int(b - a)
            if spans and a == 0 and t[0] - spans[-1]["last"] <= gap_ns:
                spans[-1]["n"] += n
                spans[-1]["last"] = int(t[b - 1])
            else:
                spans.append(dict(i0=total + int(a), n=n,
                                  first=int(t[a]), last=int(t[b - 1])))
        pieces.append(v)
        total += t.size
    if not pieces:
        raise ValueError("no efe/c1 data in the given files")
    c1 = np.concatenate(pieces) if len(pieces) > 1 else pieces[0]
    rows = []
    for s in spans:
        span = (s["last"] - s["first"]) / 1e9
        fs = (s["n"] - 1) / span if s["n"] > 1 and span > 0 else np.nan
        rows.append(dict(i0=s["i0"], n=s["n"],
                         start=np.int64(s["first"]).astype("datetime64[ns]"), fs=fs))
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
