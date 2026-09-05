import numpy as np
import pandas as pd
import pytest
import xarray as xr

from modfish.chi.load import load_c1, range_time

DT_MS = 3.072


def _write_efe(path, t0_ms, n, values, extra_ms=0.0):
    """One L0-like file with an `efe` group whose timestamps are whole ms."""
    t = np.round(t0_ms + np.arange(n) * DT_MS + extra_ms).astype("int64")
    time = t.astype("datetime64[ms]").astype("datetime64[ns]")
    efe = xr.Dataset({"c1": ("time", values), "t1": ("time", np.zeros(n))},
                     coords={"time": time})
    ctd = xr.Dataset({"p": ("time", np.zeros(2))},
                     coords={"time": time[:2]})
    xr.DataTree.from_dict({"/ctd": ctd, "/efe": efe}).to_netcdf(path)
    return t


@pytest.fixture()
def files(tmp_path):
    rng = np.random.default_rng(0)
    n = 2000
    t0 = 1_700_000_000_000
    a = tmp_path / "a.nc"
    b = tmp_path / "b.nc"
    c = tmp_path / "c.nc"
    ta = _write_efe(a, t0, n, rng.normal(1.5, 0.01, n))
    # b overlaps a by 3 samples (acquisition rollover), then continues
    _write_efe(b, ta[-3], n, rng.normal(1.5, 0.01, n))
    # c starts 5 s after b ends: a new range
    _write_efe(c, ta[-3] + (n - 1) * DT_MS + 5000, n, rng.normal(1.5, 0.01, n))
    return [a, b, c]


def test_two_ranges_at_the_true_rate(files):
    c1, ranges = load_c1(files, gap=0.01)
    assert c1.dtype == np.float32
    assert list(ranges.columns) == ["i0", "n", "start", "fs"]
    assert len(ranges) == 2
    assert ranges.n.sum() == c1.size
    assert ranges.i0.tolist() == [0, int(ranges.n.iloc[0])]
    assert ranges.fs.values == pytest.approx(1000 / DT_MS, rel=2e-3)


def test_overlap_samples_dropped(files):
    c1, ranges = load_c1(files)
    # a (2000) + b (2000, 3 overlapping) + c (2000)
    assert c1.size == 2000 + 2000 - 3 + 2000


def test_range_time_is_uniform(files):
    _, ranges = load_c1(files)
    r = ranges.iloc[0]
    t = range_time(r.start, r.n, r.fs)
    assert t.dtype == np.dtype("datetime64[ns]")
    steps = np.diff(t).astype("int64")
    assert steps.min() == steps.max() or abs(steps.min() - steps.max()) <= 1
    assert t[0] == np.datetime64(r.start, "ns")


def test_files_without_efe_are_skipped(tmp_path, files):
    ctd = xr.Dataset({"p": ("time", np.zeros(2))},
                     coords={"time": np.array([0, 1], dtype="datetime64[ns]")})
    empty = tmp_path / "empty.nc"
    xr.DataTree.from_dict({"/ctd": ctd}).to_netcdf(empty)
    c1, ranges = load_c1([empty] + files)
    assert len(ranges) == 2


def test_no_data_raises(tmp_path):
    with pytest.raises(ValueError, match="no efe/c1"):
        load_c1([])
