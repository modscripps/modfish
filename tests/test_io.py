#!/usr/bin/env python

"""Tests for `modfish` package."""

import pathlib

import numpy as np
import pytest

import modfish


# We defined rootdir as a fixture in conftest.py
# and can use it here as input now
def test_load_fctd_grid(rootdir):
    test_grid_file = rootdir / "data/FCTDgrid.mat"
    assert type(test_grid_file) == pathlib.PosixPath
    assert test_grid_file.exists()
    ds = modfish.io.load_fctd_grid(test_grid_file)
    t0 = ds.time[0]
    assert t0 == np.datetime64("2024-11-08T01:09:11.569746804")


def test_load_epsi_grid(rootdir):
    test_grid_file = rootdir / "data/Epsigrid.mat"
    assert type(test_grid_file) == pathlib.PosixPath
    assert test_grid_file.exists()
    ds = modfish.io.load_epsi_grid(test_grid_file)
    t0 = ds.time[0]
    assert t0 == np.datetime64("2024-11-09T04:48:59.572897554")


def test_load_fctd_grid_adds_n2(rootdir):
    ds = modfish.io.load_fctd_grid(rootdir / "data/FCTDgrid.mat")
    assert ds.n2.dims == ds.t.dims
    n2 = ds.n2.values[np.isfinite(ds.n2.values)]
    assert n2.size > 0.5 * np.isfinite(ds.t.values).sum()
    assert np.median(n2) > 0
    assert ds.n2.attrs["units"] == "s$^{-2}$"


def test_load_epsi_grid_adds_n2_with_depth_only_pressure(rootdir):
    ds = modfish.io.load_epsi_grid(rootdir / "data/Epsigrid.mat")
    assert ds.p.dims == ("depth",)
    assert ds.n2.dims == ds.t.dims
    n2 = ds.n2.values[np.isfinite(ds.n2.values)]
    assert n2.size > 0
    assert np.median(n2) > 0


RAW_MAT_2025 = pathlib.Path(
    "/mnt/mod-server/MOTIVE/Cruises/skq202521s/05_processed_data/"
    "25_1205_d07_FCTD1_FrontStation/fctd_mat/FCTD25_12_05_082141.mat"
)


@pytest.mark.slow
@pytest.mark.skipif(not RAW_MAT_2025.exists(), reason="mod-server mount not available")
def test_load_fctd_raw_mat_conductivity_is_labelled_s_per_m():
    import gsw

    ds, _ = modfish.io.load_fctd_raw_mat(RAW_MAT_2025)
    assert ds.c.attrs["units"] == "S/m"
    ok = np.isfinite(ds.c) & np.isfinite(ds.t) & np.isfinite(ds.p)
    # gsw takes mS/cm; the values only make sense as S/m (SP near 34.5,
    # against 2.9 if they were mS/cm).
    sp = gsw.SP_from_C(ds.c[ok].values * 10, ds.t[ok].values, ds.p[ok].values)
    assert 33 < np.median(sp) < 36


def _write_fctd_mat(directory, start, minutes=10.0, fs=16.0, with_fast_time=True):
    """One synthetic `fctd_mat` file starting at `start` (datetime64)."""
    import scipy.io

    n = int(minutes * 60 * fs)
    t0 = (start - np.datetime64("1970-01-01")) / np.timedelta64(1, "D") + 719529
    mattime = t0 + np.arange(n) / fs / 86400
    d = dict(time=mattime, conductivity=np.full(n, 4.0), temperature=np.full(n, 20.0),
             pressure=np.linspace(0, 100, n), bb=np.zeros(n), chla=np.zeros(n),
             fDOM=np.zeros(n), longitude=np.full(n, -139.5), latitude=np.full(n, 1.0),
             uConductivity=np.zeros((n, 20)))
    if with_fast_time:
        d["time_fast"] = (t0 + np.arange(n * 20) / (fs * 20) / 86400).reshape(n, 20)
    name = pd_name(start)
    scipy.io.savemat(directory / name, d)
    return directory / name


def pd_name(start):
    import pandas as pd

    return pd.Timestamp(start).strftime("FCTD%y_%m_%d_%H%M%S.mat")


def test_load_fctd_raw_time_series_covers_exactly_the_requested_span(tmp_path):
    """Files are named by their start time. `file_times > start` dropped the
    file holding `start`, and nothing trimmed to `[start, end]`."""
    t0 = np.datetime64("2025-12-08T12:00:00")
    for k in range(3):
        _write_fctd_mat(tmp_path, t0 + np.timedelta64(10 * k, "m"))
    (tmp_path / "FCTDgrid.mat").write_bytes(b"")  # never matched
    start, end = t0 + np.timedelta64(5, "m"), t0 + np.timedelta64(15, "m")
    ds, mds = modfish.io.load_fctd_raw_time_series(tmp_path, start, end)
    for d in (ds, mds):
        assert d.time.values[0] >= start and d.time.values[-1] <= end
        assert d.time.values[0] - start < np.timedelta64(1, "s")
        assert end - d.time.values[-1] < np.timedelta64(1, "s")


def test_load_fctd_raw_mat_builds_fast_time_when_missing(tmp_path):
    f = _write_fctd_mat(tmp_path, np.datetime64("2025-12-08T12:00:00"), minutes=1.0,
                        with_fast_time=False)
    ds, mds = modfish.io.load_fctd_raw_mat(f)
    assert np.all(np.diff(mds.time.values) > np.timedelta64(0, "ns"))
    assert mds.time.values[0] == ds.time.values[0] and mds.time.values[-1] == ds.time.values[-1]


def test_load_fctd_grid_rejects_unknown_selection(rootdir):
    with pytest.raises(ValueError, match="what"):
        modfish.io.load_fctd_grid(rootdir / "data/FCTDgrid.mat", what="upcast")


def test_grid_loaders_metadata(rootdir):
    for ds in (modfish.io.load_fctd_grid(rootdir / "data/FCTDgrid.mat"),
               modfish.io.load_epsi_grid(rootdir / "data/Epsigrid.mat")):
        assert ds.SA.attrs["units"] == "g/kg"
        assert ds.sgth.attrs["long_name"] == r"$\sigma_0$"
    assert modfish.io.load_epsi_grid(rootdir / "data/Epsigrid.mat").th.attrs["long_name"] == r"$\theta$"


def test_distance_survives_missing_positions():
    import xarray as xr

    lon = np.array([-139.0, -139.01, np.nan, -139.03, -139.04])
    ds = xr.Dataset(coords=dict(time=np.arange(5), lon=("time", lon), lat=("time", np.full(5, 1.0))))
    dist = modfish.io._add_distance(ds).dist.values
    assert np.all(np.isfinite(dist)) and np.all(np.diff(dist) > 0)
    assert dist[-1] == pytest.approx(4 * 1.112, rel=0.01)  # 0.01 deg of longitude at 1N
