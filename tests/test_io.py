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
