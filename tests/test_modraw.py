#!/usr/bin/env python

"""Tests for the `modfish.modraw` module."""

import numpy as np
import pandas as pd
import pytest

import modfish

# An excerpt from FCTD25_12_08_134303.modraw, MOTIVE 2025 (SKQ2025-21S): the
# full header plus the first 400 kB of blocks, cut at a line boundary. FCTD1,
# SBE49 SN 0537, 30 s of the descent through 913 to 1008 dbar.
EXCERPT = "data/FCTD_modraw_excerpt.modraw"


@pytest.fixture
def modraw_file(rootdir):
    return rootdir / EXCERPT


def test_read_header(modraw_file):
    head = modfish.modraw.read_header(modraw_file)
    assert head.startswith("header_file_size_inbytes")
    # The first line states the header length and must be self-consistent.
    assert len(head.encode("latin-1")) == int(head.split("\n")[0].split("=")[1])


def test_header_setup(modraw_file):
    setup = modfish.modraw.header_setup(modfish.modraw.read_header(modraw_file))
    assert setup["vehicle"] == "FCTD1"
    assert setup["fishflag"] == "FCTD"
    assert setup["serialnum"] == "0537"
    assert setup["survey"] == "25_1208_d08_motiveb"


def test_sbe49_cal(modraw_file):
    cal = modfish.modraw.sbe49_cal(modfish.modraw.read_header(modraw_file))
    assert cal["ta0"] == pytest.approx(7.906933e-4)
    assert cal["cg"] == pytest.approx(-1.005065)
    assert cal["pa1"] == pytest.approx(8.981493e-3)


def test_load_ctd(modraw_file):
    ds = modfish.modraw.load_ctd(modraw_file)
    assert set(["p", "t", "c"]).issubset(ds.data_vars)
    assert ds.p.attrs["units"] == "dbar"
    assert ds.c.attrs["units"] == "S/m"

    # Two records per block, and the telemetry in this file is clean.
    assert ds.sizes["time"] == 2 * ds.attrs["n_block"]
    assert ds.attrs["n_bad_length"] == 0
    assert ds.attrs["n_bad_checksum"] == 0

    # The excerpt starts where the file does.
    assert ds.time[0].values == np.datetime64("2025-12-08T13:43:03.813000000")

    # Sampled at 16 Hz.
    dt = np.diff(ds.time.values).astype("timedelta64[ns]").astype(float) / 1e9
    assert np.median(dt) == pytest.approx(1 / 16, rel=0.02)

    # Physically sensible deep water, monotonically descending.
    assert 900 < ds.p.min() < ds.p.max() < 1100
    assert 4 < ds.t.min() < ds.t.max() < 6
    assert 3.0 < ds.c.min() < ds.c.max() < 3.5
    assert ds.p.diff("time").min() > 0
    for name in ("p", "t", "c"):
        assert not np.isnan(ds[name]).any()


def test_load_ctd_against_matlab_values(modraw_file):
    """Spot-check the calibration polynomials.

    The reference values come from the Matlab product for this file,
    `25_1208_d08_motiveb/mat/FCTD25_12_08_134303.mat`, which
    `mod_som_read_epsi_files_v4.m` produced from the same bytes.
    """
    ds = modfish.modraw.load_ctd(modraw_file)
    first = ds.isel(time=0)
    assert first.time.values == np.datetime64("2025-12-08T13:43:03.813000000")
    assert first.p.item() == pytest.approx(913.1178772138934, abs=1e-3)
    assert first.t.item() == pytest.approx(4.9620429873937155, abs=1e-4)
    assert first.c.item() == pytest.approx(3.343829895173352, abs=1e-6)


def test_load_gps_time(modraw_file):
    time = modfish.modraw.load_gps_time(modraw_file)
    assert len(time) > 0
    assert isinstance(time, pd.DatetimeIndex)
    # GPS UTC agrees with the acquisition clock the file name is based on.
    assert time[0] == pd.Timestamp("2025-12-08 13:43:25")


def test_block_counts(modraw_file):
    counts = modfish.modraw.block_counts(modraw_file)
    ds = modfish.modraw.load_ctd(modraw_file)
    assert counts["SB49"] == ds.attrs["n_block"]
    assert counts["EFE4"] > 0
    # The VectorNav was not writing during this deployment.
    assert counts["VNAV"] == 0


def test_load_ctd_time_series(modraw_file):
    single = modfish.modraw.load_ctd(modraw_file)
    combined = modfish.modraw.load_ctd_time_series([modraw_file, modraw_file])
    assert combined.sizes["time"] == 2 * single.sizes["time"]
    assert combined.attrs["n_block"] == 2 * single.attrs["n_block"]
    assert combined.attrs["vehicle"] == "FCTD1"
    assert combined.time.to_index().is_monotonic_increasing
