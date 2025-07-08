#!/usr/bin/env python

"""Tests for `modfish` package."""

import pathlib
import numpy as np

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
