"""Tests for the `process_deployment` driver: concat -> l1 -> grid -> write."""

import pytest
import xarray as xr

from modfish.fctd import process_deployment
from modfish.fctd.config import FCTDConfig, TCParams
from synth_l0 import two_cast_p, write_l0_files


def test_process_deployment_writes_both_products(tmp_path):
    files = write_l0_files(tmp_path / "l0", n_files=3, minutes=12.0, p_fn=two_cast_p)
    cfg = FCTDConfig(tc=TCParams(phase_match=False))
    l1_path, grid_path = process_deployment(files, tmp_path / "out", "d99_test", cfg)
    assert l1_path.name == "fctd_d99_test_l1.nc"
    tree = xr.open_datatree(l1_path)
    assert "casts" in tree.children
    grid = xr.open_dataset(grid_path)
    assert grid.t.dims == ("depth", "cast")


def test_process_deployment_no_casts_raises_naming_deployment(tmp_path):
    files = write_l0_files(tmp_path / "l0", n_files=3, minutes=12.0)  # default flat p_fn: no casts
    cfg = FCTDConfig(tc=TCParams(phase_match=False))
    with pytest.raises(ValueError, match="d99_nocast"):
        process_deployment(files, tmp_path / "out", "d99_nocast", cfg)
