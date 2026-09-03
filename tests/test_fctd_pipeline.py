"""Tests for the `process_deployment` driver: concat -> l1 -> grid -> write."""

import pytest
import xarray as xr

from modfish.fctd import process_deployment
from modfish.fctd.config import FCTDConfig, TCParams
from synth_l0 import two_cast_p, write_l0_files


def test_process_deployment_writes_both_products(tmp_path):
    files = write_l0_files(tmp_path / "l0", n_files=3, minutes=12.0, p_fn=two_cast_p)
    cfg = FCTDConfig(tc=TCParams())
    l1_path, grid_path = process_deployment(files, tmp_path / "out", "d99_test", cfg)
    assert l1_path.name == "fctd_d99_test_l1.nc"
    tree = xr.open_datatree(l1_path)
    assert "casts" in tree.children
    grid = xr.open_dataset(grid_path)
    assert grid.t.dims == ("depth", "cast")


def test_process_deployment_no_casts_raises_naming_deployment(tmp_path):
    files = write_l0_files(tmp_path / "l0", n_files=3, minutes=12.0)  # default flat p_fn: no casts
    cfg = FCTDConfig(tc=TCParams())
    with pytest.raises(ValueError, match="d99_nocast"):
        process_deployment(files, tmp_path / "out", "d99_nocast", cfg)


def test_process_deployment_groups_drops_efe_and_keeps_ctd_identical(tmp_path):
    files = write_l0_files(tmp_path / "l0", n_files=3, minutes=12.0, p_fn=two_cast_p)
    full_l1, full_grid = process_deployment(
        files, tmp_path / "full", "d99_full", FCTDConfig(tc=TCParams())
    )
    sel_l1, sel_grid = process_deployment(
        files, tmp_path / "sel", "d99_sel", FCTDConfig(tc=TCParams(), groups=("ctd", "gps"))
    )
    full = xr.open_datatree(full_l1)
    sel = xr.open_datatree(sel_l1)
    assert "efe" in full.children
    assert "efe" not in sel.children
    # assert_equal ignores attrs: the root `groups` attr is copied onto the
    # groups and differs by design. Check the attrs that must agree explicitly.
    xr.testing.assert_equal(sel["ctd"].to_dataset(), full["ctd"].to_dataset())
    xr.testing.assert_equal(xr.open_dataset(sel_grid), xr.open_dataset(full_grid))
    assert sel["ctd"].attrs["corrections"] == full["ctd"].attrs["corrections"]
    assert sel["ctd"]["t"].attrs == full["ctd"]["t"].attrs
    assert list(sel.attrs["groups"]) == ["ctd", "gps"]
