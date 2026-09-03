import numpy as np
import pytest

from modfish.fctd.concat import concat_l0
from modfish.fctd.config import FCTDConfig, GridParams, TCParams
from modfish.fctd.grid import grid_casts
from modfish.fctd.l1 import make_l1
from synth_l0 import two_cast_p, write_l0_files


@pytest.fixture()
def l1_tree(tmp_path):
    files = write_l0_files(tmp_path, n_files=3, minutes=12.0, p_fn=two_cast_p)
    return make_l1(concat_l0(files), FCTDConfig(tc=TCParams()))


def test_grid_dims_and_coords(l1_tree):
    g = grid_casts(l1_tree)
    assert g.t.dims == ("depth", "cast")
    assert {"direction", "time", "lon", "lat"} <= set(g.coords)
    assert np.all(np.diff(g.depth) == pytest.approx(0.5))


def test_grid_downcast_mean_temperature_sane(l1_tree):
    g = grid_casts(l1_tree)
    down = g.sel(cast=g.direction == "down")
    assert down.sizes["cast"] >= 1
    prof = down.t.isel(cast=0)
    assert np.isfinite(prof.sel(depth=slice(20, 250))).mean() > 0.9


def test_grid_no_zeros_where_empty(l1_tree):
    g = grid_casts(l1_tree, GridParams(depth_min=0.0, depth_max=500.0))
    deep = g.t.sel(depth=slice(320, 500))
    assert np.isnan(deep).all()          # no samples below ~300 dbar


def test_grid_out_of_range_rejected_not_clamped(l1_tree):
    # synthetic t = 25 - 0.02 * p; if samples below 100 dbar were clamped
    # into the deepest bin (Matlab bug B3), that bin would average samples
    # down to 300 dbar and read roughly 21 rather than 23.
    g = grid_casts(l1_tree, GridParams(depth_min=0.0, depth_max=100.0))
    edge = g.t.sel(depth=100.0, method="nearest")
    expected = 25 - 0.02 * 100.0
    assert np.nanmax(np.abs(edge - expected)) < 0.5
