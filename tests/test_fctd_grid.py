import numpy as np
import pytest
import xarray as xr

from modfish.chi import add_chi
from modfish.chi.config import FLAG_EMPTY, ChiParams
from modfish.fctd.concat import concat_l0
from modfish.fctd.config import FCTDConfig, GridParams, TCParams
from modfish.fctd.grid import _bin_geomean, _bin_or, grid_casts
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


def test_bin_geomean_and_or():
    depth = np.array([0.1, 0.2, 0.6, 0.7, 1.4])
    edges = np.array([0.0, 0.5, 1.0, 1.5])
    vals = np.array([1e-10, 1e-8, np.nan, 1e-9, 1e-9])
    gm = _bin_geomean(depth, vals, edges)
    assert gm[0] == pytest.approx(1e-9) and gm[1] == pytest.approx(1e-9) and gm[2] == pytest.approx(1e-9)
    flags = np.array([1, 2, 0, 4, 0], dtype=np.uint8)
    assert _bin_or(depth, flags, edges).tolist() == [3, 4, 0]


def test_grid_bins_chi_group(l1_tree, tmp_path):
    files = write_l0_files(tmp_path / "chi", n_files=3, minutes=12.0, p_fn=two_cast_p)
    l1 = add_chi(make_l1(concat_l0(files, groups=("ctd", "gps")), FCTDConfig()), files,
                 ChiParams(enabled=True, gain=50.0))
    g = grid_casts(l1)
    for name in ("chi", "chi_tot", "eps_chi", "r", "kmax", "chi_flag"):
        assert g[name].dims == ("depth", "cast")
    assert g.chi_flag.dtype == np.uint8
    assert g["chi"].attrs["units"] == "K^2/s"
    assert np.isfinite(g.kmax.values).any()
    plain = grid_casts(l1_tree)
    assert "chi" not in plain


def test_grid_chi_flag_masks_kmax_mean(l1_tree):
    # Controller ruling: FLAG_EMPTY (and FLAG_SLOW) windows are excluded
    # from the chi group's bin means, but chi_flag's bitwise-or over the
    # bin still carries their bits. Build a tiny synthetic /chi group
    # (skipping the real add_chi pipeline, which is expensive) with two
    # windows in the same depth bin of the same cast: one flagged
    # FLAG_EMPTY with a finite kmax, one clean. If the flagged window
    # were not masked out, the bin mean would be their average (75)
    # rather than the clean window's value alone (50).
    ctd = l1_tree["ctd"].to_dataset()
    casts_ds = l1_tree["casts"].to_dataset()
    cid = int(casts_ds["cast"].values[0])
    depth_val = float(np.nanmean(ctd["depth"].values))

    chi_ds = xr.Dataset(
        {
            "depth": ("time", np.array([depth_val, depth_val])),
            "cast": ("time", np.array([cid, cid], dtype=int)),
            "kmax": ("time", np.array([100.0, 50.0])),
            "chi_flag": ("time", np.array([FLAG_EMPTY, 0], dtype=np.uint8)),
        },
        coords={"time": np.array(["2020-01-01T00:00:00", "2020-01-01T00:00:01"], dtype="datetime64[ns]")},
    )
    chi_ds["kmax"].attrs = dict(long_name="upper integration limit", units="cpm")
    chi_ds.attrs = dict(gain=50.0, gain_source="test", antialias="som_sinc4", modfish_version="0.0.0")

    groups = {f"/{name}": child.to_dataset() for name, child in l1_tree.children.items()}
    groups["/chi"] = chi_ds
    tree = xr.DataTree.from_dict(groups)
    tree.attrs = dict(l1_tree.attrs)

    g = grid_casts(tree)
    j = list(casts_ds["cast"].values).index(cid)
    bin_idx = int(np.nanargmin(np.abs(g["depth"].values - depth_val)))
    assert g["kmax"].values[bin_idx, j] == pytest.approx(50.0)
    assert int(g["chi_flag"].values[bin_idx, j]) & FLAG_EMPTY
