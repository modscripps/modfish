import numpy as np
import pytest
import xarray as xr

from modfish.fctd.concat import concat_l0
from modfish.fctd.config import FCTDConfig, TCParams
from modfish.fctd.l1 import make_l1
from synth_l0 import two_cast_p, write_l0_files


@pytest.fixture()
def l0_tree(tmp_path):
    files = write_l0_files(tmp_path, n_files=3, minutes=12.0, p_fn=two_cast_p)
    return concat_l0(files)


def test_l1_has_groups_and_derived(l0_tree):
    cfg = FCTDConfig(tc=TCParams())
    l1 = make_l1(l0_tree, cfg)
    assert {"ctd", "casts", "efe"} <= set(l1.children)
    for v in ["t", "c", "t_raw", "c_raw", "SP", "SA", "CT", "sgth0", "depth", "cast"]:
        assert v in l1["ctd"]


def test_l1_corrections_off_t_equals_raw(l0_tree):
    l1 = make_l1(l0_tree, FCTDConfig())
    np.testing.assert_array_equal(l1["ctd"].t.data, l1["ctd"].t_raw.data)
    assert l1["ctd"].attrs["corrections"] == "none"


def test_l1_chain_stamps_processing_and_keeps_axis(l0_tree):
    cfg = FCTDConfig(tc=TCParams(lag=0.1, tau_t=0.05, lowpass=4.0, thermal_mass=True))
    tree = make_l1(l0_tree, cfg)
    ctd = tree["ctd"].to_dataset()
    # ctd.time carries the "cast" coord label_casts attaches at L1 (l0_tree
    # has no such coord yet), so .equals() on the raw time DataArrays would
    # fail on that unrelated coordinate rather than on axis identity; compare
    # values directly to test what this asserts: the correction chain did
    # not reindex.
    np.testing.assert_array_equal(
        ctd.time.data, l0_tree["ctd"].to_dataset().time.data
    )
    assert "response lag 0.100 s tau 0.050 s" in ctd.t.attrs["processing"]
    assert "thermal mass" in ctd.c.attrs["processing"]
    assert "tau1" not in ctd.attrs
    assert np.isfinite(ctd.SP.data[:-2]).all()
    # t/c attrs (units, long_name) must survive the correction chain, not
    # just processing/corrections: a future _apply_tc that overwrote the
    # pre-correction attrs wholesale, instead of merging processing on top
    # of them, would pass every assertion above and still fail here.
    assert ctd["t"].attrs.get("units") == ctd["t_raw"].attrs.get("units")
    assert ctd["c"].attrs.get("units") == ctd["c_raw"].attrs.get("units")
    assert ctd["t"].attrs.get("long_name") == ctd["t_raw"].attrs.get("long_name")
    assert ctd["c"].attrs.get("long_name") == ctd["c_raw"].attrs.get("long_name")


def test_l1_default_config_is_noop(l0_tree):
    tree = make_l1(l0_tree)
    ctd = tree["ctd"].to_dataset()
    np.testing.assert_array_equal(ctd.t.data, ctd.t_raw.data)
    assert ctd.attrs["corrections"] == "none"


def test_l1_gps_gap_masks_interior_positions(l0_tree):
    time = l0_tree["ctd"].time.values
    gps_time = l0_tree["gps"].time.values

    # Keep GPS fixes only near the start and end of the record, opening an
    # interior gap far larger than the default gps_max_gap (300 s).
    keep = (gps_time < gps_time[0] + np.timedelta64(30, "s")) | (
        gps_time > gps_time[-1] - np.timedelta64(30, "s")
    )
    assert keep.any() and not keep.all()
    gapped_gps = l0_tree["gps"].to_dataset().sel(time=gps_time[keep])

    groups = {name: node.to_dataset() for name, node in l0_tree.children.items()}
    groups["gps"] = gapped_gps
    tree = xr.DataTree.from_dict({f"/{k}": v for k, v in groups.items()})
    tree.attrs = dict(l0_tree.attrs)

    mid_lo = int(0.4 * time.size)
    mid_hi = int(0.6 * time.size)
    assert mid_hi > mid_lo

    l1 = make_l1(tree, FCTDConfig(tc=TCParams()))
    ctd = l1["ctd"]
    lat = ctd.lat.data
    lon = ctd.lon.data
    # lon is masked across the gap; lat is interpolated between the fixes
    # either side, so depth (which needs only lat) survives the gap
    assert np.isnan(lon[mid_lo:mid_hi]).all()
    assert np.isfinite(lat).all()
    assert np.isfinite(ctd["depth"].data[mid_lo:mid_hi]).all()
    gps_kept = gapped_gps.lat.values
    assert lat[mid_lo:mid_hi].min() >= min(gps_kept) - 1e-9
    assert lat[mid_lo:mid_hi].max() <= max(gps_kept) + 1e-9
    assert np.isfinite(lon[:5]).all() and np.isfinite(lon[-5:]).all()
    assert ctd.attrs["latitude_source"] == "gps"
    assert ctd.attrs["n_lon_masked"] > 0
    assert "gps_max_gap" in ctd.attrs["position_note"]

    # a configured fallback no longer overrides interpolated GPS latitude
    l1b = make_l1(tree, FCTDConfig(tc=TCParams(), latitude_fallback=3.0))
    assert np.allclose(l1b["ctd"].lat.data, lat)
    assert np.isnan(l1b["ctd"].lon.data[mid_lo:mid_hi]).all()


def test_l1_no_gps_gap_leaves_no_position_note(l0_tree):
    ctd = make_l1(l0_tree, FCTDConfig(tc=TCParams()))["ctd"]
    assert ctd.attrs["n_lon_masked"] == 0
    assert "position_note" not in ctd.attrs


def test_l1_casts_group_matches_labels(l0_tree):
    l1 = make_l1(l0_tree, FCTDConfig(tc=TCParams()))
    casts = l1["casts"]
    n = casts.sizes["cast"]
    assert n >= 2
    assert l1["ctd"].cast.data.max() == n
    assert (l1["efe"].cast.data <= n).all()

    # Pin the time_ref plumbing: efe samples labeled with a given cast id
    # must actually fall within that cast's [start_time, end_time], and
    # every cast must label at least one efe sample (a <= n check alone
    # would pass even with zero labeled samples).
    efe = l1["efe"]
    for cast_id in casts.cast.data:
        start = casts.start_time.sel(cast=cast_id).values
        end = casts.end_time.sel(cast=cast_id).values
        labeled_time = efe.time.data[efe.cast.data == cast_id]
        assert labeled_time.size > 0
        assert labeled_time.min() >= start
        assert labeled_time.max() <= end


def test_l1_no_casts_raises(tmp_path):
    files = write_l0_files(tmp_path, n_files=1, minutes=12.0)  # default flat 5-dbar p_fn: no casts
    tree = concat_l0(files)
    with pytest.raises(ValueError, match="no casts detected"):
        make_l1(tree, FCTDConfig(tc=TCParams()))


def test_l1_no_gps_no_fallback_raises(tmp_path):
    files = write_l0_files(tmp_path, n_files=1, with_gps=False, p_fn=two_cast_p, minutes=12.0)
    tree = concat_l0(files)
    with pytest.raises(ValueError, match="latitude_fallback"):
        make_l1(tree, FCTDConfig(tc=TCParams()))
    cfg = FCTDConfig(latitude_fallback=2.0, tc=TCParams())
    l1 = make_l1(tree, cfg)
    assert l1["ctd"].attrs["latitude_source"] == "fallback"


def test_l1_keep_counts_tree_raises(tmp_path):
    # concat_l0(..., keep_counts=True) leaves count-typed t_raw/c_raw on
    # the ctd group; make_l1's own t_raw/c_raw assignment in _apply_tc
    # would silently overwrite them with physical t/c copies, so make_l1
    # must refuse rather than discard the kept data.
    files = write_l0_files(tmp_path, n_files=3, minutes=12.0, p_fn=two_cast_p)
    tree = concat_l0(files, keep_counts=True)
    assert "t_raw" in tree["ctd"].to_dataset().data_vars
    with pytest.raises(ValueError, match="keep_counts"):
        make_l1(tree, FCTDConfig(tc=TCParams()))


def test_l1_thermal_mass_profile_reaches_the_correction(l0_tree):
    prof = {"p": [10.0, 200.0], "product": [0.12, 0.24]}
    base = dict(lag=0.03, tau_t=0.05, lowpass=4.0, thermal_mass=True, beta=1 / 12)
    with_prof = make_l1(l0_tree, FCTDConfig(tc=TCParams(**base, thermal_mass_profile=prof)))
    const = make_l1(l0_tree, FCTDConfig(tc=TCParams(**base, alpha=0.01)))
    assert "product=[0.12, 0.24]" in with_prof["ctd"].attrs["corrections"]
    assert not np.array_equal(with_prof["ctd"].c.data, const["ctd"].c.data)


def test_l1_and_grid_carry_modfish_provenance(l0_tree):
    from modfish.fctd.grid import grid_casts
    from modfish.utils import provenance

    l1 = make_l1(l0_tree, FCTDConfig())
    prov = provenance()
    for key, value in prov.items():
        assert l1["ctd"].attrs[key] == value
        assert grid_casts(l1).attrs[key] == value


def test_l1_dpdt_has_no_spike_across_a_time_gap(tmp_path):
    from modfish.fctd.concat import concat_l0

    files = write_l0_files(tmp_path / "gap", n_files=2, minutes=6.0, p_fn=two_cast_p, gap_s=240.0)
    l1 = make_l1(concat_l0(files, groups=("ctd", "gps")), FCTDConfig())
    dpdt = l1["ctd"]["dPdt"].values
    assert np.nanmax(np.abs(dpdt)) < 10.0


def test_position_interpolates_across_the_antimeridian():
    from modfish.fctd.l1 import _add_position

    t0 = np.datetime64("2025-12-06T00:00:00")
    ctd_t = t0 + (np.arange(0, 61, 1) * 1e9).astype("timedelta64[ns]")
    gps_t = t0 + (np.array([0, 60]) * 1e9).astype("timedelta64[ns]")
    ctd = xr.Dataset(coords=dict(time=ctd_t))
    gps = xr.Dataset(dict(lat=("time", [1.0, 1.0]), lon=("time", [179.9, -179.9])),
                     coords=dict(time=gps_t))
    lon = _add_position(ctd, gps, FCTDConfig())["lon"].values
    assert np.all(np.abs(lon) >= 179.9 - 1e-9)
    assert lon[30] == pytest.approx(180.0, abs=1e-6) or lon[30] == pytest.approx(-180.0, abs=1e-6)
    # an ordinary track keeps its sign convention
    gps2 = gps.assign(lon=("time", [-140.0, -139.0]))
    lon2 = _add_position(ctd, gps2, FCTDConfig())["lon"].values
    assert lon2[30] == pytest.approx(-139.5)
