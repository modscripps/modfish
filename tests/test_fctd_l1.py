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
    cfg = FCTDConfig(tc=TCParams(phase_match=False))
    l1 = make_l1(l0_tree, cfg)
    assert {"ctd", "casts", "efe"} <= set(l1.children)
    for v in ["t", "c", "t_raw", "c_raw", "SP", "SA", "CT", "sgth0", "depth", "cast"]:
        assert v in l1["ctd"]


def test_l1_corrections_off_t_equals_raw(l0_tree):
    l1 = make_l1(l0_tree, FCTDConfig(tc=TCParams(phase_match=False)))
    np.testing.assert_array_equal(l1["ctd"].t.data, l1["ctd"].t_raw.data)
    assert "phase_correct" not in l1["ctd"].attrs["corrections"]


def test_l1_phase_match_stamps_attrs_keeps_axis(l0_tree):
    cfg = FCTDConfig(tc=TCParams(phase_match=True, tcfit=(50, 290)))
    l1 = make_l1(l0_tree, cfg)
    assert l1["ctd"].sizes["time"] == l0_tree["ctd"].sizes["time"]
    assert "tau1" in l1["ctd"].attrs
    assert np.isnan(l1["ctd"].t.data).sum() > 0        # trimmed edges are NaN
    assert not np.array_equal(l1["ctd"].t.data, l1["ctd"].t_raw.data)
    # Explicit tcfit is recorded as-given: dedicated attr and corrections string.
    assert tuple(l1["ctd"].attrs["tcfit"]) == (50, 290)
    assert "tcfit=(50, 290)" in l1["ctd"].attrs["corrections"]


def test_l1_phase_match_default_tcfit_resolved_not_none(l0_tree):
    # When tcfit is left at its config default (None), phase_correct picks
    # a range internally (tc.add_tcfit_default); the resolved range, not
    # the unresolved None, must land in ctd.attrs and the corrections
    # string.
    cfg = FCTDConfig(tc=TCParams(phase_match=True, tcfit=None))
    l1 = make_l1(l0_tree, cfg)
    ctd = l1["ctd"]
    assert ctd.attrs["tcfit"] is not None
    assert "tcfit=None" not in ctd.attrs["corrections"]
    assert f"tcfit={ctd.attrs['tcfit']}" in ctd.attrs["corrections"]


def test_l1_phase_match_thermal_mass_viscous_conductivity_finite(l0_tree):
    # Regression test: phase_correct's trimmed output must feed
    # thermal_mass_correction (and viscous heating) while still NaN-free,
    # not after being reindexed onto the full axis. thermal_mass's
    # recursive filter propagates the first NaN it sees through every
    # later sample, so a premature reindex would silently turn the whole
    # `c` record NaN.
    cfg = FCTDConfig(
        tc=TCParams(
            phase_match=True,
            thermal_mass=True,
            viscous_heating=True,
            tcfit=(50, 290),
        )
    )
    l1 = make_l1(l0_tree, cfg)
    ctd = l1["ctd"]

    c = ctd.c.data
    finite = np.isfinite(c)
    assert finite.mean() > 0.95

    # NaNs, if any, are confined to the trimmed edges, not scattered
    # through the interior.
    interior_lo = int(0.05 * c.size)
    interior_hi = int(0.95 * c.size)
    nan_idx = np.flatnonzero(~finite)
    if nan_idx.size:
        assert not np.any((nan_idx > interior_lo) & (nan_idx < interior_hi))

    t = ctd.t.data
    t_raw = ctd.t_raw.data
    interior = slice(interior_lo, interior_hi)
    assert np.isfinite(t[interior]).all()
    assert not np.allclose(t[interior], t_raw[interior])

    corrections = ctd.attrs["corrections"]
    assert "phase_correct" in corrections
    assert "thermal_mass_correction" in corrections
    assert "viscous_heating_temperature_correction" in corrections
    assert "Pr=" in corrections
    assert "scale=" in corrections

    # t/c attrs (units, long_name) must survive the correction chain,
    # including the plain-arithmetic viscous-heating step.
    assert ctd["t"].attrs.get("units") == ctd["t_raw"].attrs.get("units")
    assert ctd["c"].attrs.get("units") == ctd["c_raw"].attrs.get("units")
    assert ctd["t"].attrs.get("long_name") == ctd["t_raw"].attrs.get("long_name")
    assert ctd["c"].attrs.get("long_name") == ctd["c_raw"].attrs.get("long_name")


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

    l1 = make_l1(tree, FCTDConfig(tc=TCParams(phase_match=False)))
    lat = l1["ctd"].lat.data
    lon = l1["ctd"].lon.data
    assert np.isnan(lon[mid_lo:mid_hi]).all()
    assert np.isnan(lat[mid_lo:mid_hi]).all()
    assert np.isfinite(lon[:5]).all()
    assert np.isfinite(lat[:5]).all()
    assert np.isfinite(lon[-5:]).all()
    assert np.isfinite(lat[-5:]).all()

    cfg_fallback = FCTDConfig(
        tc=TCParams(phase_match=False), latitude_fallback=3.0
    )
    l1b = make_l1(tree, cfg_fallback)
    assert np.allclose(l1b["ctd"].lat.data[mid_lo:mid_hi], 3.0)
    assert np.isnan(l1b["ctd"].lon.data[mid_lo:mid_hi]).all()
    assert np.isfinite(l1b["ctd"].lat.data[:5]).all()
    assert not np.allclose(l1b["ctd"].lat.data[:5], 3.0)


def test_l1_casts_group_matches_labels(l0_tree):
    l1 = make_l1(l0_tree, FCTDConfig(tc=TCParams(phase_match=False)))
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
        make_l1(tree, FCTDConfig(tc=TCParams(phase_match=False)))


def test_l1_no_gps_no_fallback_raises(tmp_path):
    files = write_l0_files(tmp_path, n_files=1, with_gps=False, p_fn=two_cast_p, minutes=12.0)
    tree = concat_l0(files)
    with pytest.raises(ValueError, match="latitude_fallback"):
        make_l1(tree, FCTDConfig(tc=TCParams(phase_match=False)))
    cfg = FCTDConfig(latitude_fallback=2.0, tc=TCParams(phase_match=False))
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
        make_l1(tree, FCTDConfig(tc=TCParams(phase_match=False)))
