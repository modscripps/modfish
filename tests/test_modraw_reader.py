import numpy as np
import pytest
import xarray as xr

import modfish


@pytest.fixture(scope="module")
def tree(rootdir):
    return modfish.modraw.read(rootdir / "data/FCTD_modraw_excerpt.modraw")


def test_read_returns_datatree_with_expected_groups(tree):
    assert isinstance(tree, xr.DataTree)
    assert set(tree.children) == {"ctd", "efe", "ecop", "gps"}  # no ALTI in fixture


def test_read_ctd_group_contains_load_ctd(tree, rootdir):
    # read() frames the full file and captures SB49 blocks that sit inside
    # the (wrongly) declared header span, so it holds a superset of what the
    # read_body-based load_ctd sees; the legacy records must match exactly.
    ref = modfish.modraw.load_ctd(rootdir / "data/FCTD_modraw_excerpt.modraw")
    n_extra = tree["ctd"].ds.sizes["time"] - ref.sizes["time"]
    assert n_extra >= 0
    np.testing.assert_array_equal(tree["ctd"].time.values[n_extra:], ref.time.values)
    np.testing.assert_allclose(tree["ctd"].ds.p.values[n_extra:], ref.p.values)


def test_read_root_attrs_carry_quality_tallies(tree):
    # n_resync counts failed frame attempts. On a healthy full-file scan
    # that includes every interleaved NMEA sentence (not SOM-framed, so the
    # scanner correctly rejects them: 147 here) plus 5 header text lines.
    # Corruption is signaled by the checksum/length tallies, which the spec
    # requires to be zero on clean files.
    assert tree.attrs["n_resync"] == 152
    assert tree.attrs["n_bad_checksum"] == 0
    assert tree.attrs["n_blocks_SB49"] == 238
    assert tree.attrs["vehicle"] == "FCTD1"


def test_read_laptop_time_diagnostics_present(tree):
    assert tree.ds.sizes["block"] == tree.attrs["n_frames"]
    assert tree.ds.block_tag.values[0] in ("SOM3", "DCAL", "SB49", "EFE4", "ECOP", "ALTI")


def test_read_roundtrips_through_netcdf(tree, tmp_path):
    path = tmp_path / "excerpt.nc"
    tree.to_netcdf(path)
    back = xr.open_datatree(path)
    np.testing.assert_allclose(back["ctd"].ds.p.values, tree["ctd"].ds.p.values)
