import numpy as np
import pandas as pd
import pytest
import xarray as xr

from modfish.fctd.concat import concat_l0
from synth_l0 import FAKE_CAL_TA0, write_l0_files


def test_concat_merges_sorted_unique_time(tmp_path):
    files = write_l0_files(tmp_path, n_files=3, minutes=5.0)
    tree = concat_l0(files)
    t = tree["ctd"].time.data
    assert (np.diff(t) > np.timedelta64(0, "ns")).all()   # sorted, unique
    assert tree.attrs["n_files"] == 3


def test_concat_drops_counts_by_default(tmp_path):
    files = write_l0_files(tmp_path, n_files=2)
    tree = concat_l0(files)
    assert "t_raw" not in tree["ctd"]
    assert "pt_raw" not in tree["ctd"]
    tree2 = concat_l0(files, keep_counts=True)
    assert "t_raw" in tree2["ctd"]


def test_concat_union_of_groups(tmp_path):
    files = write_l0_files(tmp_path, n_files=2, with_efe=False)
    more = write_l0_files(tmp_path / "b", n_files=1, with_efe=True)
    tree = concat_l0(files + more)
    assert "efe" in tree.children
    assert tree["efe"].sizes["time"] > 0


def test_concat_empty_list_raises(tmp_path):
    with pytest.raises(ValueError):
        concat_l0([])


def test_concat_no_ctd_group_raises(tmp_path):
    ds = xr.Dataset(
        coords={"time": ("time", pd.date_range("2026-01-01", periods=3, freq="s"))},
        data_vars={"lat": ("time", [1.0, 2.0, 3.0])},
    )
    tree = xr.DataTree.from_dict({"/gps": ds})
    path = tmp_path / "no_ctd.nc"
    tree.to_netcdf(path)
    with pytest.raises(ValueError):
        concat_l0([path])


def test_concat_preserves_variable_and_group_attrs(tmp_path):
    files = write_l0_files(tmp_path, n_files=2, minutes=1.0)
    tree = concat_l0(files)
    ctd = tree["ctd"].ds
    assert ctd.p.attrs["units"] == "dbar"
    assert ctd.t.attrs["long_name"] == "temperature"
    assert ctd.c.attrs["units"] == "S/m"
    assert ctd.attrs["ta0"] == FAKE_CAL_TA0


def test_concat_groups_loads_only_the_requested_groups(tmp_path):
    files = write_l0_files(tmp_path, n_files=2, minutes=1.0, with_efe=True, with_gps=True)
    full = concat_l0(files)
    tree = concat_l0(files, groups=("ctd", "gps"))
    assert set(tree.children) == {"ctd", "gps"}
    xr.testing.assert_identical(tree["ctd"].to_dataset(), full["ctd"].to_dataset())
    assert tree.attrs["groups"] == ["ctd", "gps"]
    assert full.attrs["groups"] == "all"


def test_concat_groups_missing_in_a_file_is_tolerated(tmp_path):
    files = write_l0_files(tmp_path, n_files=2, minutes=1.0, with_gps=False)
    tree = concat_l0(files, groups=("ctd", "gps"))
    assert set(tree.children) == {"ctd"}


def test_concat_groups_without_ctd_raises(tmp_path):
    files = write_l0_files(tmp_path, n_files=1, minutes=1.0)
    with pytest.raises(ValueError, match="ctd"):
        concat_l0(files, groups=("gps",))
