import numpy as np
import pytest
import xarray as xr

from modfish.fctd.concat import concat_l0
from synth_l0 import write_l0_files


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
