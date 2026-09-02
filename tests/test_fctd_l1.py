import numpy as np
import pytest

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


def test_l1_casts_group_matches_labels(l0_tree):
    l1 = make_l1(l0_tree, FCTDConfig(tc=TCParams(phase_match=False)))
    n = l1["casts"].sizes["cast"]
    assert n >= 2
    assert l1["ctd"].cast.data.max() == n
    assert (l1["efe"].cast.data <= n).all()


def test_l1_no_gps_no_fallback_raises(tmp_path):
    files = write_l0_files(tmp_path, n_files=1, with_gps=False, p_fn=two_cast_p, minutes=12.0)
    tree = concat_l0(files)
    with pytest.raises(ValueError, match="latitude_fallback"):
        make_l1(tree, FCTDConfig(tc=TCParams(phase_match=False)))
    cfg = FCTDConfig(latitude_fallback=2.0, tc=TCParams(phase_match=False))
    l1 = make_l1(tree, cfg)
    assert l1["ctd"].attrs["latitude_source"] == "fallback"
