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


def test_concat_groups_accepts_a_one_shot_generator(tmp_path):
    files = write_l0_files(tmp_path, n_files=2, minutes=1.0, with_gps=True)
    tree = concat_l0(files, groups=(g for g in ("ctd", "gps")))
    assert set(tree.children) == {"ctd", "gps"}
    assert tree.attrs["groups"] == ["ctd", "gps"]


def _write_ctd_file(path, time, p):
    """One L0 file holding only a `ctd` group with the given samples, in file order."""
    p = np.asarray(p, dtype=float)
    ds = xr.Dataset(
        coords={"time": ("time", np.asarray(time, dtype="datetime64[ns]"))},
        data_vars={"p": ("time", p), "t": ("time", 10.0 - p / 100), "c": ("time", 4.0 - p / 1000)},
    )
    xr.DataTree.from_dict({"/ctd": ds}).to_netcdf(path)
    return path


def _ramp(t0, n, p0, rate=-2.4, dt_ms=62.5):
    """`n` samples at 16 Hz from `t0`, pressure changing at `rate` dbar/s."""
    time = np.datetime64(t0, "ns") + (np.arange(n) * dt_ms * 1e6).astype("timedelta64[ns]")
    return time, p0 + rate * np.arange(n) * dt_ms / 1000


def test_concat_drops_misstamped_leading_records(tmp_path):
    # modscripps/modfish#41: after a restart, a file's first records can carry
    # timestamps 0.2 to 2.3 s late while their values continue into the rest of
    # the file. Sorted by time they interleave with the real record.
    ta, pa = _ramp("2025-12-13T08:08:00", 64, 800.0)
    tb, pb = _ramp("2025-12-13T08:11:47.785", 68, 723.7)
    tb = tb.copy()
    tb[:4] += np.timedelta64(2277, "ms")  # stamped late, values in sequence
    a = _write_ctd_file(tmp_path / "a.nc", ta, pa)
    b = _write_ctd_file(tmp_path / "b.nc", tb, pb)
    ctd = concat_l0([a, b])["ctd"].ds
    np.testing.assert_array_equal(ctd.p.values[64:], pb[4:])
    assert (np.diff(ctd.time.values) > np.timedelta64(0, "ns")).all()
    assert ctd.attrs["n_misstamped"] == 4


def test_concat_exact_duplicate_records_are_not_misstamped(tmp_path):
    t, p = _ramp("2025-12-10T23:53:11", 32, 763.0, rate=3.0)
    order = np.r_[np.arange(12), 10, 11, np.arange(12, 32)]  # two records repeated
    f = _write_ctd_file(tmp_path / "a.nc", t[order], p[order])
    ctd = concat_l0([f])["ctd"].ds
    np.testing.assert_array_equal(ctd.p.values, p)
    assert ctd.attrs["n_misstamped"] == 0


def test_concat_drops_stale_first_sample_after_gap(tmp_path):
    # modscripps/modfish#41: the first sample after an acquisition gap can be a
    # reading 15 to 20 s old, 30 to 50 dbar off the settled record.
    ta, pa = _ramp("2024-11-27T23:05:26", 64, 24.0)
    tb, pb = _ramp("2024-11-27T23:16:17.384", 64, 1067.0 - 47.2 + 2.4 * 0.0625)
    pb = pb.copy()
    pb[0] = 1067.0
    a = _write_ctd_file(tmp_path / "a.nc", ta, pa)
    b = _write_ctd_file(tmp_path / "b.nc", tb, pb)
    ctd = concat_l0([a, b])["ctd"].ds
    assert ctd.sizes["time"] == 127
    np.testing.assert_array_equal(ctd.p.values[64:], pb[1:])
    assert ctd.attrs["n_stale_after_gap"] == 1


def test_concat_keeps_first_sample_after_gap_below_threshold(tmp_path):
    ta, pa = _ramp("2024-11-27T23:05:26", 64, 24.0)
    tb, pb = _ramp("2024-11-27T23:16:17.384", 64, 1019.8)
    pb = pb.copy()
    pb[0] += 3.0  # inside the 5 dbar tolerance
    a = _write_ctd_file(tmp_path / "a.nc", ta, pa)
    b = _write_ctd_file(tmp_path / "b.nc", tb, pb)
    ctd = concat_l0([a, b])["ctd"].ds
    assert ctd.sizes["time"] == 128
    assert ctd.attrs["n_stale_after_gap"] == 0


def test_concat_does_not_test_the_first_sample_of_the_record(tmp_path):
    t, p = _ramp("2024-11-27T23:05:26", 64, 24.0)
    p = p.copy()
    p[0] += 40.0
    f = _write_ctd_file(tmp_path / "a.nc", t, p)
    ctd = concat_l0([f])["ctd"].ds
    assert ctd.sizes["time"] == 64


def test_concat_drops_whole_misstamped_leading_block(tmp_path):
    # MOTIVE 2025 d11: four leading records stamped 0.21 s late, less than the
    # block's own 0.25 s, so the first of them does not overlap a later record.
    t, p = _ramp("2025-12-11T00:56:04.796", 64, 848.6)
    t = t.copy()
    t[:4] += np.timedelta64(210, "ms")
    f = _write_ctd_file(tmp_path / "a.nc", t, p)
    ctd = concat_l0([f])["ctd"].ds
    np.testing.assert_array_equal(ctd.p.values, p[4:])
    assert ctd.attrs["n_misstamped"] == 4
