import numpy as np
import pytest
import xarray as xr

from modfish.chi import add_chi
from modfish.chi.config import FLAG_MEANINGS, ChiParams
from modfish.fctd.concat import concat_l0
from modfish.fctd.config import FCTDConfig
from modfish.fctd.l1 import make_l1
from synth_l0 import two_cast_p, write_l0_files

VARS = {"cast", "depth", "p", "lon", "lat", "spd", "chi", "kmax", "n_bins", "range_id",
        "chi_flag", "chi_tot", "eps_chi", "r", "n2", "Tz", "Sz", "Rrho"}


@pytest.fixture()
def l1_and_files(tmp_path):
    files = write_l0_files(tmp_path, n_files=3, minutes=6.0, p_fn=two_cast_p)
    l1 = make_l1(concat_l0(files, groups=("ctd", "gps")), FCTDConfig())
    return l1, files


def test_add_chi_builds_group(l1_and_files):
    l1, files = l1_and_files
    params = ChiParams(enabled=True, gain=50.0, gain_source="synthetic")
    out = add_chi(l1, files, params)
    assert "chi" in out.children and "ctd" in out.children and "efe" not in out.children
    chi = out["chi"].to_dataset()
    assert VARS <= set(chi.variables)
    assert chi.chi_flag.dtype == np.uint8
    assert chi.sizes["time"] > 100
    step = np.diff(chi.time.values[:10]).astype("timedelta64[ms]").astype(int)
    assert np.all(np.abs(step - 250) <= 4)
    assert chi.attrs["gain"] == 50.0 and chi.attrs["gain_source"] == "synthetic"
    assert chi.attrs["antialias"] == "som_sinc4"
    assert chi.attrs["flag_meanings"] == FLAG_MEANINGS
    assert "modfish_version" in chi.attrs and chi.attrs["n_ranges"] >= 1
    assert set(np.unique(chi.cast.values)) - {0} == set(out["casts"].to_dataset().cast.values)
    efe0 = xr.open_dataset(files[0], group="efe")
    expected_t0 = efe0["time"].values[0] + np.timedelta64(int(params.window / 2 * 1e9), "ns")
    assert chi.time.values[0] == expected_t0


def test_add_chi_windows_inside_casts_have_speed_and_depth(l1_and_files):
    l1, files = l1_and_files
    chi = add_chi(l1, files, ChiParams(enabled=True, gain=50.0))["chi"].to_dataset()
    inside = chi.isel(time=(chi.cast > 0).values)
    assert np.isfinite(inside.depth.values).all()
    assert np.nanmedian(inside.spd.values) > 0.5


def test_add_chi_closure_off(l1_and_files):
    l1, files = l1_and_files
    chi = add_chi(l1, files, ChiParams(enabled=True, gain=50.0, closure=False))["chi"].to_dataset()
    assert "chi_tot" not in chi and "chi" in chi


def test_add_chi_requires_enabled_params(l1_and_files):
    l1, files = l1_and_files
    with pytest.raises(ValueError, match="gain"):
        add_chi(l1, files, ChiParams())
