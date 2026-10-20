import numpy as np
import pytest
import xarray as xr

import modfish.utils
from modfish.chi import add_chi
from modfish.chi.config import FLAG_MEANINGS, FLAG_N2, FLAG_NOENV, ChiParams
from modfish.chi.load import load_c1
from modfish.chi.pipeline import _record_floor, chi_dataset
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
    assert chi.attrs.get("modfish_commit") == modfish.utils.provenance().get("modfish_commit")
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


def test_nan_closure_input_flags_no_environment(l1_and_files):
    """A window with a spectrum but a NaN closure input reads as missing
    environment (bit 64), not as inverted stratification (bit 16)."""
    l1, files = l1_and_files
    params = ChiParams(enabled=True, gain=50.0)
    c1, ranges = load_c1(files, gap=params.gap)
    ctd = l1["ctd"].to_dataset()
    sgth0 = ctd["sgth0"].values.copy()
    lo = sgth0.size // 3
    sgth0[lo : lo + 16 * 120] = np.nan  # 2 min, wider than closure_window
    ctd["sgth0"] = ("time", sgth0)
    ds = chi_dataset(ctd, l1["casts"].to_dataset(), c1, ranges, params)
    hit = np.isnan(ds.n2.values) & np.isfinite(ds.chi.values)
    assert hit.sum() > 10
    assert np.all(ds.chi_flag.values[hit] & FLAG_NOENV)
    assert not np.any(ds.chi_flag.values[hit] & FLAG_N2)


def test_chi_group_carries_phi_and_floor_provenance(l1_and_files):
    l1, files = l1_and_files
    params = ChiParams(enabled=True, gain=50.0, gain_source="synthetic")
    chi = add_chi(l1, files, params)["chi"].to_dataset()
    assert "phi" in chi.data_vars
    assert chi["phi"].dims == ("time",)
    for key in ("noise_source", "noise_records", "noise_measured", "noise_nu",
                "record_floor_f", "record_floor_n", "record_flatness_20hz"):
        assert key in chi.attrs, key
    assert len(chi.attrs["record_floor_f"]) == len(chi.attrs["record_floor_n"])
    assert "snr" not in chi.attrs and "noise_floor" not in chi.attrs


def test_noise_none_leaves_phi_at_zero(l1_and_files):
    l1, files = l1_and_files
    params = ChiParams(enabled=True, gain=50.0, gain_source="synthetic", noise=None)
    chi = add_chi(l1, files, params)["chi"].to_dataset()
    assert float(chi["phi"].max()) == 0.0
    assert "noise_source" not in chi.attrs


def test_chi_group_record_floor_estimate_is_finite_on_the_fixture(l1_and_files):
    """End-to-end: `add_chi`'s diagnostic capture and `_record_floor`'s
    consumption of it (the `env["depth"][diag_idx]` indexing, the
    multi-range `diag` accumulation, the real array shapes) must line up
    well enough that the fixture clears both spectrum-count guards, not
    just the hand-built-tuple path exercised above."""
    l1, files = l1_and_files
    params = ChiParams(enabled=True, gain=50.0, gain_source="synthetic")
    chi = add_chi(l1, files, params)["chi"].to_dataset()
    assert np.all(np.isfinite(chi.attrs["record_floor_n"]))
    assert np.all(np.array(chi.attrs["record_floor_n"]) > 0)
    assert np.isfinite(chi.attrs["record_flatness_20hz"])


def test_record_floor_pools_only_the_largest_frequency_grid():
    """Two ranges of one deployment can produce different Welch grid
    lengths (different `fs`). `_record_floor` must group diagnostic
    entries by their frequency grid and pool the largest group instead of
    concatenating mismatched arrays."""
    rng = np.random.default_rng(0)
    nu = 11.04
    f_small = np.linspace(4.0, 160.0, 41)
    f_big = np.linspace(4.0, 160.0, 82)
    dep_small = rng.uniform(0, 100, size=50)
    dep_big = rng.uniform(0, 100, size=1200)
    P_small = rng.uniform(1e-10, 2e-10, size=(50, f_small.size))
    P_big = rng.uniform(1e-10, 2e-10, size=(1200, f_big.size))
    diag = [(P_small, dep_small, f_small, nu), (P_big, dep_big, f_big, nu)]
    _, n, flat = _record_floor(diag)
    assert np.all(np.isfinite(n))
    assert np.isfinite(flat)


def test_record_floor_uses_the_current_welch_dof():
    """With nsec = 0.25 s the window spectra have nu = 27, not the shipped
    floor's 11. Inverting the chi-square quantile with the wrong nu puts
    the floor estimate about 1.4x high."""
    from modfish.chi.spectra import welch_dof, window_spectrum

    fs, window, nsec, level = 325.52, 2.0, 0.25, 1e-9
    rng = np.random.default_rng(2)
    nw = int(round(window * fs))
    sigma = np.sqrt(level * fs / 2)
    spectra = [window_spectrum(rng.normal(0.0, sigma, nw), fs, nsec) for _ in range(1500)]
    f = spectra[0][0]
    P = np.array([s[1] for s in spectra])
    dep = rng.uniform(0, 100, size=P.shape[0])
    diag = [(P, dep, f, welch_dof(fs, window, nsec))]
    _, n, _ = _record_floor(diag)
    # 4 Hz is the first bin above DC at 0.25 s segments, where the
    # per-segment detrend removes power; check 10, 20 and 37.5 Hz
    assert np.all(n[1:] == pytest.approx(level, rel=0.15))
    # the shipped floor's nu in place of the spectra's own would read high
    _, n_wrong, _ = _record_floor([(P, dep, f, 11.04)])
    assert np.all(n_wrong[1:] > 1.25 * level)


def test_chi_group_carries_window_direction(l1_and_files):
    """Each window carries the direction of the cast it falls in, so upcast
    windows (sensor likely in the wake) can be told apart without a join."""
    l1, files = l1_and_files
    chi = add_chi(l1, files, ChiParams(enabled=True, gain=50.0))["chi"].to_dataset()
    casts = l1["casts"].to_dataset()
    assert "direction" in chi
    assert set(np.unique(chi.direction.values.astype(str))) <= {"down", "up", "none"}
    assert np.all(chi.direction.values[chi.cast.values == 0] == "none")
    for cid, d in zip(casts.cast.values, casts.direction.values, strict=True):
        assert np.all(chi.direction.values[chi.cast.values == cid] == d)
    assert {"down", "up"} <= set(np.unique(chi.direction.values.astype(str)))
