import numpy as np
import pytest
import xarray as xr

from modfish.chi.batchelor import FractionTable, band_fraction
from modfish.chi.closure import closure, solve_epsilon, stratification
from modfish.chi.config import FLAG_N2, FLAG_RCLIP, FLAG_RRHO, ChiParams

P = ChiParams(enabled=True, gain=50.0)


@pytest.fixture(scope="module")
def table():
    return FractionTable.build(1.0, 16.67, P.nu, P.D, P.q)


def _ctd(minutes=5.0, fs=16.0, tz=-0.01, sz=0.0):
    n = int(minutes * 60 * fs)
    time = np.datetime64("2025-12-06T00:00:00") + (np.arange(n) / fs * 1e9).astype("timedelta64[ns]")
    p = 100.0 + 3.0 * np.arange(n) / fs  # 3 dbar/s descent
    t = 20.0 + tz * (p - 100.0)
    SP = 35.0 + sz * (p - 100.0)
    lon, lat = np.full(n, -139.5), np.full(n, 1.0)
    import gsw
    SA = gsw.SA_from_SP(SP, p, lon, lat)
    CT = gsw.CT_from_t(SA, t, p)
    sgth0 = gsw.sigma0(SA, CT)
    depth = -gsw.z_from_p(p, lat)
    return xr.Dataset(
        dict(p=("time", p), t=("time", t), SP=("time", SP), SA=("time", SA), CT=("time", CT),
             sgth0=("time", sgth0), lon=("time", lon), lat=("time", lat), depth=("time", depth)),
        coords=dict(time=time))


def test_stratification_recovers_linear_gradients():
    ctd = _ctd(tz=-0.01)
    centers = ctd.time.values[16 * 60 : 16 * 240 : 16 * 10]
    s = stratification(ctd, centers, P)
    assert set(s.data_vars) >= {"n2", "Tz", "Sz", "alpha", "beta", "Rrho"}
    assert s.Tz.values == pytest.approx(-0.01, rel=0.02)
    assert np.all(s.n2.values > 0)
    assert np.abs(s.Sz.values).max() < 1e-4
    assert np.all(np.abs(s.Rrho.values) > 100)  # Sz ~ 0


def test_stratification_negative_n2_when_inverted():
    ctd = _ctd(tz=+0.05)
    centers = ctd.time.values[16 * 60 : 16 * 240 : 16 * 10]
    s = stratification(ctd, centers, P)
    assert np.all(s.n2.values < 0)


def test_solve_epsilon_matches_fixed_point(table):
    chi_pe_hat = np.array([1e-10, 1e-9, 1e-8])
    kmax = np.full(3, 12.5)
    eps, r, clipped = solve_epsilon(chi_pe_hat, kmax, table, P.gamma)
    for x, e, rr in zip(chi_pe_hat, eps, r):
        e_fp = 1e-9
        for _ in range(60):
            e_fp = x / (2 * P.gamma * band_fraction(e_fp, 1.0, 12.5, P.nu, P.D, P.q))
        assert e == pytest.approx(e_fp, rel=0.02)
        assert rr == pytest.approx(band_fraction(e_fp, 1.0, 12.5, P.nu, P.D, P.q), rel=0.02)
        assert 2 * P.gamma * e * rr == pytest.approx(x, rel=0.02)
    assert not clipped.any()


def test_solve_epsilon_clips_and_handles_nan(table):
    eps, r, clipped = solve_epsilon(np.array([1e-16, np.nan, -1.0]), np.full(3, 12.5), table, P.gamma)
    assert clipped[0] and r[0] == pytest.approx(table.r_of(12.5)[0]) and eps[0] == table.eps_grid[0]
    assert np.isnan(eps[1:]).all() and np.isnan(r[1:]).all()


def test_solve_epsilon_above_table_is_nan_and_clipped(table):
    r_col = table.r_of(12.5)
    g_last = 2 * P.gamma * table.eps_grid[-1] * r_col[-1]
    eps, r, clipped = solve_epsilon(np.array([1e3 * g_last]), np.full(1, 12.5), table, P.gamma)
    assert clipped[0]
    assert np.isnan(eps[0]) and np.isnan(r[0])


def test_closure_outputs_and_flags(table):
    ctd = _ctd(tz=-0.01)
    centers = ctd.time.values[16 * 60 : 16 * 240 : 16 * 10]
    strat = stratification(ctd, centers, P)
    chi = np.full(centers.size, 6e-11)
    kmax = np.full(centers.size, 12.5)
    out = closure(chi, kmax, strat, P, table)
    assert set(out.data_vars) >= {"chi_pe", "eps_chi", "r", "chi_tot", "flag"}
    assert np.all(out.chi_tot.values >= chi)
    assert np.all(out.r.values <= 1.0) and np.all(out.r.values > 0)
    assert np.isfinite(out.eps_chi.values).all()
    # inverted stratification: NaN with the n2 flag
    strat_inv = stratification(_ctd(tz=+0.05), centers, P)
    out = closure(chi, kmax, strat_inv, P, table)
    assert np.isnan(out.eps_chi.values).all() and np.all(out.flag.values & FLAG_N2)
    # Rrho near zero caps the factor
    strat_cap = strat.copy()
    strat_cap["Rrho"] = ("time", np.full(centers.size, 0.1))
    out = closure(chi, kmax, strat_cap, P, table)
    assert np.all(out.flag.values & FLAG_RRHO)
    assert np.all(out.chi_pe.values == pytest.approx(
        (P.g * strat.alpha.values) ** 2 * chi / strat.n2.values * P.rrho_factor_max))
    # tiny chi_pe: r clipped at the table's own floor value
    out = closure(np.full(centers.size, 1e-20), kmax, strat, P, table)
    assert np.all(out.flag.values & FLAG_RCLIP)
    assert np.all(out.r.values == pytest.approx(table.r_of(12.5)[0]))
    # huge chi_pe: above the table, eps_chi and chi_tot are NaN, flagged
    huge_chi = chi * 1e6
    out = closure(huge_chi, kmax, strat, P, table)
    assert np.all(out.flag.values & FLAG_RCLIP)
    assert np.isnan(out.eps_chi.values).all() and np.isnan(out.chi_tot.values).all()


def test_closure_nan_rrho_stays_nan_unflagged(table):
    ctd = _ctd(tz=-0.01)
    centers = ctd.time.values[16 * 60 : 16 * 240 : 16 * 10]
    strat = stratification(ctd, centers, P)
    chi = np.full(centers.size, 6e-11)
    kmax = np.full(centers.size, 12.5)
    strat_mixed = strat.copy()
    Rrho = strat.Rrho.values.copy()
    Rrho[0] = np.nan
    Rrho[1] = 0.0
    strat_mixed["Rrho"] = ("time", Rrho)
    out = closure(chi, kmax, strat_mixed, P, table)
    assert np.isnan(out.chi_pe.values[0])
    assert np.isnan(out.eps_chi.values[0])
    assert np.isnan(out.chi_tot.values[0])
    assert out.flag.values[0] == 0
    assert out.flag.values[1] & FLAG_RRHO
    assert out.chi_pe.values[1] == pytest.approx(
        (P.g * strat.alpha.values[1]) ** 2 * chi[1] / strat.n2.values[1] * P.rrho_factor_max)


def test_chi_pe_scales_as_alpha_squared(table):
    """The (A2) prefactor is (g alpha)^2 / n2, so chi_pe goes as alpha^2."""
    ctd = _ctd(tz=-0.01)
    centers = ctd.time.values[16 * 60 : 16 * 240 : 16 * 10]
    strat = stratification(ctd, centers, P)
    chi = np.full(centers.size, 6e-11)
    kmax = np.full(centers.size, 12.5)
    base = closure(chi, kmax, strat, P, table)
    strat2 = strat.copy()
    strat2["alpha"] = ("time", 2.0 * strat["alpha"].values)
    doubled = closure(chi, kmax, strat2, P, table)
    assert np.isfinite(base.chi_pe.values).all()
    assert doubled.chi_pe.values == pytest.approx(4.0 * base.chi_pe.values, rel=1e-12)


def test_one_nan_does_not_void_the_boxcar():
    """A single missing sgth0 sample leaves n2 finite at every center."""
    ctd = _ctd(tz=-0.01)
    centers = ctd.time.values[16 * 60 : 16 * 240 : 16 * 10]
    base = stratification(ctd, centers, P)
    holed = ctd.copy()
    sgth0 = ctd["sgth0"].values.copy()
    sgth0[sgth0.size // 2] = np.nan
    holed["sgth0"] = ("time", sgth0)
    one = stratification(holed, centers, P)
    assert np.isfinite(one.n2.values).all()
    assert one.n2.values == pytest.approx(base.n2.values, rel=0.01)


def test_nan_stretch_longer_than_the_window_gives_nan_n2():
    """Past half the boxcar invalid, the mean is NaN rather than biased."""
    ctd = _ctd(tz=-0.01)
    size = max(int(round(P.closure_window * 16.0)), 3)
    mid = ctd.sizes["time"] // 2
    holed = ctd.copy()
    sgth0 = ctd["sgth0"].values.copy()
    sgth0[mid - size : mid + size] = np.nan
    holed["sgth0"] = ("time", sgth0)
    out = stratification(holed, ctd.time.values[[mid]], P)
    assert np.isnan(out.n2.values[0])


def test_nan_n2_is_not_flagged_as_inverted(table):
    """Bit 16 means a measured n2 <= 0; a NaN n2 leaves it clear so that
    `chi_dataset` can mark the window as missing environment (bit 64)."""
    ctd = _ctd(tz=-0.01)
    centers = ctd.time.values[16 * 60 : 16 * 240 : 16 * 10]
    strat = stratification(ctd, centers, P)
    n2 = strat["n2"].values.copy()
    n2[0] = np.nan
    n2[1] = -1e-6
    strat = strat.copy()
    strat["n2"] = ("time", n2)
    chi = np.full(centers.size, 6e-11)
    out = closure(chi, np.full(centers.size, 12.5), strat, P, table)
    assert out.flag.values[0] == 0
    assert np.isnan(out.chi_pe.values[0]) and np.isnan(out.eps_chi.values[0])
    assert out.flag.values[1] & FLAG_N2
