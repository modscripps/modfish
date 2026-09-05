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
    assert clipped[0] and r[0] == 1.0 and eps[0] == table.eps_grid[0]
    assert np.isnan(eps[1:]).all() and np.isnan(r[1:]).all()


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
        P.g * strat.alpha.values / P.rho_0 * chi / strat.n2.values * P.rrho_factor_max))
    # tiny chi_pe: r clipped
    out = closure(np.full(centers.size, 1e-20), kmax, strat, P, table)
    assert np.all(out.flag.values & FLAG_RCLIP) and np.all(out.r.values == 1.0)
