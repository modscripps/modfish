import numpy as np
import pytest

from modfish.chi.batchelor import FractionTable, band_fraction, spectrum

NU, D, Q = 1.3e-6, 1.4e-7, 3.7


@pytest.mark.parametrize("eps", [1e-10, 1e-8, 1e-6])
def test_spectrum_integrates_to_chi_over_6D(eps):
    # the check that caught the review's erf bug; erf gives 0.549 of this
    chi = 1e-9
    eta = (NU**3 / eps) ** 0.25
    k = np.linspace(0.0, 30 / eta, 200001)
    P = spectrum(k, eps, chi, NU, D, Q)
    assert np.trapezoid(P, k) == pytest.approx(chi / (6 * D), rel=1e-3)


def test_spectrum_accepts_scalar_k():
    # spectrum should handle array_like k, including scalars
    chi = 1e-9
    eps = 1e-8
    k_scalar = 5.0
    P_scalar = spectrum(k_scalar, eps, chi, NU, D, Q)
    # result should be finite and non-negative
    assert np.isfinite(P_scalar)
    assert P_scalar >= 0.0
    # scalar result should match array result at same k
    P_array = spectrum(np.array([k_scalar]), eps, chi, NU, D, Q)
    assert P_scalar == pytest.approx(P_array[0])


@pytest.mark.parametrize(
    "eps, expected",
    [(1e-10, 0.498), (1e-9, 0.224), (1e-8, 0.086), (1e-7, 0.030), (1e-6, 0.010)],
)
def test_band_fraction_matches_review_table(eps, expected):
    # note section 3, stage 5 table, column "1 to 12.5 cpm"
    assert band_fraction(eps, 1.0, 12.5, NU, D, Q) == pytest.approx(expected, abs=0.0015)


def test_band_fraction_widens_with_kmax_and_falls_with_eps():
    assert band_fraction(1e-9, 1.0, 12.5, NU, D, Q) < band_fraction(1e-9, 1.0, 16.67, NU, D, Q)
    assert band_fraction(1e-9, 1.0, 12.5, NU, D, Q) > band_fraction(1e-8, 1.0, 12.5, NU, D, Q)


def test_fraction_table_reproduces_direct_integration():
    tab = FractionTable.build(kmin=1.0, kmax_max=16.67, nu=NU, D=D, q=Q)
    assert tab.r.shape == (tab.eps_grid.size, tab.kmax_grid.size)
    col = tab.r_of(12.5)
    i = np.searchsorted(tab.eps_grid, 1e-8)
    assert col[i] == pytest.approx(band_fraction(tab.eps_grid[i], 1.0, 12.5, NU, D, Q), rel=1e-3)
    # interpolated column between grid points
    mid = 0.5 * (tab.kmax_grid[3] + tab.kmax_grid[4])
    assert np.all(tab.r_of(mid) <= tab.r_of(tab.kmax_grid[4]))
    assert np.all(tab.r_of(mid) >= tab.r_of(tab.kmax_grid[3]))


def test_fraction_table_monotone_in_eps():
    tab = FractionTable.build(kmin=1.0, kmax_max=12.5, nu=NU, D=D, q=Q)
    col = tab.r_of(12.5)
    # At eps = 1e-12 about 6 % of the variance sits below kmin = 1 cpm and under 1 %
    # above 12.5 cpm, so r peaks one grid step in (0.9351, 0.9368, 0.9363, ...) and
    # falls monotonically from there. g(eps) = 2 gamma eps r stays strictly increasing.
    i = int(col.argmax())
    assert i <= 2
    assert np.all(np.diff(col[i:]) <= 0)
    assert col[0] > 0.9
