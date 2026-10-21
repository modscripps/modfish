import numpy as np
import pytest

from modfish.chi.batchelor import BandResponse, band_fraction, spectrum, welch_kernels
from modfish.chi.config import ChiParams
from modfish.chi.spectra import band_kmax, integrate, spectral_factor, window_spectrum

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


FS = 325.52


def test_welch_kernels_pass_white_noise_above_the_lowest_bin():
    # a white spectrum comes back at its level, except bin 1, where the
    # per-segment linear detrend removes about a quarter of the power
    f, f_bins, K = welch_kernels(FS, 0.5)
    df = f[1] - f[0]
    gain = K.sum(axis=1) * df
    assert gain[1] == pytest.approx(0.73, abs=0.01)
    band = f_bins < 50.0  # fmax_cap
    assert np.allclose(gain[2:][band[2:]], 1.0, atol=2e-3)
    assert f_bins[1] == pytest.approx(FS / round(0.5 * FS))


@pytest.fixture(scope="module")
def response():
    return BandResponse.build(ChiParams(gain=1.0), FS, spd_max=2.0)


@pytest.mark.parametrize("spd", [0.5, 0.61, 0.75, 1.0, 1.25, FS / 163, 1.9973])
def test_band_response_sums_the_bins_integrate_sums(response, spd):
    params = ChiParams(gain=1.0)
    f, _ = window_spectrum(np.zeros(4 * round(0.5 * FS)), FS, params.nsec)
    k = f / spd
    expected = np.flatnonzero((k > params.kmin) & (k < band_kmax(spd, params)))
    assert np.array_equal(response.band_bins(spd, FS), expected)


@pytest.mark.parametrize("spd", [0.5, 1.0, 1.9])
def test_band_response_has_one_peak_and_invertible_closure(response, spd):
    # r rises while the Batchelor peak moves into the band and falls once
    # it moves past kmax; solve_epsilon needs eps * r strictly increasing
    col = response.r_of(spd, FS)
    i = int(col.argmax())
    assert np.all(np.diff(col[: i + 1]) >= 0)
    assert np.all(np.diff(col[i:]) <= 0)
    assert np.all(np.diff(response.eps_grid * col) > 0)
    assert np.all((col > 0) & (col < 1))


def test_band_response_interpolates_in_spd(response):
    # an off-grid speed against a table built on a grid through it
    exact = BandResponse.build(ChiParams(gain=1.0), FS, spd_max=2.0, spd_min=0.913)
    assert response.r_of(0.913, FS) == pytest.approx(exact.r_of(0.913, FS), rel=1e-3)


@pytest.mark.parametrize("ieps, spd", [(20, 0.5), (40, 0.75), (60, 1.6)])
def test_band_response_matches_the_welch_band_sum(response, ieps, spd):
    """`r` must be the fraction of chi the band sum of a Welch estimate
    recovers. Synthetic c1 with the Batchelor spectrum pushed back through
    the acquisition factor, then `window_spectrum` and `integrate`, as
    `run_range` does. The exact Batchelor integral over the band was 10 %
    above the band sum at eps 1e-10 and 0.5 m/s."""
    params = ChiParams(gain=1.0)
    eps = response.eps_grid[ieps]
    n = 2**21
    rng = np.random.default_rng(ieps)
    fn = np.fft.rfftfreq(n, 1 / FS)[1:]
    S = np.zeros(n // 2 + 1)
    S[1:] = (6 * params.D * spectrum(fn / spd, eps, 1.0, params.nu, params.D, params.q)
             / spectral_factor(fn, FS, spd, params))
    X = np.sqrt(S * FS * n / 4) * (rng.standard_normal(S.size)
                                     + 1j * rng.standard_normal(S.size))
    x = np.fft.irfft(X, n)
    f, Pf = window_spectrum(x, FS, params.nsec)
    Pk = Pf.copy()
    Pk[1:] *= spectral_factor(f[1:], FS, spd, params)
    r_act = integrate(f / spd, Pk, None, params.kmin, band_kmax(spd, params),
                      1.0, 1 / 6)[0]
    assert response.r_of(spd, FS)[ieps] == pytest.approx(r_act, rel=0.015)
