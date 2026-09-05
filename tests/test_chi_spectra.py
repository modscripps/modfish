import numpy as np
import pytest
from numpy.fft import irfft, rfftfreq

from modfish.chi.batchelor import band_fraction, spectrum
from modfish.chi.config import FLAG_EMPTY, FLAG_NOISE, FLAG_RAIL, FLAG_SLOW, ChiParams
from modfish.chi.response import antialias, derivative, preemphasis_inverse
from modfish.chi.spectra import (
    correct_spectrum,
    dtdc,
    integrate,
    noise_kmax,
    run_range,
    window_slices,
    window_spectrum,
)

FS = 325.52
P = ChiParams(enabled=True, gain=50.0)


def test_dtdc_near_sbe_linearization():
    # Matlab dCdT_SBE = 0.1 (1 + 0.006 (T - 20)) S/m per K; gsw within 3 %
    for t in (5.0, 20.0, 28.0):
        expected = 1.0 / (0.1 * (1 + 0.006 * (t - 20.0)))
        assert dtdc(35.0, t, 100.0) == pytest.approx(expected, rel=0.03)


def test_window_spectrum_white_noise_level():
    rng = np.random.default_rng(1)
    level = 2e-9  # V^2/Hz one-sided
    x = rng.normal(0.0, np.sqrt(level * FS / 2), int(60 * FS))
    f, Pf = window_spectrum(x, FS, nsec=0.5)
    assert f[0] == 0.0 and f[-1] == pytest.approx(FS / 2, rel=1e-2)
    assert Pf[2:-1].mean() == pytest.approx(level, rel=0.05)


def test_correct_spectrum_closed_form():
    f = np.array([3.0, 10.0, 37.5])
    Pf = np.ones(3)
    spd = 3.0
    k, Pk = correct_spectrum(f, Pf, FS, spd, P)
    assert k == pytest.approx(f / spd)
    expected = (spd * P.gain**2 * derivative(f / spd)
                * preemphasis_inverse(f, P.R24, P.R25, P.R22, P.C19)
                / antialias(f, FS, "som_sinc4"))
    assert Pk == pytest.approx(expected)


def test_noise_kmax_and_flags():
    f = np.linspace(0, FS / 2, 82)
    spd = 3.0
    Pf = np.full(f.size, 10 * P.snr * P.noise_floor)
    assert noise_kmax(f, Pf, spd, P) == np.inf
    Pf[f > 30.0] = 0.5 * P.noise_floor  # drops below 3x floor above 30 Hz = 10 cpm
    assert noise_kmax(f, Pf, spd, P) == pytest.approx(10.0, abs=f[1] / spd)
    assert noise_kmax(f, Pf, spd, ChiParams(enabled=True, gain=1.0, snr=0.0)) == np.inf


def test_integrate_counts_bins_strictly_inside():
    k = np.arange(0.0, 20.0, 0.5)
    Pk = np.ones(k.size)
    chi, n, k_hi = integrate(k, Pk, 1.0, 12.5, dtdc_val=10.0, D=1.4e-7)
    assert n == int(((k > 1.0) & (k < 12.5)).sum())
    assert chi == pytest.approx(6 * 1.4e-7 * 100.0 * n * 0.5)
    assert k_hi == pytest.approx(12.0 + 0.25)  # last bin 12.0, half a bin above it
    _, n0, k0 = integrate(k, Pk, 1.0, 1.2, dtdc_val=10.0, D=1.4e-7)
    assert n0 == 0 and np.isnan(k0)


def test_window_slices():
    starts, centers = window_slices(int(10 * FS), FS, P)
    nw, ns = round(P.window * FS), round(P.step * FS)
    assert starts[0] == 0 and np.all(np.diff(starts) == ns)
    assert starts[-1] + nw <= int(10 * FS)
    assert centers[0] == pytest.approx(nw / 2 / FS)


def _synthetic_volts(eps, chi, spd, gain, seconds, params, seed=2):
    """Raw volts whose corrected gradient spectrum is Batchelor(eps, chi)
    in temperature-gradient units, built by inverting the correction chain
    in the Fourier domain."""
    rng = np.random.default_rng(seed)
    n = int(seconds * FS)
    f = rfftfreq(n, 1 / FS)
    k = f / spd
    dt_dc = 10.0
    Pk_target = spectrum(k, eps, chi, params.nu, params.D, params.q) / dt_dc**2  # (S/m)^2/m^2 per cpm
    factor = (spd * gain**2 * derivative(k)
              * preemphasis_inverse(f, params.R24, params.R25, params.R22, params.C19)
              / antialias(f, FS, params.antialias))
    with np.errstate(divide="ignore", invalid="ignore"):
        Pf = np.where(factor > 0, Pk_target / factor, 0.0)  # V^2/Hz
    Pf[0] = 0.0
    amp = np.sqrt(Pf * FS * n / 2)
    phase = np.exp(2j * np.pi * rng.random(f.size))
    x = irfft(amp * phase, n=n)
    return x + 1.5, dt_dc


def test_chain_recovers_batchelor_chi():
    eps, chi, spd, gain = 1e-8, 1e-9, 3.0, 50.0
    params = ChiParams(enabled=True, gain=gain, snr=0.0)
    x, dt_dc = _synthetic_volts(eps, chi, spd, gain, seconds=120.0, params=params)
    starts, _ = window_slices(x.size, FS, params)
    nwin = starts.size
    out = run_range(x, FS, np.full(nwin, spd), np.full(nwin, dt_dc), params)
    cap = min(params.kmax_cap, params.fmax_cap / spd)
    # Expected on the estimator's own bins (rectangle rule over the interior bins of the
    # 0.5 s segment, dk = 0.666 cpm at 3 m/s): the continuous band fraction over [1, 12.5]
    # is 2.6 % higher because the sum stops half a bin below the cap. The chain itself
    # recovers the discrete expectation to 0.3 %.
    f_bins = np.fft.rfftfreq(int(round(params.nsec * FS)), 1 / FS)
    k_bins = f_bins / spd
    sel = (k_bins > params.kmin) & (k_bins < cap)
    dk = k_bins[1] - k_bins[0]
    expected = 6 * params.D * spectrum(k_bins, eps, chi, params.nu, params.D, params.q)[sel].sum() * dk
    assert np.isfinite(out["chi"]).all()
    assert np.median(out["chi"]) == pytest.approx(expected, rel=0.03)
    # kmax reports the band the sum covered: the upper edge of the last bin
    assert np.all(out["kmax"] == pytest.approx(k_bins[sel][-1] + dk / 2))
    assert np.all(out["kmax"] < cap) and np.all(out["kmax"] > cap - dk)
    assert np.all(out["flag"] == 0)
    # the closure's r must be evaluated over that same band: check the two agree to 1 %
    assert band_fraction(eps, params.kmin, float(out["kmax"][0]), params.nu, params.D, params.q) * chi == pytest.approx(expected, rel=0.01)


def test_run_range_flags():
    rng = np.random.default_rng(3)
    n = int(10 * FS)
    x = rng.normal(1.5, 1e-3, n)
    starts, _ = window_slices(n, FS, P)
    nwin = starts.size
    spd = np.full(nwin, 3.0)
    dt_dc = np.full(nwin, 10.0)
    out = run_range(x, FS, spd, dt_dc, P)
    assert out["flag"].dtype == np.uint8
    slow = spd.copy(); slow[0] = 0.1
    out = run_range(x, FS, slow, dt_dc, P)
    assert out["flag"][0] & FLAG_SLOW and np.isnan(out["chi"][0])
    railed = x.copy(); railed[:50] = 2.5
    out = run_range(railed, FS, spd, dt_dc, P)
    assert out["flag"][0] & FLAG_RAIL and not (out["flag"][-1] & FLAG_RAIL)
    quiet = rng.normal(1.5, np.sqrt(0.2 * P.noise_floor * FS / 2), n)  # below the floor
    out = run_range(quiet, FS, spd, dt_dc, P)
    assert np.all(out["flag"] & FLAG_EMPTY) and np.isnan(out["chi"]).all()
    assert np.all(out["flag"] & FLAG_NOISE)  # the cut fired below both caps
    out = run_range(x, FS, spd, np.full(nwin, np.nan), P)
    assert np.isnan(out["chi"]).all() and np.all(out["flag"] & 64)
