import numpy as np
import pytest
from numpy.fft import irfft, rfftfreq

from modfish.chi.batchelor import band_fraction, spectrum
from modfish.chi.config import FLAG_EMPTY, FLAG_NOISE, FLAG_RAIL, FLAG_SLOW, ChiParams
from modfish.chi.noise import NoiseFloor
from modfish.chi.response import antialias, derivative, preemphasis_inverse
from modfish.chi.spectra import (
    correct_spectrum,
    dtdc,
    integrate,
    run_range,
    spectral_factor,
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


def test_spectral_factor_reproduces_correct_spectrum():
    f = np.array([2.0, 10.0, 30.0])
    Pf = np.array([1.0, 2.0, 3.0])
    spd = 3.0
    k, Pk = correct_spectrum(f, Pf, FS, spd, P)
    assert Pk == pytest.approx(Pf * spectral_factor(f, FS, spd, P))


def test_integrate_counts_bins_strictly_inside():
    k = np.arange(0.0, 20.0, 0.5)
    Pk = np.ones(k.size)
    chi, chi_noise, n, k_hi = integrate(k, Pk, None, 1.0, 12.5, dtdc_val=10.0, D=1.4e-7)
    assert n == int(((k > 1.0) & (k < 12.5)).sum())
    assert chi == pytest.approx(6 * 1.4e-7 * 100.0 * n * 0.5)
    assert k_hi == pytest.approx(12.0 + 0.25)  # last bin 12.0, half a bin above it
    _, _, n0, k0 = integrate(k, Pk, None, 1.0, 1.2, dtdc_val=10.0, D=1.4e-7)
    assert n0 == 0 and np.isnan(k0)


def test_integrate_subtracts_the_floor_inside_the_band():
    k = np.arange(0.0, 20.0, 0.5)
    Pk = np.full(k.size, 3.0)
    Nk = np.full(k.size, 1.0)
    chi, chi_noise, n, k_hi = integrate(k, Pk, Nk, 1.0, 12.5, dtdc_val=10.0, D=1.4e-7)
    c = 6 * 1.4e-7 * 100.0 * 0.5
    assert n == int(((k > 1.0) & (k < 12.5)).sum())
    assert chi == pytest.approx(c * 2.0 * n)
    assert chi_noise == pytest.approx(c * 1.0 * n)
    assert k_hi == pytest.approx(12.25)


def test_integrate_with_no_floor_matches_a_zero_floor():
    k = np.arange(0.0, 20.0, 0.5)
    Pk = np.full(k.size, 3.0)
    a = integrate(k, Pk, None, 1.0, 12.5, dtdc_val=10.0, D=1.4e-7)
    b = integrate(k, Pk, np.zeros(k.size), 1.0, 12.5, dtdc_val=10.0, D=1.4e-7)
    assert a[0] == pytest.approx(b[0]) and a[1] == 0.0


def test_integrate_never_clips_negative_bins():
    """A window at the floor must average to zero across realizations.

    Clipping per bin rectifies the estimator scatter and biases chi high
    exactly where the correction matters most.
    """
    rng = np.random.default_rng(7)
    k = np.arange(0.0, 20.0, 0.5)
    Nk = np.full(k.size, 1.0)
    nu = 11.04
    chis = []
    for _ in range(4000):
        Pk = Nk * rng.chisquare(nu, k.size) / nu
        chis.append(integrate(k, Pk, Nk, 1.0, 12.5, dtdc_val=10.0, D=1.4e-7)[0])
    chis = np.array(chis)
    scale = 6 * 1.4e-7 * 100.0 * 0.5 * int(((k > 1.0) & (k < 12.5)).sum())
    assert abs(chis.mean()) < 0.05 * scale, "band sum is biased; is a clip present?"
    assert (chis < 0).mean() == pytest.approx(0.5, abs=0.1)


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
    params = ChiParams(enabled=True, gain=gain)
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
    nf = NoiseFloor.from_builtin("fctd_2026")
    at_floor = rng.normal(1.5, np.sqrt(2.4e-10 * FS / 2), n)
    out = run_range(at_floor, FS, spd, dt_dc, P, noise=nf)
    assert np.isfinite(out["chi"]).all(), "the noise path must never produce NaN"
    assert np.all(out["flag"] & FLAG_NOISE), "a window at the floor is noise-dominated"
    assert (out["chi"] < 0).any(), "at the floor, scatter must send some windows negative"
    assert np.nanmedian(out["phi"]) > 0.5
    out = run_range(x, FS, spd, np.full(nwin, np.nan), P)
    assert np.isnan(out["chi"]).all() and np.all(out["flag"] & 64)


def test_run_range_checks_the_per_window_lengths():
    """A mismatched environment array is a caller bug, not a silent
    IndexError partway through the window loop."""
    rng = np.random.default_rng(4)
    n = int(10 * FS)
    x = rng.normal(1.5, 1e-3, n)
    starts, _ = window_slices(n, FS, P)
    nwin = starts.size
    spd = np.full(nwin, 3.0)
    dt_dc = np.full(nwin, 10.0)
    with pytest.raises(ValueError, match="one entry per window"):
        run_range(x, FS, spd[:-1], dt_dc, P)
    with pytest.raises(ValueError, match="one entry per window"):
        run_range(x, FS, spd, dt_dc[:-1], P)


def test_run_range_kmax_no_longer_depends_on_the_spectrum():
    rng = np.random.default_rng(11)
    n = int(10 * FS)
    loud = rng.normal(1.5, 1e-3, n)
    quiet = rng.normal(1.5, 1e-7, n)
    starts, _ = window_slices(n, FS, P)
    spd = np.full(starts.size, 3.0)
    dt_dc = np.full(starts.size, 10.0)
    nf = NoiseFloor.from_builtin("fctd_2026")
    a = run_range(loud, FS, spd, dt_dc, P, noise=nf)
    b = run_range(quiet, FS, spd, dt_dc, P, noise=nf)
    assert a["kmax"] == pytest.approx(b["kmax"], nan_ok=True)


def test_run_range_phi_brackets():
    rng = np.random.default_rng(12)
    n = int(10 * FS)
    starts, _ = window_slices(n, FS, P)
    spd = np.full(starts.size, 3.0)
    dt_dc = np.full(starts.size, 10.0)
    nf = NoiseFloor.from_builtin("fctd_2026")
    loud = rng.normal(1.5, 1e-3, n)
    out = run_range(loud, FS, spd, dt_dc, P, noise=nf)
    assert np.nanmedian(out["phi"]) < 0.05, "signal far above the floor"
    off = run_range(loud, FS, spd, dt_dc, P, noise=None)
    assert np.all(off["phi"] == 0.0)
    assert not np.any(off["flag"] & FLAG_NOISE)


def test_run_range_diagnostic_subsample():
    rng = np.random.default_rng(13)
    n = int(20 * FS)
    x = rng.normal(1.5, 1e-3, n)
    starts, _ = window_slices(n, FS, P)
    spd = np.full(starts.size, 3.0)
    dt_dc = np.full(starts.size, 10.0)
    out = run_range(x, FS, spd, dt_dc, P, noise=None, diag_stride=7)
    assert out["diag_Pf"].shape[0] == out["diag_idx"].size
    assert out["diag_Pf"].shape[1] == out["diag_f"].size
    assert np.all(np.diff(out["diag_idx"]) == 7)
    off = run_range(x, FS, spd, dt_dc, P, noise=None, diag_stride=0)
    assert off["diag_idx"].size == 0 and off["diag_Pf"].size == 0
