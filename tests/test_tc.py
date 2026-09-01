import numpy as np
import xarray as xr
import pytest

from modfish import tc


def make_synthetic_ctd(tau=0.06, lag=0.08, fs=16.0, minutes=40, seed=7):
    """t is a first-order-lagged, delayed copy of the signal driving c.

    phase_correct's model corrects t by H = (1 + i 2 pi f tau) exp(i 2 pi f L),
    so build t_meas from the clean signal s with the inverse of H and expect
    (tau, L) back from the fit.

    The default lag is 0.08 s rather than a larger value because the fit only
    works while the cross-spectral phase stays inside 2 pi over the resolved
    band. phase_correct folds the phase by 2 pi once, which repairs the wrap
    past -pi but not the wrap past -2 pi. At fs = 16 Hz the highest resolved
    frequency is just under 8 Hz, where the phase is
    -arctan(2 pi f tau) - 2 pi f L, so the fold covers lags up to about 0.10 s.
    `test_phase_correct_fit_breaks_when_phase_wraps_past_2pi` pins that limit.
    """
    rng = np.random.default_rng(seed)
    n = int(fs * 60 * minutes)
    time = np.datetime64("2024-11-20") + (np.arange(n) / fs * 1e9).astype(
        "timedelta64[ns]"
    )
    s = rng.standard_normal(n).cumsum()          # red-noise signal
    s = (s - s.mean()) / s.std()
    f = np.fft.rfftfreq(n, d=1 / fs)
    H_inv = np.exp(-1j * 2 * np.pi * f * lag) / (1 + 1j * 2 * np.pi * f * tau)
    t_meas = np.fft.irfft(np.fft.rfft(s) * H_inv, n)
    p = np.linspace(0, 600, n)                   # one long downcast
    ds = xr.Dataset(
        coords=dict(time=("time", time)),
        data_vars=dict(
            t=("time", 10 + 0.5 * t_meas),
            c=("time", 3.5 + 0.05 * s),
            p=("time", p),
            lon=("time", np.full(n, -140.0)),
            lat=("time", np.full(n, 2.0)),
            dPdt=("time", np.gradient(p) * fs),
        ),
    )
    return ds


def test_add_tcfit_default_600dbar_record_starts_at_200():
    # thresholds: p.max() > 1000 -> 500, > 300 -> 200, else 50
    ds = make_synthetic_ctd()                # p reaches 600 dbar
    ds = tc.add_tcfit_default(ds)
    assert ds.attrs["tcfit"][0] == 200
    assert ds.attrs["tcfit"][1] == pytest.approx(600, abs=1)


def test_phase_correct_recovers_tau_and_lag():
    ds = make_synthetic_ctd(tau=0.06, lag=0.08)
    out = tc.phase_correct(ds, tcfit=(100, 600))
    assert out.attrs["tau1"] == pytest.approx(0.06, rel=0.3)
    # verified convention: correction multiplies t's spectrum by
    # exp(+i 2 pi f L1), which advances t by L1; compensating a physical
    # delay of t behind c therefore fits L1 = +lag
    assert out.attrs["L1"] == pytest.approx(0.08, abs=0.05)


def test_phase_correct_fit_breaks_when_phase_wraps_past_2pi():
    """Documented limitation, not a porting defect.

    The single 2 pi fold in phase_correct cannot repair a phase that wraps
    past -2 pi. At fs = 16 Hz a lag of 0.125 s drives the phase past -2 pi
    above about 6.3 Hz, and on synthetic data whose coherence stays near one
    all the way to Nyquist those wrapped points carry full weight and pull the
    fit away from the truth. Real records lose coherence well before Nyquist,
    which is why the fold suffices there. Verified to reproduce gvpy.mod
    bit-for-bit on this input.
    """
    ds = make_synthetic_ctd(tau=0.06, lag=0.125)
    out, spec = tc.phase_correct(ds, tcfit=(100, 600), return_spectra=True)
    # the fold leaves everything past -2 pi short by exactly one turn
    f = spec.f.data
    true_phase = -np.arctan(2 * np.pi * f * 0.06) - 2 * np.pi * f * 0.125
    wrapped = true_phase < -2 * np.pi
    assert wrapped.sum() > 5
    assert np.allclose(
        spec.phase.data[wrapped], true_phase[wrapped] + 2 * np.pi, atol=0.05
    )
    assert np.allclose(spec.phase.data[~wrapped], true_phase[~wrapped], atol=0.05)
    # and the fit consequently misses badly
    assert out.attrs["tau1"] > 0.3

    # restricted to the band where the fold does its job, the same cost
    # function recovers the truth, which is what shows the model and the sign
    # convention are right
    from scipy import optimize

    ok = f < 6.2
    x = optimize.fmin(
        func=tc.atanfit,
        x0=[0, 0],
        args=(f[ok], spec.phase.data[ok], np.diag(spec.coh.data[ok])),
        disp=False,
    )
    assert x[0] == pytest.approx(0.06, rel=0.3)
    assert x[1] == pytest.approx(0.125, abs=0.05)


def test_phase_correct_carries_dpdt_and_stamps_attrs():
    ds = make_synthetic_ctd()
    out = tc.phase_correct(ds, tcfit=(100, 600))
    assert "dPdt" in out
    assert {"tau1", "L1"} <= set(out.attrs)


def test_phase_correct_uses_tcfit_attr_when_no_argument():
    ds = make_synthetic_ctd()
    ds.attrs["tcfit"] = (100, 600)
    out = tc.phase_correct(ds)
    assert out.attrs["tau1"] == pytest.approx(0.06, rel=0.3)


def test_phase_correct_falls_back_to_default_tcfit():
    ds = make_synthetic_ctd()
    out = tc.phase_correct(ds)
    # add_tcfit_default was used internally without touching the caller's ds
    assert "tcfit" not in ds.attrs
    assert out.attrs["tcfit"][0] == 200


def test_phase_correct_reads_sampling_interval_from_time_axis():
    """8 Hz data must not be treated as 16 Hz."""
    ds = make_synthetic_ctd(tau=0.06, lag=0.08, fs=8.0, minutes=80)
    out = tc.phase_correct(ds, tcfit=(100, 600))
    assert out.attrs["L1"] == pytest.approx(0.08, abs=0.05)


def test_phase_correct_output_length_is_span_trimmed_by_half_segment():
    ds = make_synthetic_ctd()
    N = 128
    out = tc.phase_correct(ds, N=N, tcfit=(100, 600))
    ii = np.squeeze(np.argwhere(ds.p.data > 1))
    n = int(np.floor((ii[-1] - ii[0] + 1) / N) * N)
    assert out.time.size == n - N // 2


def test_phase_correct_improves_high_freq_coherence_phase():
    ds = make_synthetic_ctd(tau=0.06, lag=0.08)
    out, spec = tc.phase_correct(ds, tcfit=(100, 600), return_spectra=True)
    f_hi = spec.f > 2.0
    assert np.abs(spec.phase_corrected[f_hi]).mean() < np.abs(
        spec.phase[f_hi]
    ).mean()


def test_plot_spectra_draws_four_panels():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ds = make_synthetic_ctd()
    _, spec = tc.phase_correct(ds, tcfit=(100, 600), return_spectra=True)
    ax = tc.plot_spectra(spec)
    assert ax.shape == (2, 2)
    plt.close("all")


def test_lowpassfilter_removes_high_frequency():
    fs = 16.0
    t = np.arange(0, 60, 1 / fs)
    x = np.sin(2 * np.pi * 0.1 * t) + np.sin(2 * np.pi * 6 * t)
    lp = tc.lowpassfilter(x, lowcut=1.0, fs=fs)
    resid = lp - np.sin(2 * np.pi * 0.1 * t)
    assert np.abs(resid[int(fs) : -int(fs)]).max() < 0.1
