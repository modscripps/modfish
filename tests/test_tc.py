import gsw
import numpy as np
import pytest
import xarray as xr

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
    # Pins that the attr path actually ran (not the argument path, and not
    # add_tcfit_default silently overriding it): ds.attrs["tcfit"] comes
    # through onto the output unchanged, exact tuple.
    assert out.attrs["tcfit"] == (100, 600)


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


def test_thermal_mass_correction_matches_reference_recursion():
    ds = make_synthetic_ctd().isel(time=slice(0, 64))
    alpha, beta = 0.03, 1 / 7
    fn = 8.0  # Nyquist of the 16 Hz synthetic record
    aa = 4 * fn * alpha / beta / (1 + 4 * fn / beta)
    bb = 1 - 2 * aa / alpha
    gamma = 0.1
    # gvpy mod.py:1037-1039: forward diff with dTp[0] duplicated from dTp[1]
    dTp = np.diff(ds.t.data, prepend=ds.t.data[0])
    dTp[0] = dTp[1]
    ctm = np.zeros_like(dTp)
    for i in range(1, len(ctm)):
        ctm[i] = -bb * ctm[i - 1] + aa * gamma * dTp[i]
    out = tc.thermal_mass_correction(ds, alpha=alpha, beta=beta, dcdt="constant")
    np.testing.assert_allclose(out.c.data, ds.c.data + ctm, rtol=1e-10)


def test_thermal_mass_correction_does_not_mutate_input():
    ds = make_synthetic_ctd().isel(time=slice(0, 64))
    before = ds.c.data.copy()
    tc.thermal_mass_correction(ds)
    np.testing.assert_array_equal(ds.c.data, before)


def test_thermal_mass_correction_alpha_zero_raises_value_error():
    # bb = 1 - 2 * aa / alpha divides by alpha, so alpha = 0 used to raise a
    # bare ZeroDivisionError from deep inside the recursion setup. The guard
    # turns it into a ValueError naming the parameter and the way to skip the
    # step.
    ds = make_synthetic_ctd().isel(time=slice(0, 64))
    with pytest.raises(ValueError, match="alpha"):
        tc.thermal_mass_correction(ds, alpha=0.0)


def test_find_lags_recovers_known_lag():
    ds = make_synthetic_ctd(tau=0.0, lag=4 / 16.0)  # pure 4-sample lag
    out = tc.find_lags(ds)
    assert float(out.lag.median()) == pytest.approx(4 / 16.0, abs=1 / 16.0)
    assert out.dPdt.sizes["segment"] == out.lag.sizes["segment"]


def _clean_signal(ds):
    # make_synthetic_ctd: c = 3.5 + 0.05 * s, so s = (c - 3.5) / 0.05
    return (ds.c.data - 3.5) / 0.05


def test_response_correction_pure_lag_shifts_impulse_and_leaves_trailing_nan():
    ds = make_synthetic_ctd().isel(time=slice(0, 256))
    imp = np.zeros(256); imp[100] = 1.0
    ds["t"] = ("time", imp)
    out = tc.response_correction(ds, lag=4 / 16.0, tau_t=0.0)
    assert np.argmax(np.nan_to_num(out.t.data)) == 96
    assert np.isnan(out.t.data[-4:]).all()
    assert np.isfinite(out.t.data[:-4]).all()


def test_response_correction_negative_lag_raises():
    ds = make_synthetic_ctd().isel(time=slice(0, 64))
    with pytest.raises(ValueError):
        tc.response_correction(ds, lag=-0.05, tau_t=0.0)


def test_response_correction_restores_interior_nan_only():
    ds = make_synthetic_ctd().isel(time=slice(0, 512))
    ds["t"][200:210] = np.nan
    out = tc.response_correction(ds, lag=0.08, tau_t=0.06)
    assert np.isnan(out.t.data[200:210]).all()
    assert np.isfinite(out.t.data[:200]).all()
    assert np.isfinite(out.t.data[210:-2]).all()


def test_response_correction_linear_ramp_is_shifted_exactly():
    # A linear ramp has no discontinuity to wrap-around ring on its own, so
    # this isolates the endpoint-line bookkeeping (subtract before the FFT,
    # add back analytically) from the ringing the next test targets: the
    # output must equal the advanced/scaled line to near machine precision.
    ds = make_synthetic_ctd().isel(time=slice(0, 512))
    fs = 16.0
    dt = 1 / fs
    n = ds.sizes["time"]
    t_idx = np.arange(n) * dt
    a, b = 3.0, 0.7
    ds["t"] = ("time", a + b * t_idx)

    lag = 1.3 * dt  # "1.3 samples"
    tau_t = 0.05
    out = tc.response_correction(ds, lag=lag, tau_t=tau_t)

    expected = a + b * (t_idx + lag) + tau_t * b
    ntrail = int(np.ceil(lag * fs))
    np.testing.assert_allclose(
        out.t.data[:-ntrail], expected[:-ntrail], atol=1e-9
    )


def test_response_correction_step_between_ends_does_not_ring():
    # trend is centered on 0 (-7.5..7.5), not 0..15: an off-center trend
    # (tried first) shifts the record's mean temperature enough that SP's
    # own nonlinear dependence on T changes the roughness by itself, even
    # after the fix (measured post-fix ratio 0.799 there, outside the 10%
    # tolerance below), confounding that nonlinearity with the wrap defect
    # this test targets. Centering the trend on 0 avoids the confound.
    #
    # Measured 2026-09-02: before the fix, r_step=0.0080219, r_detrend=
    # 0.0045184, ratio 1.775; after the fix, r_step=0.0044993, r_detrend=
    # 0.0045112, ratio 0.997. 10 % tolerance (1.1) separates them.
    def roughness(ds_in):
        out = tc.correct(ds_in, lag=0.03, tau_t=0.03, lowpass=4.0)
        SP = gsw.SP_from_C(10 * out.c.data, out.t.data, out.p.data)
        out = out.assign(SP=("time", SP))
        return tc.salinity_roughness(out, 50, 600)

    ds = make_synthetic_ctd()
    trend = np.linspace(-7.5, 7.5, ds.sizes["time"])

    ds_step = ds.copy(deep=True)
    ds_step["t"] = ds_step["t"] + trend

    r_step = roughness(ds_step)
    r_detrend = roughness(ds.copy(deep=True))

    assert r_step == pytest.approx(r_detrend, rel=0.1)


def test_correct_true_parameters_recover_clean_signal():
    # Review check 2026-09-02 measured 0.013 for the FFT application and
    # 0.089 for a central-difference time-domain version; 0.1 separates them.
    tau, lag = 0.06, 0.08
    ds = make_synthetic_ctd(tau=tau, lag=lag)
    out = tc.correct(ds, lag=lag, tau_t=tau, lowpass=4.0)
    s = _clean_signal(ds)
    s_lp = tc.lowpassfilter(s, lowcut=4.0, fs=16.0)
    recovered = (out.t.data - 10.0) / 0.5
    resid = np.nanstd(recovered[200:-200] - s_lp[200:-200])
    resid_raw = np.std(((ds.t.data - 10.0) / 0.5)[200:-200] - s_lp[200:-200])
    assert resid < 0.1 * resid_raw


def test_correct_keeps_time_axis_and_stamps_processing():
    ds = make_synthetic_ctd()
    out = tc.correct(ds, lag=0.1, tau_t=0.05, lowpass=4.0, thermal_mass=True)
    assert out.time.equals(ds.time)
    assert "response lag 0.100 s tau 0.050 s" in out.t.attrs["processing"]
    assert "thermal mass" in out.c.attrs["processing"]
    assert "lowpass 4.0 Hz" in out.c.attrs["processing"]
    assert out.attrs["corrections"] != "none"


def test_correct_defaults_are_noop():
    ds = make_synthetic_ctd()
    out = tc.correct(ds)
    np.testing.assert_array_equal(out.t.data, ds.t.data)
    np.testing.assert_array_equal(out.c.data, ds.c.data)
    assert out.t.attrs["processing"] == "none"
    assert out.attrs["corrections"] == "none"


def test_thermal_mass_sbe_dcdt_equals_constant_at_20degc():
    # Window moved from the brief's [0:64] to [300:364]: make_synthetic_ctd
    # builds t with a whole-record irfft(rfft(s) * H_inv), and s (a cumsum,
    # not periodic) leaves a Gibbs-type edge transient at the record start
    # (measured: t swings 17.4-19.9 degC over the first 4 samples even after
    # recentering to mean 20, vs +-0.03 degC by [300:364]). At the edge, the
    # sbe/constant difference is 3.5e-5 (350x the tolerance below); away from
    # it, where the window is actually close to 20 degC as intended, it is
    # 1.4e-8. This test is about the dcdt formula, not the fixture's startup
    # transient, so the window avoids it.
    ds = make_synthetic_ctd().isel(time=slice(300, 364))
    ds["t"] = ds.t - ds.t.mean() + 20.0
    a = tc.thermal_mass_correction(ds, dcdt="sbe")
    b = tc.thermal_mass_correction(ds, dcdt="constant")
    np.testing.assert_allclose(a.c.data, b.c.data, atol=1e-7)


def test_thermal_mass_nan_in_t_does_not_poison_c():
    ds = make_synthetic_ctd().isel(time=slice(0, 64))
    ds["t"][30:33] = np.nan
    out = tc.thermal_mass_correction(ds)
    assert np.isfinite(out.c.data).all()


def test_viscous_heating_formula_no_scale():
    v = np.array([0.0, 1.0, 2.0])
    np.testing.assert_allclose(
        tc.viscous_heating_temperature_correction(v, Pr=12.4),
        0.8e-4 * np.sqrt(12.4) * v**2,
    )


def test_find_lags_returns_dataset_with_pressure():
    ds = make_synthetic_ctd(tau=0.0, lag=4 / 16.0)
    out = tc.find_lags(ds, lowpass=4.0)
    assert set(out.data_vars) >= {"lag", "dPdt", "p"}
    assert float(out.lag.median()) == pytest.approx(4 / 16.0, abs=1 / 16.0)


def test_salinity_roughness_zero_for_smooth_profile():
    ds = make_synthetic_ctd()
    ds["SP"] = ("time", np.linspace(34.0, 35.0, ds.sizes["time"]))
    assert tc.salinity_roughness(ds, 50, 600) == pytest.approx(0.0, abs=1e-9)


def test_salinity_roughness_cast_boundary_and_edge_spikes_ignored_interior_counted():
    # Two labeled casts (cast id 0 outside, as label_casts produces), each
    # long enough that edge=2.0 s (32 samples at 16 Hz) trims a real chunk
    # off both ends. SP is a smooth ramp within each cast so the baseline
    # roughness is ~0; single-sample spikes are then injected at three
    # positions to check what the edge exclusion and per-cast segmentation
    # do and do not catch.
    fs = 16.0
    n_cast = 200
    gap = 20
    n = 2 * n_cast + gap
    time = np.datetime64("2024-01-01") + (np.arange(n) / fs * 1e9).astype(
        "timedelta64[ns]"
    )
    p = np.linspace(50.0, 300.0, n)
    cast = np.zeros(n, dtype=int)
    cast[:n_cast] = 1
    cast[n_cast + gap : n_cast + gap + n_cast] = 2
    sp_base = np.concatenate(
        [
            np.linspace(34.0, 34.5, n_cast),
            np.full(gap, np.nan),
            np.linspace(34.5, 35.0, n_cast),
        ]
    )

    def make_ds(sp):
        return xr.Dataset(
            coords=dict(time=("time", time), cast=("time", cast)),
            data_vars=dict(SP=("time", sp), p=("time", p)),
        )

    edge = 2.0
    nedge = int(round(edge * fs))  # 32
    assert nedge < n_cast // 2  # sanity: edge trim leaves an interior

    baseline = tc.salinity_roughness(make_ds(sp_base.copy()), 0, 1000, edge=edge)
    assert baseline == pytest.approx(0.0, abs=1e-6)

    # Spike at the last sample of cast 1: sits exactly at the cast boundary
    # and inside cast 1's own trailing edge window (local index n_cast - 1,
    # trimmed since the core is seg[nedge : n_cast - nedge]).
    sp_boundary = sp_base.copy()
    sp_boundary[n_cast - 1] += 5.0
    boundary_roughness = tc.salinity_roughness(make_ds(sp_boundary), 0, 1000, edge=edge)
    assert boundary_roughness == pytest.approx(0.0, abs=1e-6)

    # Spike well inside cast 2's own edge window (5 samples past its start,
    # nedge=32 samples are trimmed there), not at a cast boundary.
    sp_edge = sp_base.copy()
    sp_edge[n_cast + gap + 5] += 5.0
    edge_roughness = tc.salinity_roughness(make_ds(sp_edge), 0, 1000, edge=edge)
    assert edge_roughness == pytest.approx(0.0, abs=1e-6)

    # Spike in the interior of cast 2 (100 samples in, well past nedge=32
    # from either end of the 200-sample cast): must be counted.
    sp_interior = sp_base.copy()
    sp_interior[n_cast + gap + 100] += 5.0
    interior_roughness = tc.salinity_roughness(make_ds(sp_interior), 0, 1000, edge=edge)
    assert interior_roughness > 0.1


def test_lag_tau_cost_map_minimum_at_true_pair():
    # Review check 2026-09-02 on this record: minimum at lag 0.08, tau 0.07
    # on a 0.01 s grid with the record ends excluded; the 16x12=192-pair
    # grid below runs in about 2.3 s.
    tau, lag = 0.06, 0.08
    ds = make_synthetic_ctd(tau=tau, lag=lag)
    lags = np.arange(0.0, 0.16, 0.01)
    taus = np.arange(0.0, 0.12, 0.01)
    cm = tc.lag_tau_cost_map(ds, lags, taus, lowpass=4.0, pmin=50, pmax=600)
    i = cm.cost.argmin(dim=("lag", "tau_t"))
    assert float(cm.lag[i["lag"]]) == pytest.approx(lag, abs=0.015)
    assert float(cm.tau_t[i["tau_t"]]) == pytest.approx(tau, abs=0.015)


def test_lag_tau_cost_map_minimum_at_true_pair_with_end_to_end_step():
    # Same as test_lag_tau_cost_map_minimum_at_true_pair, but t carries a
    # 15 degC trend (centered on 0, see the comment in
    # test_response_correction_step_between_ends_does_not_ring for why) so
    # the record's first and last samples differ sharply (the notebook 04
    # defect: without the response_correction fix, the periodic FFT rings
    # on that step and the cost map along a lag scan turns into a sawtooth
    # with minima at whole 16 Hz samples, missing the true lag).
    #
    # Measured 2026-09-02 after the fix: minimum at lag 0.08, tau 0.07,
    # same as the unstepped test's minimum (the trend does not move it).
    tau, lag = 0.06, 0.08
    ds = make_synthetic_ctd(tau=tau, lag=lag)
    trend = np.linspace(-7.5, 7.5, ds.sizes["time"])
    ds["t"] = ds["t"] + trend

    lags = np.arange(0.0, 0.16, 0.01)
    taus = np.arange(0.0, 0.12, 0.01)
    cm = tc.lag_tau_cost_map(ds, lags, taus, lowpass=4.0, pmin=50, pmax=600)
    i = cm.cost.argmin(dim=("lag", "tau_t"))
    assert float(cm.lag[i["lag"]]) == pytest.approx(lag, abs=0.015)
    assert float(cm.tau_t[i["tau_t"]]) == pytest.approx(tau, abs=0.015)


def test_find_lags_peak_at_lag_axis_end_does_not_raise():
    # find_corrs takes range(peak - 1, peak + 2) around the raw correlation
    # peak for the quadratic refinement; when the peak sits at the last
    # index of the correlation array that runs one past the end and used to
    # raise IndexError (notebook 04, 1 of 152 real d09 casts). Constructed
    # so the very first window's correlation peaks at the last lag: c has a
    # sharp step near the end of the window, t has one near the start, so
    # after the zero-phase low-pass their cross-correlation over this short
    # window is dominated by the single almost-non-overlapping extreme lag.
    fs = 16.0
    window = 80
    n = window * 4
    time = np.datetime64("2024-01-01") + (np.arange(n) / fs * 1e9).astype(
        "timedelta64[ns]"
    )
    c = np.zeros(n)
    c[79:] = 1000.0
    t = np.zeros(n)
    t[1:] = 1000.0
    p = np.linspace(0.0, 100.0, n)
    dPdt = np.gradient(p) * fs

    ds = xr.Dataset(
        coords=dict(time=("time", time)),
        data_vars=dict(t=("time", t), c=("time", c), p=("time", p), dPdt=("time", dPdt)),
    )
    out = tc.find_lags(ds, window=window, lowpass=4.0)
    assert out.lag.sizes["segment"] > 0
    # the fallback returns the raw peak lag, here the extreme lag of the
    # correlation array: window - 2 = 78 sample steps at 16 Hz, negated by
    # the sign flip that puts the result in the "positive means t lags c"
    # convention.
    first = float(out.lag.values[0])
    assert first == pytest.approx(-(window - 2) / fs)


def make_downup_pair(minutes=4, offset_up=0.0):
    """One down cast and its exact time-reverse as the up cast.

    `offset_up` adds a constant to `c` on the up cast so the separation is
    nonzero and known in sign.
    """
    down = make_synthetic_ctd(minutes=minutes)
    n = down.sizes["time"]
    dt = down.time.data[1] - down.time.data[0]
    up_time = down.time.data[-1] + dt * (np.arange(n) + 1)
    full_time = np.concatenate([down.time.data, up_time])
    full_vars = {}
    for v in ["t", "c", "p", "lon", "lat"]:
        down_v = down[v].data
        full_vars[v] = np.concatenate([down_v, down_v[::-1]])
    full_vars["c"][n:] += offset_up
    full_vars["dPdt"] = np.gradient(full_vars["p"]) * 16.0
    ds = xr.Dataset(
        coords=dict(time=("time", full_time)),
        data_vars={k: ("time", v) for k, v in full_vars.items()},
    )
    ds = ds.assign(SP=("time", gsw.SP_from_C(10 * ds.c.data, ds.t.data, ds.p.data)))
    ds = ds.assign_coords(cast=("time", np.concatenate([np.full(n, 1), np.full(n, 2)])))
    casts = xr.Dataset(
        data_vars=dict(
            start_time=("cast", [full_time[0], full_time[n]]),
            end_time=("cast", [full_time[n - 1], full_time[-1]]),
            direction=("cast", ["down", "up"]),
        ),
        coords=dict(cast=("cast", [1, 2])),
    )
    tbins = np.linspace(ds.t.min().item(), ds.t.max().item(), 20)
    return ds, casts, tbins


def test_downup_separation_zero_for_identical_down_up():
    # two casts with mirrored p and identical T-S relation: the up cast is
    # the exact time-reverse of the down cast, so every (t, c, p) triple
    # (and therefore every (t, SP) pair) that occurs on the way down recurs
    # on the way up; binning by temperature must then give the same mean SP
    # per bin on both sides.
    ds, casts, tbins = make_downup_pair()
    out = tc.downup_separation(ds, casts, tbins, pmin=1.0)
    assert out.sep.sizes["pair"] == 1
    assert float(out.sep.isel(pair=0)) == pytest.approx(0.0, abs=1e-9)
    assert out.attrs["mean"] == pytest.approx(0.0, abs=1e-9)


def test_downup_separation_pmax_at_record_max_equals_unrestricted():
    ds, casts, tbins = make_downup_pair(offset_up=0.01)
    full = tc.downup_separation(ds, casts, tbins, pmin=1.0)
    capped = tc.downup_separation(ds, casts, tbins, pmin=1.0, pmax=float(ds.p.max()) + 1)
    assert capped.attrs["mean"] == pytest.approx(full.attrs["mean"], rel=1e-12)
    assert full.attrs["mean"] > 0


def test_downup_separation_pmax_below_pmin_gives_no_pairs():
    ds, casts, tbins = make_downup_pair(offset_up=0.01)
    out = tc.downup_separation(ds, casts, tbins, pmin=300.0, pmax=100.0)
    assert out.sizes["pair"] == 0
    assert np.isnan(out.attrs["mean"])


def test_downup_separation_pmax_restricts_to_the_shallow_band():
    # the offset is applied only below 300 dbar on the up cast; a pmax of
    # 300 must then see no separation while the unrestricted cost does
    ds, casts, tbins = make_downup_pair()
    deep = ds.p.data > 300
    c = ds.c.data.copy()
    c[deep & (ds.cast.data == 2)] += 0.01
    ds = ds.assign(c=("time", c))
    ds = ds.assign(SP=("time", gsw.SP_from_C(10 * ds.c.data, ds.t.data, ds.p.data)))
    shallow = tc.downup_separation(ds, casts, tbins, pmin=1.0, pmax=300.0)
    full = tc.downup_separation(ds, casts, tbins, pmin=1.0)
    assert shallow.attrs["mean"] == pytest.approx(0.0, abs=1e-9)
    assert full.attrs["mean"] > 1e-4


def test_rosette_rms_scales_with_offset():
    # fctd SP = ctd SP + 0.01 on the same depth grid -> rms == 0.01
    depth = np.arange(50.0, 600.0, 1.0)
    s1 = 34.0 + 0.001 * depth  # smooth reference salinity profile
    ctd = xr.Dataset(
        data_vars=dict(s1=("depth", s1)),
        coords=dict(depth=("depth", depth)),
    )
    fctd = xr.Dataset(
        data_vars=dict(SP=("time", s1 + 0.01), depth=("time", depth)),
        coords=dict(time=("time", np.arange(depth.size))),
    )
    rms = tc.rosette_rms(fctd, ctd, 50, 600)
    assert rms == pytest.approx(0.01, abs=1e-9)


def test_rosette_rms_bins_multiple_fctd_samples_per_ctd_depth():
    # fctd sampled at 16 Hz-equivalent depth spacing (16 samples per metre)
    # against a 1 m ctd grid, several fctd samples land in every ctd bin.
    # A one-period-per-metre oscillation is added on top of the offset: 16
    # equally spaced samples spanning exactly one period sum to ~0
    # regardless of phase, so the bin *mean* recovers the offset exactly
    # while any single sample (e.g. a first-sample-per-bin bug) would not.
    # This isolates the bin-averaging path itself (unlike
    # test_rosette_rms_scales_with_offset, whose constant per-bin SP cannot
    # distinguish averaging from picking one sample).
    depth_grid = np.arange(50.0, 150.0, 1.0)
    s1 = np.full(depth_grid.size, 34.0)
    ctd = xr.Dataset(
        data_vars=dict(s1=("depth", s1)),
        coords=dict(depth=("depth", depth_grid)),
    )

    offset = 0.02
    fine_depth = np.arange(49.5, 150.5, 1 / 16)
    oscillation = 0.3 * np.cos(2 * np.pi * fine_depth)
    fine_sp = 34.0 + offset + oscillation
    fctd = xr.Dataset(
        data_vars=dict(SP=("time", fine_sp), depth=("time", fine_depth)),
        coords=dict(time=("time", np.arange(fine_depth.size))),
    )

    rms = tc.rosette_rms(fctd, ctd, 49.0, 151.0)
    assert rms == pytest.approx(offset, abs=1e-6)


def test_thermal_mass_cost_map_dims_and_beta_is_inverse_tau():
    # The objective reads ds_corr.SP, not ds_corr.c: this exercises the
    # gsw.SP_from_C(...) / assign(SP=...) recompute inside the per-pair
    # function, not just the thermal-mass correction on c that a
    # c-only objective would already cover.
    ds = make_synthetic_ctd(minutes=1).isel(time=slice(0, 512))
    uncorrected_sp = gsw.SP_from_C(10 * ds.c.data, ds.t.data, ds.p.data)
    uncorrected_mean = float(np.nanmean(uncorrected_sp))

    def mean_sp(ds_corr):
        return float(np.nanmean(ds_corr.SP.data))

    alphas = np.array([0.02, 0.05])
    taus = np.array([5.0, 10.0])
    cm = tc.thermal_mass_cost_map(ds, alphas, taus, mean_sp)

    assert cm.cost.dims == ("alpha", "tau")
    assert cm.cost.shape == (alphas.size, taus.size)
    assert np.isfinite(cm.cost.data).all()

    # beta = 1/tau, not tau itself: the map's grid evaluation at one pair
    # must match a manual call with that inversion applied, computing SP
    # from the manually corrected c/t/p the same way the map itself does.
    # If the map used beta=tau instead, this would fail
    # (taus[0]=5.0 != 1/taus[0]=0.2); if it forgot to recompute SP and
    # left the objective reading stale, uncorrected SP, `actual` below
    # would equal `uncorrected_mean` instead of `expected`.
    ref = tc.thermal_mass_correction(ds, alpha=alphas[1], beta=1 / taus[0])
    ref_sp = gsw.SP_from_C(10 * ref.c.data, ref.t.data, ref.p.data)
    expected = float(np.nanmean(ref_sp))
    actual = float(cm.cost.sel(alpha=alphas[1], tau=taus[0]))
    assert actual == pytest.approx(expected)
    assert abs(actual - uncorrected_mean) > 1e-4

    # cost varies with alpha at fixed tau, and with tau at fixed alpha
    fixed_tau = cm.cost.sel(tau=taus[0]).data
    assert fixed_tau[0] != pytest.approx(fixed_tau[1])
    fixed_alpha = cm.cost.sel(alpha=alphas[0]).data
    assert fixed_alpha[0] != pytest.approx(fixed_alpha[1])


def make_gapped_pair(minutes=20, gap_start_s=600.0, gap_s=200.0, jump=5.0):
    """A synthetic record and the same record with a chunk removed and a jump across it.

    The full record is `make_synthetic_ctd` with a 5 degC offset added to `t`
    over the samples that are then removed, so `full` is continuous in time
    and the 5 degC change appears only in the gapped record, across its gap
    (the matching `c` change is left out on purpose: the test compares the
    correction of the gapped record with the correction of the full one, so
    only the chain's response to the gap matters). Returns (full, gapped,
    kept), `kept` the boolean mask of the retained samples in `full`.

    The offset is ramped over the removed interval instead of stepped at its
    end so that `full` carries no step of its own. A step there would put the
    same spike and thermal-mass transient into the reference that the fix is
    meant to remove from `out`, and the two would partly cancel: measured on
    this record, the correction of the gapped record then sat 5.4e-5 from the
    reference in `c` before the fix and 4.1e-4 after it, so the comparison
    ran backwards. `gapped` is bit-identical under either construction, since
    the samples the ramp passes through are the removed ones.
    """
    full = make_synthetic_ctd(minutes=minutes)
    fs = 16.0
    i0, i1 = int(gap_start_s * fs), int((gap_start_s + gap_s) * fs)
    t = full.t.data.copy()
    t += jump * np.clip((np.arange(t.size) - i0) / (i1 - i0), 0.0, 1.0)
    full = full.assign(t=("time", t))
    kept = np.ones(full.sizes["time"], dtype=bool)
    kept[i0:i1] = False
    return full, full.isel(time=kept), kept


def test_correct_gap_free_record_is_bit_identical():
    ds = make_synthetic_ctd(minutes=5)
    kw = dict(lag=0.03, tau_t=0.05, lowpass=5.0, thermal_mass=True, alpha=0.01, beta=1 / 12)
    out = tc.correct(ds, **kw)
    # the reference: the same call on a copy is deterministic; the point of
    # the test is that the uniform-grid path is skipped when there is no gap
    ref = tc.correct(ds.copy(deep=True), **kw)
    np.testing.assert_array_equal(out.t.data, ref.t.data)
    np.testing.assert_array_equal(out.c.data, ref.c.data)
    assert np.array_equal(out.time.data, ds.time.data)


def test_correct_gapped_record_matches_full_record_away_from_the_gap():
    full, gapped, kept = make_gapped_pair()
    kw = dict(lag=0.03, tau_t=0.05, lowpass=5.0, thermal_mass=True, alpha=0.01, beta=1 / 12)
    ref = tc.correct(full, **kw).isel(time=kept)
    out = tc.correct(gapped, **kw)
    assert np.array_equal(out.time.data, gapped.time.data)
    # exclude 60 s on either side of the gap: the ramp the fill draws across
    # the gap is not the true signal, so the recursion's state differs there.
    # 60 s is five thermal-mass relaxation times at beta = 1/12, which is what
    # it takes for that state difference to fall under the 1e-5 tolerance
    # (3.8e-5 at 30 s, 3.1e-6 at 60 s).
    fs = 16.0
    i0 = int(600.0 * fs)
    far = np.ones(out.sizes["time"], dtype=bool)
    far[i0 - int(60 * fs): i0 + int(60 * fs)] = False
    dt_ = np.abs(out.t.data - ref.t.data)[far]
    dc_ = np.abs(out.c.data - ref.c.data)[far]
    assert np.nanmax(dt_) < 1e-3
    assert np.nanmax(dc_) < 1e-5


def test_correct_gapped_record_has_no_spike_at_the_gap():
    full, gapped, kept = make_gapped_pair()
    out = tc.correct(gapped, lag=0.03, tau_t=0.05, lowpass=5.0)
    # before the fix the derivative term turned the 5 degC step into a spike
    # of several degrees; after it, the corrected t stays within the range
    # of the raw t plus a margin at every sample
    lo, hi = np.nanmin(gapped.t.data) - 0.5, np.nanmax(gapped.t.data) + 0.5
    assert np.nanmin(out.t.data) > lo
    assert np.nanmax(out.t.data) < hi


def test_response_and_thermal_mass_alone_use_the_uniform_grid():
    full, gapped, kept = make_gapped_pair()
    r_ref = tc.response_correction(full, lag=0.03, tau_t=0.05).isel(time=kept)
    r_out = tc.response_correction(gapped, lag=0.03, tau_t=0.05)
    m_ref = tc.thermal_mass_correction(full, alpha=0.01, beta=1 / 12).isel(time=kept)
    m_out = tc.thermal_mass_correction(gapped, alpha=0.01, beta=1 / 12)
    # 60 s exclusion, as in the test above
    fs = 16.0
    i0 = int(600.0 * fs)
    far = np.ones(r_out.sizes["time"], dtype=bool)
    far[i0 - int(60 * fs): i0 + int(60 * fs)] = False
    assert np.nanmax(np.abs(r_out.t.data - r_ref.t.data)[far]) < 1e-2
    assert np.nanmax(np.abs(m_out.c.data - m_ref.c.data)[far]) < 1e-5


def test_uniform_slots_rejects_two_samples_in_one_slot():
    ds = make_synthetic_ctd(minutes=1)
    time = ds.time.data.copy()
    time[10] = time[9]
    with pytest.raises(ValueError):
        tc._uniform_slots(time)


def test_uniform_slots_tolerates_stamp_drift_within_a_gap_free_stretch():
    """A record whose stamps drift against one global interval keeps its slots.

    The FCTD's 1 ms-quantized stamps run at 62 or 63 ms and their phase
    against the 62.5 ms mean drifts by more than half a sample over a
    multi-hour record. Rounding every sample's offset from `time[0]` onto one
    global grid then puts two samples in the same slot: 12700 times on the
    2025 d07 record and 16481 times on d12, both of which have every time
    step strictly positive. Counting slots along the steps instead leaves a
    gap-free stretch exactly as it was.
    """
    n = 20000
    step_ms = np.where(np.arange(n - 1) < n // 2, 62.0, 63.0)
    time = np.datetime64("2024-11-20") + np.concatenate(
        ([0.0], np.cumsum(step_ms))
    ).astype("timedelta64[ms]")

    slots, n_slots, dt = tc._uniform_slots(time)
    assert np.array_equal(slots, np.arange(n))
    assert n_slots == n

    # the same record under the global-rounding rule collides
    offset = (time - time[0]) / np.timedelta64(1, "s")
    global_slots = np.round(offset / dt).astype("int64")
    assert np.any(np.diff(global_slots) <= 0)


def test_uniform_slots_opens_only_the_missing_slots_at_a_gap():
    ds = make_synthetic_ctd(minutes=2)
    fs = 16.0
    kept = np.ones(ds.sizes["time"], dtype=bool)
    kept[320:640] = False  # 20 s of samples removed, 320 slots
    slots, n_slots, dt = tc._uniform_slots(ds.isel(time=kept).time.data)
    assert n_slots == ds.sizes["time"]
    assert np.array_equal(slots, np.flatnonzero(kept))
    assert dt == pytest.approx(1 / fs)


def test_uniform_slots_counts_a_gap_against_the_median_step():
    """The gap threshold is the one `sampling_interval` calls a regular step.

    On the FCTD's 1 ms-quantized 16 Hz stamps the median step reads 63 ms
    against a 62.5 ms mean, so a 94 ms step is regular for the interval
    estimate (under 1.5 times the median, 94.5 ms) while a threshold taken
    off the mean (1.5 times 62.5 ms, 93.75 ms) would call it a gap and open a
    second slot. The two rules have to agree, or a record grows slots the
    interval estimate says are not missing.
    """
    n = 2000
    step_ms = np.diff(np.round(np.arange(n) / 16 * 1000))
    step_ms[n // 2] = 94.0
    time = np.datetime64("2024-11-20") + np.concatenate(
        ([0.0], np.cumsum(step_ms))
    ).astype("timedelta64[ms]")

    assert np.median(step_ms) == 63.0  # the premise: median 63, mean 62.5
    slots, n_slots, dt = tc._uniform_slots(time)
    # the 94 ms step counts towards the mean interval, as it does in
    # `sampling_interval`, which pulls dt 0.03 % above 62.5 ms
    assert dt == pytest.approx(0.0625, abs=1e-4)
    assert n_slots == n
    assert np.array_equal(slots, np.arange(n))
