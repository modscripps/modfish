import numpy as np
import pytest
from numpy.fft import irfft, rfft, rfftfreq

from modfish.chi.config import ChiParams
from modfish.chi.fit import fit_gain
from modfish.chi.response import preemphasis_response

FS1, FS16 = 325.52, 16.0


def _pair(gain_true, seconds=300.0, spd=3.0, seed=4):
    """A common conductivity signal seen by the SBE 49 (S/m at 16 Hz) and
    by the microconductivity channel (preemphasized, divided by the gain,
    volts at 325.52 Hz)."""
    rng = np.random.default_rng(seed)
    n = int(seconds * FS1)
    f = rfftfreq(n, 1 / FS1)
    # red conductivity spectrum, k^-1 in the gradient => f^-3 in c
    amp = np.where(f > 0, f, np.inf) ** -1.5
    amp[0] = 0.0
    c_spec = amp * np.exp(2j * np.pi * rng.random(f.size))
    c_hi = irfft(c_spec, n=n)
    c_hi = 4.0 + 0.01 * c_hi / c_hi.std()
    t_hi = np.arange(n) / FS1
    t_lo = np.arange(0, seconds, 1 / FS16)
    c_ctd = np.interp(t_lo, t_hi, c_hi)
    p = ChiParams()
    H = preemphasis_response(f, p.R24, p.R25, p.R22, p.C19)
    volts = irfft(rfft(c_hi - 4.0) * H, n=n) / gain_true + 1.5
    return c_ctd, volts, spd


def test_fit_gain_recovers_known_gain():
    c_ctd, volts, spd = _pair(gain_true=50.0)
    params = ChiParams(enabled=True, gain=1.0)
    g = fit_gain(c_ctd, FS16, volts, FS1, spd, params)
    assert g == pytest.approx(50.0, rel=0.02)  # measured +0.6 % with blackmanharris
    # Hamming leaks the red 16 Hz spectrum into the band: +2.6 % (measured), and a
    # longer nsec makes it worse (+5.1 % at 64 s), so the default window is the fix.
    g_ham = fit_gain(c_ctd, FS16, volts, FS1, spd, params, window="hamming")
    assert g_ham == pytest.approx(50.0, rel=0.05)
    assert g_ham > g


def test_fit_gain_independent_of_params_gain():
    c_ctd, volts, spd = _pair(gain_true=25.0)
    g1 = fit_gain(c_ctd, FS16, volts, FS1, spd, ChiParams(enabled=True, gain=1.0))
    g2 = fit_gain(c_ctd, FS16, volts, FS1, spd, ChiParams(enabled=True, gain=99.0))
    assert g1 == pytest.approx(g2, rel=1e-6)


def test_fit_gain_casts_one_down_cast():
    import pandas as pd
    import xarray as xr
    from modfish.chi.fit import fit_gain_casts

    seconds = 300.0
    c_ctd, volts, spd = _pair(gain_true=40.0, seconds=seconds)
    t0 = np.datetime64("2025-12-06T00:00:00")
    t16 = t0 + (np.arange(c_ctd.size) / FS16 * 1e9).astype("timedelta64[ns]")
    depth = spd * np.arange(c_ctd.size) / FS16
    ctd = xr.Dataset(dict(c=("time", c_ctd), depth=("time", depth)), coords=dict(time=t16))
    casts = xr.Dataset(
        dict(start_time=("cast", [t16[16 * 30]]), end_time=("cast", [t16[16 * 270]]),
             direction=("cast", np.array(["down"], dtype=object))),
        coords=dict(cast=[1]))
    ranges = pd.DataFrame([dict(i0=0, n=volts.size, start=t0, fs=FS1)])
    df = fit_gain_casts(ctd, volts.astype(np.float32), ranges, casts, ChiParams())
    assert df.cast.tolist() == [1]
    assert df.gain.iloc[0] == pytest.approx(40.0, rel=0.02)
    assert df.spd.iloc[0] == pytest.approx(spd, rel=0.01)


def test_fit_gain_casts_skips_nan_cast_and_logs_warning(caplog):
    import pandas as pd
    import xarray as xr
    from modfish.chi.fit import fit_gain_casts

    seconds = 300.0
    c_ctd, volts, spd = _pair(gain_true=40.0, seconds=seconds)
    t0 = np.datetime64("2025-12-06T00:00:00")
    t16 = t0 + (np.arange(c_ctd.size) / FS16 * 1e9).astype("timedelta64[ns]")
    depth = spd * np.arange(c_ctd.size) / FS16
    nan_lo, nan_hi = 100, 400
    c_ctd_nan = c_ctd.copy()
    c_ctd_nan[nan_lo:nan_hi] = np.nan
    ctd = xr.Dataset(dict(c=("time", c_ctd_nan), depth=("time", depth)), coords=dict(time=t16))
    casts = xr.Dataset(
        dict(start_time=("cast", [t16[16 * 30], t16[nan_lo + 5]]),
             end_time=("cast", [t16[16 * 270], t16[nan_hi - 5]]),
             direction=("cast", np.array(["down", "down"], dtype=object))),
        coords=dict(cast=[1, 2]))
    ranges = pd.DataFrame([dict(i0=0, n=volts.size, start=t0, fs=FS1)])
    with caplog.at_level("WARNING"):
        df = fit_gain_casts(ctd, volts.astype(np.float32), ranges, casts, ChiParams())
    assert df.cast.tolist() == [1]
    assert df.gain.iloc[0] == pytest.approx(40.0, rel=0.02)
    assert any("cast 2" in r.getMessage() for r in caplog.records)


def test_fit_gain_casts_skips_up_cast():
    import pandas as pd
    import xarray as xr
    from modfish.chi.fit import fit_gain_casts

    seconds = 300.0
    c_ctd, volts, spd = _pair(gain_true=40.0, seconds=seconds)
    t0 = np.datetime64("2025-12-06T00:00:00")
    t16 = t0 + (np.arange(c_ctd.size) / FS16 * 1e9).astype("timedelta64[ns]")
    depth = spd * np.arange(c_ctd.size) / FS16
    ctd = xr.Dataset(dict(c=("time", c_ctd), depth=("time", depth)), coords=dict(time=t16))
    casts = xr.Dataset(
        dict(start_time=("cast", [t16[16 * 30]]), end_time=("cast", [t16[16 * 270]]),
             direction=("cast", np.array(["up"], dtype=object))),
        coords=dict(cast=[1]))
    ranges = pd.DataFrame([dict(i0=0, n=volts.size, start=t0, fs=FS1)])
    df = fit_gain_casts(ctd, volts.astype(np.float32), ranges, casts, ChiParams())
    assert df.empty
    assert list(df.columns) == ["cast", "gain", "spd", "n_ctd", "n_c1"]


def test_fit_gain_casts_skips_too_few_bins_cast(caplog):
    import pandas as pd
    import xarray as xr
    from modfish.chi.fit import fit_gain_casts

    seconds = 300.0
    c_ctd, volts, spd = _pair(gain_true=40.0, seconds=seconds)
    t0 = np.datetime64("2025-12-06T00:00:00")
    t16 = t0 + (np.arange(c_ctd.size) / FS16 * 1e9).astype("timedelta64[ns]")
    depth = spd * np.arange(c_ctd.size) / FS16
    ctd = xr.Dataset(dict(c=("time", c_ctd), depth=("time", depth)), coords=dict(time=t16))
    casts = xr.Dataset(
        dict(start_time=("cast", [t16[16 * 30]]), end_time=("cast", [t16[16 * 270]]),
             direction=("cast", np.array(["down"], dtype=object))),
        coords=dict(cast=[1]))
    ranges = pd.DataFrame([dict(i0=0, n=volts.size, start=t0, fs=FS1)])
    with caplog.at_level("WARNING"):
        df = fit_gain_casts(ctd, volts.astype(np.float32), ranges, casts, ChiParams(),
                             band=(0.05, 0.06))
    assert df.empty
    assert any("cast 1" in r.getMessage() for r in caplog.records)
