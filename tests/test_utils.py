#!/usr/bin/env python

"""Tests for `modfish` package."""

import gsw
import numpy as np
import pytest

import modfish


def test_mattime_to_datetime64():
    pass


def _quantized_time(fs=16.0, seconds=600, quantum_ms=1):
    """`seconds` of `fs` Hz timestamps rounded to `quantum_ms`."""
    ms = (
        np.round(np.arange(int(fs * seconds) + 1) / fs * 1000 / quantum_ms) * quantum_ms
    )
    return np.datetime64("2025-12-06") + ms.astype("timedelta64[ms]")


def _jittered_16hz(seconds=600):
    """16 Hz stamps at 1 ms quantization with the step pattern of a real record.

    A repeating 61, 63, 63, 63 ms pattern: the mean step is 62.5 ms, the
    median 63 ms. Rounding an exact 62.5 ms grid alone gives equal counts
    of 62 and 63 and a median of 62.5; the real FCTD stamps carry enough
    jitter that 63 is the modal step (modscripps/modfish#20).
    """
    steps = np.tile([61, 63, 63, 63], 4 * seconds)
    ms = np.concatenate([[0], np.cumsum(steps)])
    return np.datetime64("2025-12-06") + ms.astype("timedelta64[ms]")


def test_sampling_interval_ms_quantized_16hz_is_exact():
    time = _jittered_16hz()
    # The median step is the bias the helper exists to avoid: 63 ms on a
    # 62.5 ms grid, 0.8 % low in rate.
    assert np.median(np.diff(time) / np.timedelta64(1, "s")) == pytest.approx(0.063)
    assert modfish.utils.sampling_interval(time) == pytest.approx(0.0625, rel=1e-9)


def test_sampling_interval_8hz_is_not_read_as_16hz():
    assert modfish.utils.sampling_interval(_quantized_time(fs=8.0)) == pytest.approx(
        0.125, rel=1e-9
    )


def test_sampling_interval_excludes_gaps_and_duplicate_stamps():
    time = _jittered_16hz()
    time = np.concatenate([time[:5000], time[5000:] + np.timedelta64(3600, "s")])
    time = np.insert(time, 100, time[100])
    # The gap swallows one regular step, so the mean over the remaining
    # 9599 steps sits within 1e-6 of 62.5 ms; the median would read 63.
    assert modfish.utils.sampling_interval(time) == pytest.approx(0.0625, rel=1e-5)


def test_sampling_interval_single_sample_raises():
    with pytest.raises(ValueError, match="two samples"):
        modfish.utils.sampling_interval(_quantized_time()[:1])


def _linear_profile(up=False):
    """0 to 1000 dbar at 0.5 dbar, temperature and salinity linear in pressure."""
    p = np.arange(0, 1000.5, 0.5)
    t = 20 - 16 * p / 1000
    s = 34 + p / 1000
    if up:
        return s[::-1], t[::-1], p[::-1]
    return s, t, p


def test_nsqfcn_matches_gsw_nsquared_on_smooth_profile():
    s, t, p = _linear_profile()
    lon, lat = -125.0, 45.0
    n2, pout = modfish.utils.nsqfcn(s, t, p, p0=0, dp=10, lon=lon, lat=lat)
    SA = gsw.SA_from_SP(s, p, lon, lat)
    CT = gsw.CT_from_t(SA, t, p)
    n2_gsw, p_mid = gsw.Nsquared(SA, CT, p, lat)
    ref = np.interp(pout, p_mid, n2_gsw)
    assert n2.shape == pout.shape
    assert pout[0] >= p[0]
    np.testing.assert_allclose(np.diff(pout), 10.0)
    assert (n2 > 0).all()
    # dp is in dbar where gsw differences in Pa, about 1 % apart at depth.
    np.testing.assert_allclose(n2, ref, rtol=0.03)


def test_nsqfcn_upward_profile_gives_the_downward_result():
    down = modfish.utils.nsqfcn(*_linear_profile(), p0=0, dp=10, lon=-125.0, lat=45.0)
    up = modfish.utils.nsqfcn(
        *_linear_profile(up=True), p0=0, dp=10, lon=-125.0, lat=45.0
    )
    np.testing.assert_allclose(up[0], down[0])
    np.testing.assert_allclose(up[1], down[1])


def test_nsqfcn_all_nan_profile_returns_nan():
    s, t, p = _linear_profile()
    n2, pout = modfish.utils.nsqfcn(s * np.nan, t, p, p0=0, dp=10, lon=-125.0, lat=45.0)
    assert np.isnan(n2) and np.isnan(pout)


def test_nsqfcn_profile_shorter_than_filter_padding_returns_nan():
    s, t, p = _linear_profile()
    n2, pout = modfish.utils.nsqfcn(
        s[:30], t[:30], p[:30], p0=0, dp=10, lon=-125.0, lat=45.0
    )
    assert np.isnan(n2) and np.isnan(pout)


def _git(*args, cwd):
    import subprocess

    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)


def test_provenance_in_this_checkout_names_version_and_head():
    import importlib.metadata
    import subprocess
    from pathlib import Path

    prov = modfish.utils.provenance()
    assert prov["modfish_version"] == importlib.metadata.version("modfish")
    pkg = Path(modfish.__file__).parent
    head = subprocess.run(["git", "-C", str(pkg), "rev-parse", "--short", "HEAD"],
                          capture_output=True, text=True)
    if head.returncode != 0:
        pytest.skip("modfish not running from a git checkout")
    assert prov["modfish_commit"].split("+")[0] == head.stdout.strip()


def test_provenance_outside_git_has_no_commit(tmp_path):
    (tmp_path / "__init__.py").write_text("")
    prov = modfish.utils.provenance(tmp_path / "__init__.py")
    assert set(prov) == {"modfish_version"}


def test_provenance_skips_an_untracked_install_inside_another_repo(tmp_path):
    _git("init", "-q", cwd=tmp_path)
    _git("-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "--allow-empty", "-m", "x", cwd=tmp_path)
    site = tmp_path / ".venv" / "modfish"
    site.mkdir(parents=True)
    (site / "__init__.py").write_text("")
    assert "modfish_commit" not in modfish.utils.provenance(site / "__init__.py")


def test_provenance_marks_a_dirty_tree_and_ignores_untracked_files(tmp_path):
    _git("init", "-q", cwd=tmp_path)
    (tmp_path / "__init__.py").write_text("a = 1\n")
    _git("add", "__init__.py", cwd=tmp_path)
    _git("-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "-m", "x", cwd=tmp_path)
    (tmp_path / "notes.txt").write_text("untracked")
    clean = modfish.utils.provenance(tmp_path / "__init__.py")["modfish_commit"]
    assert "+" not in clean and len(clean) >= 7
    (tmp_path / "__init__.py").write_text("a = 2\n")
    assert modfish.utils.provenance(tmp_path / "__init__.py")["modfish_commit"] == clean + "+dirty"


def _time16(n, start="2025-12-06"):
    return np.datetime64(start) + np.round(np.arange(n) / 16 * 1000).astype("timedelta64[ms]")


def test_gap_aware_rate_matches_index_gradient_without_gaps():
    from scipy.ndimage import uniform_filter1d

    t = _time16(800)
    x = 3.0 * np.arange(800) / 16 + np.sin(np.arange(800) / 30)
    fs = 1 / modfish.utils.sampling_interval(t)
    old = uniform_filter1d(np.gradient(x) * fs, 16, mode="nearest")
    np.testing.assert_array_equal(modfish.utils.gap_aware_rate(x, t, 1.0), old)


def test_gap_aware_rate_has_no_spike_across_a_gap():
    t = np.concatenate([_time16(400), _time16(400, "2025-12-06T00:05:00")])
    x = np.concatenate([3.0 * np.arange(400) / 16, 700 + 3.0 * np.arange(400) / 16])
    rate = modfish.utils.gap_aware_rate(x, t, 1.0)
    np.testing.assert_allclose(rate, 3.0, rtol=1e-4)  # ms-quantized stamps


def test_gap_aware_rate_is_nan_on_a_single_sample_run():
    t = np.concatenate([_time16(100), _time16(1, "2025-12-06T00:01:00"), _time16(100, "2025-12-06T00:02:00")])
    x = np.arange(201, dtype=float)
    rate = modfish.utils.gap_aware_rate(x, t, 1.0)
    assert np.isnan(rate[100]) and np.isfinite(np.delete(rate, 100)).all()
