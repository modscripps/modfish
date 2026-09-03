#!/usr/bin/env python

"""Tests for `modfish` package."""

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
