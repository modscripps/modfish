import numpy as np
import xarray as xr

from modfish.fctd.casts import casts_to_dataset, find_casts, label_casts


def sawtooth_pressure(n_casts=3, depth=300.0, fs=16.0, w=1.0, surface=60.0):
    """n_casts down/up pairs to `depth` at speed w, `surface` s pauses at 5 dbar."""
    seg_down = np.linspace(5, depth, int(depth / w * fs))
    seg_up = seg_down[::-1]
    pause = np.full(int(surface * fs), 5.0)
    p = np.concatenate(
        [np.concatenate([pause, seg_down, seg_up]) for _ in range(n_casts)] + [pause]
    )
    time = np.datetime64("2024-11-20") + (np.arange(len(p)) / fs * 1e9).astype(
        "timedelta64[ns]"
    )
    rng = np.random.default_rng(1)
    return p + rng.normal(0, 0.05, len(p)), time


def test_find_casts_counts_and_directions():
    p, time = sawtooth_pressure(n_casts=3)
    casts = find_casts(p, time)
    assert len(casts) == 6
    assert list(casts.direction) == ["down", "up"] * 3
    assert list(casts.cast) == [1, 2, 3, 4, 5, 6]


def test_find_casts_short_bounce_rejected():
    p, time = sawtooth_pressure(n_casts=1, depth=8.0)  # < min_range
    casts = find_casts(p, time)
    assert len(casts) == 0


def test_find_casts_cast_reaches_record_end():
    p, time = sawtooth_pressure(n_casts=1)
    stop = np.flatnonzero(p > 250)[-1]  # truncate mid-upcast
    casts = find_casts(p[:stop], time[:stop])
    assert list(casts.direction) == ["down", "up"]
    assert casts.iloc[-1].i1 >= stop - (16 * 16)  # up to one smooth window from the end


def test_find_casts_gap_splits_nothing_spurious():
    p, time = sawtooth_pressure(n_casts=2)
    casts_full = find_casts(p, time)
    assert (casts_full.i1 > casts_full.i0).all()
    ranges = p[casts_full.i1.values] - p[casts_full.i0.values]
    assert (np.abs(ranges) > 200).all()


def test_label_casts_marks_outside_zero():
    p, time = sawtooth_pressure(n_casts=1)
    casts = find_casts(p, time)
    ds = xr.Dataset(coords={"time": ("time", time)}, data_vars={"p": ("time", p)})
    ds = label_casts(ds, casts)
    assert ds.cast.data[0] == 0  # surface pause before first cast
    assert set(np.unique(ds.cast.data)) == {0, 1, 2}


def test_label_casts_other_time_axis():
    p, time = sawtooth_pressure(n_casts=1)
    casts = find_casts(p, time)
    fast_time = np.datetime64("2024-11-20") + (
        np.arange(len(p) * 4) / 64.0 * 1e9
    ).astype("timedelta64[ns]")
    fast = xr.Dataset(coords={"time": ("time", fast_time)})
    fast = label_casts(fast, casts, time_ref=time)
    assert set(np.unique(fast.cast.data)) <= {0, 1, 2}
    assert (fast.cast.data > 0).any()


def test_casts_to_dataset_shape():
    p, time = sawtooth_pressure(n_casts=2)
    casts = find_casts(p, time)
    cds = casts_to_dataset(casts, time)
    assert cds.sizes["cast"] == 4
    assert str(cds.direction.data[0]) == "down"
