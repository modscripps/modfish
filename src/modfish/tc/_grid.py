"""Gap handling shared by the `modfish.tc` corrections and estimators."""

import logging

import numpy as np
import xarray as xr

from modfish.utils import sampling_interval

logger = logging.getLogger(__name__)


def _fill_gaps(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fill NaN in a 1-D array so it can pass through a filter or an FFT.

    Interior NaN are filled by linear interpolation against the sample
    index; NaN at either edge are filled with the nearest finite value
    (constant extrapolation), since there is nothing to interpolate
    between. A single `numpy.interp` call gives both: values outside the
    range of its `xp` (the finite indices) are clamped to the boundary
    `fp` values, which is exactly edge-hold.

    Parameters
    ----------
    x : numpy.ndarray
        One-dimensional array, possibly containing NaN.

    Returns
    -------
    filled : numpy.ndarray
        Copy of `x` with every NaN replaced.
    was_nan : numpy.ndarray
        Boolean mask, same shape as `x`, True where `x` was NaN. Callers
        use this to restore NaN after the finite-input step (filtering,
        FFT) that required this function.

    Notes
    -----
    An all-NaN input has no finite value to interpolate from or clamp
    to; `numpy.interp` raises in that case, so this function does too.
    """
    x = np.asarray(x, dtype=float)
    mask = np.isnan(x)
    filled = x.copy()
    if mask.any():
        idx = np.arange(x.size)
        filled[mask] = np.interp(idx[mask], idx[~mask], x[~mask])
    return filled, mask


def _uniform_slots(time: np.ndarray) -> tuple[np.ndarray, int, float]:
    """Slot index of every sample on a uniform grid at the record's sampling interval.

    A concatenated deployment record has time gaps, so its samples do not
    sit on a uniform grid even though every step of the correction chain
    treats them as if they did. This walks the record sample by sample and
    advances one slot per regular step, `round(step / dt)` slots per step
    that reaches 1.5 times the median step. A gap therefore opens the
    slots it is missing and every other sample keeps the position it
    already had.

    Parameters
    ----------
    time : numpy.ndarray
        Timestamps, dtype datetime64, strictly increasing, at least two of
        them.

    Returns
    -------
    slots : numpy.ndarray
        Integer slot index of every sample, strictly increasing from
        `slots[0] = 0`.
    n_slots : int
        Number of slots on the grid, `slots[-1] + 1`. Equal to `time.size`
        when the record has no gap.
    dt : float
        Sampling interval [s] from `modfish.utils.sampling_interval`.

    Raises
    ------
    ValueError
        If any time step is zero or negative. Two samples would then land
        in the same slot or out of order, and laying the record on a grid
        would drop or reorder samples.

    Notes
    -----
    A gap is detected against the median step, the same rule
    `modfish.utils.sampling_interval` uses to decide which steps go into
    its mean, so every step it counts towards the interval takes exactly
    one slot here. The returned `dt` is that mean and sets only how many
    slots a gap spans. The two differ by 0.8 % on the FCTD's 1 ms-quantized
    16 Hz stamps, where the median step reads 63 ms against a 62.5 ms
    mean, enough for a 94 ms step to be a regular step for the interval
    and a gap for a threshold taken off the mean. A step shorter than half
    the median takes one slot as well; only a step of zero or less raises.

    Accumulating slots along the steps, in place of rounding each sample's
    offset from `time[0]` onto one global grid, is what keeps the mapping
    usable on a real record. The FCTD's stamps drift by more than half a
    sample against any single interval over a multi-hour record: on the
    2025 MOTIVE records that global rounding put two samples in one slot
    12700 times on d07 and 16481 times on d12, while every time step in
    both records is strictly positive. Accumulation has no such drift, and
    it reproduces the old index-as-time behavior exactly inside every
    gap-free stretch.
    """
    time = np.asarray(time)
    dt = sampling_interval(time)
    step = np.diff(time) / np.timedelta64(1, "s")
    if np.any(step <= 0):
        bad = int(np.argmax(step <= 0))
        raise ValueError(
            f"time step {step[bad]} s at sample {bad + 1} is not positive; "
            "two samples would fall in the same slot of the uniform time "
            "grid, or out of order"
        )
    gap = step >= 1.5 * np.median(step)
    advance = np.where(gap, np.round(step / dt), 1.0).astype("int64")
    slots = np.concatenate(([0], np.cumsum(advance)))
    return slots, int(slots[-1]) + 1, dt


def _on_uniform_grid(ds: xr.Dataset, variables: tuple[str, ...], fn) -> xr.Dataset:
    """Run `fn` on `ds` re-laid onto a uniform time grid, return results on `ds`'s own axis.

    Builds a Dataset whose `time` is `time[0] + arange(n_slots) * dt` and
    whose `variables` hold each sample at its slot and NaN elsewhere,
    calls `fn` on it, then takes the original slots back out of the
    result. A time gap therefore reaches `fn` as a run of NaN, which the
    gap fill inside each step bridges with a linear ramp, in place of the
    step between two adjacent samples it would otherwise see.

    Parameters
    ----------
    ds : xarray.Dataset
        CTD time series on a `time` coordinate of dtype datetime64.
    variables : tuple of str
        Names of the variables `fn` reads. Those absent from `ds` are
        skipped.
    fn : callable
        Takes a Dataset and returns a Dataset on the same time axis.

    Returns
    -------
    out : xarray.Dataset
        Deep copy of `ds` with every time-dimensioned variable of `fn`'s
        result written back at the original samples' slots, carrying
        `fn`'s variable and dataset attributes. The time axis is `ds`'s.

    Notes
    -----
    When `n_slots` equals the number of samples the record has no gap and
    `fn(ds)` is called directly, so a gap-free record goes through the
    existing code path with no re-laying at all and its output is
    unchanged to the last bit.

    The uniform axis is built from integer nanoseconds so that no float
    rounding drifts along a multi-million-sample record. Slot 0 and the
    last slot hold original samples by construction, so the inserted NaN
    are always interior and the fill across them is an interpolation.
    """
    slots, n_slots, dt = _uniform_slots(ds.time.data)
    if n_slots == ds.sizes["time"]:
        return fn(ds)

    logger.info(
        "%d samples span %d slots at %g s; running on a uniform time grid",
        ds.sizes["time"],
        n_slots,
        dt,
    )
    time_uniform = ds.time.data[0] + (np.arange(n_slots) * round(dt * 1e9)).astype(
        "timedelta64[ns]"
    )
    data_vars = {}
    for v in variables:
        if v in ds:
            data = np.asarray(ds[v].data)
            if not np.issubdtype(data.dtype, np.floating):
                data = data.astype(float)
            padded = np.full(n_slots, np.nan, dtype=data.dtype)
            padded[slots] = data
            data_vars[v] = (ds[v].dims, padded, dict(ds[v].attrs))
    uniform = xr.Dataset(
        coords=dict(time=("time", time_uniform)),
        data_vars=data_vars,
        attrs=dict(ds.attrs),
    )

    result = fn(uniform)

    out = ds.copy(deep=True)
    for name, da in result.data_vars.items():
        if "time" in da.dims:
            out[name] = (da.dims, np.asarray(da.data)[slots], dict(da.attrs))
        else:
            out[name] = da
    out.attrs = dict(result.attrs)
    return out
