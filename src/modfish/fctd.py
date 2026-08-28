# coding: utf-8
"""
FCTD-specific functions. Data-reading routines are in io.py.
"""

from pathlib import Path
from typing import Literal
import matplotlib.pyplot as plt
import gsw
import numpy as np
import scipy
import xarray as xr

import modfish


def split_casts(d):
    pass


def smooth_pressure_derivative(pressure, kernel_size=1024):
    kernel_size_median = kernel_size

    # 1. Median Filter (Smoothing)
    # Median filters typically require an odd kernel size for a centered output. We use the closest
    # odd number (257) for robustness and centering.
    try:
        # Kernel size must be an odd integer, so we use 257
        p = scipy.signal.medfilt(pressure, kernel_size=kernel_size_median + 1)
    except ValueError:
        # Fallback if N < 257 (for short test data)
        print(
            "Warning: Input too short for a 257-point median filter. Adjusting kernel size."
        )
        p = scipy.signal.medfilt(pressure, kernel_size=17)

    # Step 2a: Moving Average Smoothing (First Pass)
    # Kernel definition (1D boxcar filter)
    kernel_ma = np.ones(kernel_size) / kernel_size
    # Applying the moving average filter (1D convolution)
    p_smooth = np.convolve(p, kernel_ma, mode="same")

    # Step 2b: Differentiation (First Derivative)
    dp_v = np.diff(p_smooth, axis=0)
    # Note: The result dp_v will be shorter than p_smooth by 1 sample (N-1 length)

    # Step 2c: Second Moving Average (Smoothing the derivative)
    # Since dp_v is 1D and kernel_ma is 1D, we can reuse the kernel and np.convolve.
    dp = np.convolve(dp_v, kernel_ma, mode="same")

    return dp


def find_casts(pressure, smooth=1024):
    dp = smooth_pressure_derivative(pressure, kernel_size=smooth)
    down_lim = 0.025
    dn = dp > down_lim
    # startdown = dn(find(diff(dn)>1)+1);
    start_down_ind = np.flatnonzero(np.diff(dn)>1) + 1

        # dn = [0, dn];


    # % find jumps in indices to indicate a start of a profile
    # startdown = dn(find(diff(dn)>1)+1);

    # if isempty(startdown)
        # return;
    # end

    # dn = dn(2:end);
    # FCTD.drop = 0*FCTD.time;

    # if dn(1)<startdown(1)
        # startdown=[dn(1) startdown];
    # end

    # if startdown(end)<dn(end)
        # startdown = [startdown dn(end)];
    # end


    # for i=1:(length(startdown)-1)
        # in = intersect(startdown(i):startdown(i+1)-1,dn);
        # FCTD.drop(in) = i;
    # end
    # end
    return start_down


def find_local_extrema_xarray(
    da: xr.DataArray,
    dim: str,
    direction: Literal["min", "max"],
    window: int = 3,
    min_val: float = -np.inf,
    max_val: float = np.inf,
    min_distance: int = 1,
) -> xr.DataArray:
    """Finds local extrema (minima or maxima) in an xarray.DataArray
    with minimum distance separation.

    A point is considered a final extremum if:
    1. It is a local extremum on the data smoothed by the 'window' rolling mean.
    2. Its original value must fall within the inclusive range [min_val, max_val].
    3. It is the most extreme point within 'min_distance' of itself
       (greedy suppression).

    Parameters
    ----------
    da : xarray.DataArray
        The input time series or data array to search for extrema.
    dim : str
        The name of the dimension along which the operation should be
        performed (e.g., 'time', 'x').
    direction : {'min', 'max'}
        The type of extremum to search for: 'min' for minima, 'max' for maxima.
    window : int
        The size of the window used for the rolling mean smoothing operation.
    min_val : float
        The minimum original value (inclusive) an extremum must have to be kept.
    max_val : float
        The maximum original value (inclusive) an extremum must have to be kept.
    min_distance : int
        The minimum number of data points (array indices) required between
        each returned extremum to enforce separation (greedy suppression).

    Returns
    -------
    xarray.DataArray
        A 1D boolean array where ``True`` indicates a final, suppressed local
        extremum at that index.
    """
    if window % 2 == 0:
        print(
            f"Warning: Window size ({window}) is even. "
            "An odd window is recommended for a centered rolling mean."
        )
    if direction not in ["min", "max"]:
        raise ValueError("The 'direction' argument must be 'min' or 'max'.")

    # 1. Apply a centered rolling mean to smooth the data
    smoothed_da = da.rolling({dim: window}, center=True).mean()

    # 2. Find all potential local extrema on the *smoothed* data
    prev_vals = smoothed_da.shift({dim: 1})
    next_vals = smoothed_da.shift({dim: -1})

    if direction == "min":
        # Local minimum: current point is strictly less than neighbors
        is_extremum = (smoothed_da < prev_vals) & (smoothed_da < next_vals)
    else:  # direction == 'max'
        # Local maximum: current point is strictly greater than neighbors
        is_extremum = (smoothed_da > prev_vals) & (smoothed_da > next_vals)

    is_extremum = is_extremum.fillna(False)

    # 3. Create a mask for the specified data value range
    is_in_range = (da >= min_val) & (da <= max_val)

    # 4. Combine masks to get all candidates
    candidate_mask = is_extremum & is_in_range

    # 5. Prepare candidate values for greedy suppression
    # Non-candidates must be "out of the way" when sorting.
    # For 'min' search, non-candidates are +inf (will sort to the end).
    # For 'max' search, non-candidates are -inf (will sort to the end).
    fill_value = np.inf if direction == "min" else -np.inf
    candidate_values = da.where(candidate_mask, fill_value)

    # 6. Apply the 1D greedy suppression function along the 'dim'
    final_mask = xr.apply_ufunc(
        _greedy_suppression_1d,
        candidate_values,
        kwargs={
            "min_distance": min_distance,
            "direction": direction,  # Pass direction to the helper
        },
        input_core_dims=[[dim]],
        output_core_dims=[[dim]],
        exclude_dims=set((dim,)),
        dask="parallelized",
        output_dtypes=[bool],
    )

    return final_mask


def _greedy_suppression_1d(
    values: np.ndarray, min_distance: int, direction: Literal["min", "max"]
) -> np.ndarray:
    """Perform greedy suppression on a 1D array of candidates (minima or maxima).

    This function identifies candidates (non-inf values), sorts them by value
    (most extreme first), and keeps only the most extreme point within any
    given `min_distance` window.

    Parameters
    ----------
    values : numpy.ndarray
        1D array. Candidates have real values, non-candidates are marked by
        :py:obj:`numpy.inf` (for 'min') or :py:obj:`numpy.NINF` (for 'max').
    min_distance : int
        The minimum required distance (in array indices) between kept points.
    direction : {'min', 'max'}
        The type of extremum to search for. Controls the sorting order.

    Returns
    -------
    numpy.ndarray
        A 1D boolean mask. True indicates that the point at that
        index was kept after suppression.
    """
    final_mask = np.zeros_like(values, dtype=bool)

    # 1. Handle simple case
    if min_distance <= 1:
        # Check for *any* non-fill value (np.inf or -np.inf)
        is_candidate = np.logical_and(np.isfinite(values), values != 0)
        final_mask[is_candidate] = True
        return final_mask

    # 2. Get indices of candidates, sorted from MOST EXTREME to least extreme
    if direction == "min":
        # Sort ascending (lowest first)
        sorted_indices = np.argsort(values)
        # Condition to stop (hit the fill value)
        stop_condition = np.isinf
    else:  # direction == 'max'
        # Sort descending (highest first). Use negative values for argsort trick.
        sorted_indices = np.argsort(-values)
        # Condition to stop (hit the fill value)
        stop_condition = lambda x: x == -np.inf  # Use lambda for clarity on -inf

    # A mask to track points that have been "suppressed"
    suppressed = np.zeros_like(values, dtype=bool)

    # 3. Iterate through candidates in order of preference
    for i in sorted_indices:
        # Stop once we hit the fill values
        if stop_condition(values[i]):
            break

        # If this point is not already suppressed, keep it
        if not suppressed[i]:
            final_mask[i] = True

            # Suppress all points within min_distance of this one
            start = max(
                0, i - min_distance + 1
            )  # Start needs to include the start of the window
            end = min(
                len(values), i + min_distance
            )  # End needs to be exclusive, covers min_distance on the right

            suppressed[start:end] = True

    return final_mask
