#!/usr/bin/env python
# coding: utf-8
"""Utilities"""

from pathlib import Path
import numpy as np
import pandas as pd
import scipy
from munch import munchify


def sampling_interval(time) -> float:
    """Mean sampling interval of a time axis over its gap-free stretches.

    Parameters
    ----------
    time : array_like
        Timestamps, dtype datetime64, at least two of them.

    Returns
    -------
    float
        Sampling interval in seconds. The sampling rate is its inverse.

    Raises
    ------
    ValueError
        If `time` has fewer than two samples, or no step lies within the
        regular range described below.

    Notes
    -----
    Timestamps quantized to 1 ms make a 62.5 ms step (16 Hz, the FCTD's
    SBE49) alternate between 62 and 63 ms, so the median step reads 63 ms
    and a rate taken from it is 0.8 % low. The mean of the regular steps
    reads 62.5 ms. Steps outside 0.5 to 1.5 times the median step are
    left out of the mean: time gaps and duplicate or backward stamps in a
    concatenated record then perturb at most the samples next to them,
    never the interval. Every function in `modfish.tc` and the cast
    detection in `modfish.fctd` take their rate from here so that fitted
    time constants and applied ones share one convention
    (modscripps/modfish#20).

    Examples
    --------
    >>> t = np.datetime64("2025-12-06") + np.round(np.arange(1601) / 16 * 1000).astype("timedelta64[ms]")
    >>> sampling_interval(t)
    0.0625
    """
    time = np.asarray(time)
    if time.size < 2:
        raise ValueError("need at least two samples to estimate a sampling interval")
    dt = np.diff(time) / np.timedelta64(1, "s")
    dt_med = np.median(dt)
    regular = (dt > 0.5 * dt_med) & (dt < 1.5 * dt_med)
    if not regular.any():
        raise ValueError("no regular time step found; is the time axis monotonic?")
    return float(dt[regular].mean())


def mattime_to_datetime64(dnum):
    """Convert Matlab datenum time format to numpy's datetime64 format.

    Parameters
    ----------
    dtnum : array-like
        Time in Matlab datenum format.

    Returns
    -------
    time : np.datetime64
        Time in numpy datetime64 format

    Notes
    -----
    In Matlab, datevec(719529) = [1970 1 1 0 0 0]
    """
    t = pd.to_datetime(dnum - 719529, unit="D")
    if isinstance(t, pd.Timestamp):
        time = t.to_datetime64()
    elif isinstance(t, pd.DatetimeIndex):
        time = t.values
    return time


def datetime64_to_str(dt64, unit="D"):
    """Convert numpy datetime64 object or array to str or array of str.

    Parameters
    ----------
    dt64 : np.datetime64 or array-like
        Time in numpy datetime64 format
    unit : str, optional
        Date unit. Defaults to "D".

    Returns
    -------
    str or array of str

    Notes
    -----
    Valid date unit formats are listed at
    https://numpy.org/doc/stable/reference/arrays.datetime.html#arrays-dtypes-dateunits

    """

    return np.datetime_as_string(dt64, unit=unit).replace("T", " ")


def loadmat(filename, onevar=False, verbose=False):
    """
    Load Matlab .mat files and return as dictionary with .dot-access.

    Parameters
    ----------
    filename : str
        Path to .mat file
    onevar : bool
        Set to true if there is only one variable in the mat file.

    Returns
    -------
    out : dict (Munch)
        Data in a munchified dictionary.
    """

    def _check_keys(dict):
        """
        checks if entries in dictionary are mat-objects. If yes
        todict is called to change them to nested dictionaries
        """
        for key in dict:
            ni = np.size(dict[key])
            if ni <= 1:
                if isinstance(dict[key], scipy.io.matlab.mat_struct):
                    dict[key] = _todict(dict[key])
            else:
                for i in range(0, ni):
                    if isinstance(dict[key][i], scipy.io.matlab.mat_struct):
                        dict[key][i] = _todict(dict[key][i])
        return dict

    def _todict(matobj):
        """
        A recursive function which constructs from matobjects nested dictionaries
        """
        dict = {}
        for strg in matobj._fieldnames:
            elem = matobj.__dict__[strg]
            if isinstance(elem, scipy.io.matlab.mat_struct):
                dict[strg] = _todict(elem)
            else:
                dict[strg] = elem
        return dict

    data = scipy.io.loadmat(filename, struct_as_record=False, squeeze_me=True)
    out = _check_keys(data)

    # Check if there is only one variable in the dataset. If so, directly
    # return only this variable as munchified dataset.
    if not onevar:
        dk = list(out.keys())
        actual_keys = [k for k in dk if k[:2] != "__"]
        if len(actual_keys) == 1:
            if verbose:
                print("found only one variable, returning munchified data structure")
            return munchify(out[actual_keys[0]])
        else:
            out2 = {}
            for k in actual_keys:
                out2[k] = out[k]
            return munchify(out2)

    # for legacy, keep the option in here as well.
    if onevar:
        # let's check if there is only one variable in there and return it
        kk = list(out.keys())
        outvars = []
        for k in kk:
            if k[:2] != "__":
                outvars.append(k)
        if len(outvars) == 1:
            if verbose:
                print("returning munchified data structure")
            return munchify(out[outvars[0]])
        else:
            if verbose:
                print("found more than one var...")
            return out
    else:
        return out


def parse_filename_datetime(file):
    yy, mm, dd, time = file.stem[4:].split("_")
    dtstr = f"20{yy}-{mm}-{dd} {time[:2]}:{time[2:4]}:{time[4:6]}"
    return np.datetime64(dtstr)


def process_file_path(path_input):
    """
    Validate and convert a file path input to a pathlib.Path object.

    This function accepts either a string representing a path or an existing
    pathlib.Path object. If a string is provided, it is converted to a Path
    object.

    Parameters
    ----------
    path_input : {pathlib.Path, str}
        The file path input, which can be an absolute or relative path string,
        or an existing Path object.

    Returns
    -------
    pathlib.Path
        The guaranteed Path object.

    Raises
    ------
    TypeError
        If the input is neither a pathlib.Path object nor a string (str).

    Examples
    --------
    >>> from pathlib import Path
    >>> # Input is a string
    >>> process_file_path('/home/data/file.txt')
    Path('/home/data/file.txt')

    >>> # Input is already a Path object
    >>> p = Path('temp.csv')
    >>> process_file_path(p)
    Path('temp.csv')
    """
    if isinstance(path_input, Path):
        file_path = path_input

    elif isinstance(path_input, str):
        file_path = Path(path_input)

    else:
        raise TypeError(
            f"Input must be a pathlib.Path object or a string (str), "
            f"not {type(path_input).__name__}"
        )

    return file_path


def datetime_linspace(start_time, end_time, n_points):
    """Generates a NumPy array of n linearly spaced datetime64 values.

    The function determines the total duration between the start and end times,
    generates n linearly spaced fractions of this duration, and adds them
    to the start time.

    Parameters
    ----------
    start_time : numpy.datetime64
        The starting time. Must be less than or equal to `end_time`.
    end_time : numpy.datetime64
        The ending time.
    n_points : int
        The number of linearly spaced points to generate. Must be >= 2.

    Returns
    -------
    numpy.ndarray
        A 1D array of datetime64 values with the same unit as the input times.

    Raises
    ------
    ValueError
        If `start_time` is greater than `end_time`.
        If `n_points` is less than 2.

    Examples
    --------
    >>> start = np.datetime64('2025-01-01T10:00:00')
    >>> end = np.datetime64('2025-01-01T10:06:00')
    >>> vec = datetime_linspace(start, end, 4)
    >>> print(vec)
    ['2025-01-01T10:00:00' '2025-01-01T10:02:00'
     '2025-01-01T10:04:00' '2025-01-01T10:06:00']
    """
    if start_time > end_time:
        raise ValueError("start_time must be less than or equal to end_time.")
    if n_points < 2:
        raise ValueError("n_points must be 2 or greater for linspace.")

    # Calculate the total duration
    total_duration = end_time - start_time

    # Extract the base unit (e.g., 's', 'ms', 'ns') and convert to an integer
    # This gets the magnitude of the timedelta in its finest resolution.
    duration_as_int = total_duration.astype(np.int64)

    # Generate n linearly spaced integers from 0 to the total duration magnitude
    # This represents the step size in the time unit
    step_magnitudes = np.linspace(0, duration_as_int, n_points, dtype=np.int64)

    # Convert the integer magnitudes back to timedelta64 objects
    # We create a scalar timedelta of 1 unit and multiply the vector
    # This automatically retains the original time unit of the duration.
    one_unit_timedelta = total_duration.astype('timedelta64[us]') / duration_as_int
    time_deltas = step_magnitudes * one_unit_timedelta

    # Add the time deltas to the start time
    datetime_vector = start_time + time_deltas

    return datetime_vector
