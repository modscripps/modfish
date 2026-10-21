"""Decoding helpers shared by the `.modraw` tag decoders."""

import numpy as np

#: Byte value to hexadecimal digit value, -1 for anything that is not one of
#: `0-9`, `A-F`, `a-f`.
_HEX_DIGIT = np.full(256, -1, dtype=np.int64)
_HEX_DIGIT[48:58] = np.arange(10)  # 0-9
_HEX_DIGIT[65:71] = np.arange(10, 16)  # A-F
_HEX_DIGIT[97:103] = np.arange(10, 16)  # a-f

#: Largest millisecond count that fits datetime64[ns] (2**63 ns), rounded down.
_MAX_MS = (2**63 - 1) // 1_000_000


def hex_columns(chars, i0, i1):
    """Decode columns `i0:i1` of a character array as hexadecimal integers.

    Vectorized over rows. Call it once per field on the stacked records of
    a whole stream: called per block on a few rows at a time, numpy call
    overhead dominated the conversion of a file (about 7 s of a 9 s
    profile, modscripps/modfish#18).

    Parameters
    ----------
    chars : np.ndarray
        Array of uint8 characters, shape (n records, record length).
    i0, i1 : int
        Column range to decode.

    Returns
    -------
    out : np.ndarray
        Decoded values as float64, NaN where a character was not
        hexadecimal. Accumulated most significant digit first in float64,
        so any value below 2**53 (every 16-digit millisecond timestamp
        included) is exact.
    """
    digit = _HEX_DIGIT[chars[:, i0:i1]]
    ok = (digit >= 0).all(axis=1)
    digit = np.where(digit >= 0, digit, 0)
    out = np.zeros(digit.shape[0], dtype=float)
    for j in range(digit.shape[1]):
        out = out * 16 + digit[:, j]
    out[~ok] = np.nan
    return out


def ms_to_datetime(ms):
    """Milliseconds since 1970 to datetime64[ns], NaT where not representable.

    A corrupt but well-formed hex timestamp (one flipped high digit) can
    exceed the datetime64[ns] range. Such values become NaT instead of
    raising, so one bad record does not cost the whole file.

    Parameters
    ----------
    ms : array_like
        Milliseconds since 1970-01-01, integer or float (NaN allowed).

    Returns
    -------
    numpy.ndarray of datetime64[ns]
        NaT where `ms` is NaN, negative or beyond the datetime64[ns] range.
    """
    v = np.asarray(ms)
    if v.dtype.kind in "iu":
        ok = (v >= 0) & (v <= _MAX_MS)
        ms_i8 = np.where(ok, v, 0).astype("int64")
    else:
        v = v.astype(float)
        ok = np.isfinite(v) & (v >= 0) & (v <= _MAX_MS)
        ms_i8 = np.where(ok, v, 0.0).astype("int64")
    out = ms_i8.astype("datetime64[ms]").astype("datetime64[ns]")
    out[~ok] = np.datetime64("NaT", "ns")
    return out


def xor_checksum(data):
    """XOR of every byte in `data`, the checksum SOM frames and NMEA use."""
    chk = 0
    for byte in data:
        chk ^= byte
    return chk
