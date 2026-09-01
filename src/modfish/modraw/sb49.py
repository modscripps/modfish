#!/usr/bin/env python
# coding: utf-8
"""
SBE49 CTD data parsing and conversion for `.modraw` files.

Handles SBE49 output-format-24 frames and provides functions for reading and
processing the CTD time series.
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from .header import read_header, read_body, sbe49_cal, header_setup


#: One SBE49 block, from the tag to the trailing checksum.
_SB49 = re.compile(rb"\$SB49([\s\S]+?)\*([0-9A-Fa-f]{2})\r\n")

#: Length in characters of one SBE49 record: 16 timestamp + 24 data.
_REC_LEN = 40

#: Character offset of the payload within a block, past `$SB49`, the 16
#: character timestamp, the 8 character length and the 3 character `*XX`.
_DATA_OFFSET = 27


def _hex_columns(chars, i0, i1):
    """Decode columns `i0:i1` of a character array as hexadecimal integers.

    Parameters
    ----------
    chars : np.ndarray
        Array of uint8 characters, shape (n records, record length).
    i0, i1 : int
        Column range to decode.

    Returns
    -------
    out : np.ndarray
        Decoded values, NaN where a character was not hexadecimal.
    """
    sub = chars[:, i0:i1]
    out = np.zeros(sub.shape[0], dtype=float)
    ok = np.ones(sub.shape[0], dtype=bool)
    for j in range(sub.shape[1]):
        col = sub[:, j]
        digit = np.full(col.shape, -1, dtype=np.int64)
        digit = np.where((col >= 48) & (col <= 57), col - 48, digit)  # 0-9
        digit = np.where((col >= 65) & (col <= 70), col - 55, digit)  # A-F
        digit = np.where((col >= 97) & (col <= 102), col - 87, digit)  # a-f
        ok &= digit >= 0
        out = out * 16 + np.where(digit >= 0, digit, 0)
    out[~ok] = np.nan
    return out


def sbe49_to_physical(t_raw, c_raw, p_raw, pt_raw, cal):
    """Convert SBE49 engineering counts to physical units.

    Parameters
    ----------
    t_raw, c_raw, p_raw, pt_raw : array_like
        Temperature, conductivity, pressure and pressure-temperature counts.
    cal : dict
        Calibration coefficients as returned by `sbe49_cal`.

    Returns
    -------
    t : np.ndarray
        Temperature [°C].
    c : np.ndarray
        Conductivity [S/m].
    p : np.ndarray
        Pressure [dbar].

    Notes
    -----
    Temperature follows the Sea-Bird thermistor polynomial

    $$ 1/T = a_0 + a_1 \\ln r + a_2 \\ln^2 r + a_3 \\ln^3 r $$

    with $r$ the thermistor resistance derived from the counts. Conductivity
    and pressure follow the corresponding Sea-Bird polynomials. All three are
    transcribed from `mod_som_read_epsi_files_v4.m`.
    """
    mv = (np.asarray(t_raw) - 524288) / 1.6e7
    r = (mv * 2.295e10 + 9.216e8) / (6.144e4 - mv * 5.3e5)
    lr = np.log(r)
    t = (
        1.0 / (cal["ta0"] + cal["ta1"] * lr + cal["ta2"] * lr**2 + cal["ta3"] * lr**3)
        - 273.15
    )

    y = np.asarray(pt_raw) / 13107
    tt = cal["ptempa0"] + cal["ptempa1"] * y + cal["ptempa2"] * y**2
    x = np.asarray(p_raw) - cal["ptca0"] - cal["ptca1"] * tt - cal["ptca2"] * tt**2
    n = x * cal["ptcb0"] / (cal["ptcb0"] + cal["ptcb1"] * tt + cal["ptcb2"] * tt**2)
    p = (cal["pa0"] + cal["pa1"] * n + cal["pa2"] * n**2 - 14.7) * 0.689476

    f = np.asarray(c_raw) / 256 / 1000
    c = (cal["cg"] + cal["ch"] * f**2 + cal["ci"] * f**3 + cal["cj"] * f**4) / (
        1 + cal["ctcor"] * t + cal["cpcor"] * p
    )
    return t, c, p


def load_ctd(file):
    """Read the SBE49 CTD time series from one .modraw file.

    Parameters
    ----------
    file : Path or str
        Path to a .modraw file.

    Returns
    -------
    ds : xr.Dataset
        CTD time series at the SBE49 sample rate, usually 16 Hz. Carries the
        raw counts alongside the converted variables, so that frames the
        instrument returned as zero can be told apart from real values. Header
        setup fields and per-file block tallies are in `ds.attrs`.

    Notes
    -----
    `n_bad_length` counts blocks whose payload does not match the length in
    their own header, `n_bad_checksum` those failing the XOR checksum over the
    payload. Both being zero means the telemetry was clean.
    """
    file = Path(file)
    head = read_header(file)
    cal = sbe49_cal(head)
    if not cal:
        raise ValueError(f"no SBE49 calibration coefficients in the header of {file}")
    body = read_body(file)

    n_block = n_bad_length = n_bad_checksum = 0
    ts, t_raw, c_raw, p_raw, pt_raw = [], [], [], [], []
    for match in _SB49.finditer(body):
        n_block += 1
        block, checksum = match.group(1), match.group(2)
        try:
            nchar = int(block[16:24], 16)
        except ValueError:
            n_bad_length += 1
            continue
        data = block[_DATA_OFFSET:]
        if len(data) != nchar or nchar % _REC_LEN != 0:
            n_bad_length += 1
            continue
        chk = 0
        for byte in data:
            chk ^= byte
        if chk != int(checksum, 16):
            n_bad_checksum += 1
        chars = np.frombuffer(data, dtype=np.uint8).reshape(nchar // _REC_LEN, _REC_LEN)
        ts.append(_hex_columns(chars, 0, 16))
        t_raw.append(_hex_columns(chars, 16, 22))
        c_raw.append(_hex_columns(chars, 22, 28))
        p_raw.append(_hex_columns(chars, 28, 34))
        pt_raw.append(_hex_columns(chars, 34, 38))

    if not ts:
        raise ValueError(f"no usable SB49 blocks in {file}")

    ts = np.concatenate(ts)
    t_raw = np.concatenate(t_raw)
    c_raw = np.concatenate(c_raw)
    p_raw = np.concatenate(p_raw)
    pt_raw = np.concatenate(pt_raw)
    t, c, p = sbe49_to_physical(t_raw, c_raw, p_raw, pt_raw, cal)

    # The Matlab reader treats a median timestamp above 1e9 as milliseconds
    # since 1970 and anything smaller as milliseconds since power on.
    if np.nanmedian(ts) < 1e9:
        raise ValueError(
            f"{file} carries timestamps relative to power on, which are not "
            "supported yet"
        )
    time = pd.to_datetime(ts, unit="ms").values.astype("datetime64[ns]")

    ds = xr.Dataset(
        coords={"time": ("time", time)},
        data_vars={
            "p": ("time", p),
            "t": ("time", t),
            "c": ("time", c),
            "t_raw": ("time", t_raw),
            "c_raw": ("time", c_raw),
            "p_raw": ("time", p_raw),
            "pt_raw": ("time", pt_raw),
        },
    )
    ds.p.attrs = dict(long_name="pressure", units="dbar")
    ds.t.attrs = dict(long_name="temperature", units="°C")
    ds.c.attrs = dict(long_name="conductivity", units="S/m")
    for name in ("t_raw", "c_raw", "p_raw", "pt_raw"):
        ds[name].attrs = dict(long_name=f"{name} counts", units="counts")
    ds.attrs = dict(
        file=file.name,
        n_block=n_block,
        n_bad_length=n_bad_length,
        n_bad_checksum=n_bad_checksum,
        **header_setup(head),
    )
    return ds


def load_ctd_time_series(files):
    """Read and concatenate the SBE49 CTD time series of several .modraw files.

    Parameters
    ----------
    files : iterable of Path or str
        Paths to .modraw files. Sorted by time on output, so they need not be
        given in order.

    Returns
    -------
    ds : xr.Dataset
        Concatenated CTD time series. Per-file block tallies are summed into
        `ds.attrs`; setup fields are taken from the first file and only kept
        where every file agrees.
    """
    parts = [load_ctd(f) for f in files]
    if not parts:
        raise ValueError("no files given")
    ds = xr.concat(parts, dim="time", combine_attrs="drop").sortby("time")
    attrs = {}
    for key in ("n_block", "n_bad_length", "n_bad_checksum"):
        attrs[key] = int(sum(part.attrs[key] for part in parts))
    for key, value in parts[0].attrs.items():
        if key in attrs or key == "file":
            continue
        if all(part.attrs.get(key) == value for part in parts):
            attrs[key] = value
    attrs["files"] = [part.attrs["file"] for part in parts]
    ds.attrs = attrs
    return ds
