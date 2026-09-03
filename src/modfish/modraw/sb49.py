#!/usr/bin/env python
# coding: utf-8
"""
SBE49 CTD data parsing and conversion for `.modraw` files.

Handles SBE49 output-format-24 frames and provides functions for reading and
processing the CTD time series.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from .framer import Packet, frame
from .header import header_setup, read_body, read_header, sbe49_cal


#: Length in characters of one SBE49 record: 16 timestamp + 24 data.
_REC_LEN = 40


#: Byte value to hexadecimal digit value, -1 for anything that is not one of
#: `0-9`, `A-F`, `a-f`.
_HEX_DIGIT = np.full(256, -1, dtype=np.int64)
_HEX_DIGIT[48:58] = np.arange(10)  # 0-9
_HEX_DIGIT[65:71] = np.arange(10, 16)  # A-F
_HEX_DIGIT[97:103] = np.arange(10, 16)  # a-f


def _hex_columns(chars, i0, i1):
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


def decode_sb49(packets: list[Packet], cal: dict) -> xr.Dataset:
    """Decode framed SBE49 packets into a CTD time series.

    Parameters
    ----------
    packets : list of Packet
        `$SB49` packets as returned by `modfish.modraw.framer.frame` (already
        filtered to `tag == "SB49"`; passing packets of other tags produces
        garbage, since their payload does not follow the SBE49 record
        layout).
    cal : dict
        SBE49 calibration coefficients, as returned by `sbe49_cal` or
        `parse_dcal`.

    Returns
    -------
    ds : xr.Dataset
        CTD time series at the SBE49 sample rate, usually 16 Hz. Carries the
        raw counts alongside the converted variables, so that frames the
        instrument returned as zero can be told apart from real values.
        `ds.attrs["n_bad_length"]` counts payloads whose length is not a
        multiple of the 40-character record length; those payloads are
        skipped.

    Raises
    ------
    ValueError
        If no packet yields a usable record, or if the median timestamp
        looks like uptime rather than epoch milliseconds (not yet supported).
    """
    n_bad_length = 0
    payloads = []
    for packet in packets:
        data = packet.payload
        if len(data) % _REC_LEN != 0:
            n_bad_length += 1
            continue
        payloads.append(data)

    if not payloads:
        raise ValueError("no usable SB49 blocks")

    # Stack every record of the stream first and decode each field once;
    # see `_hex_columns` for why.
    chars = np.frombuffer(b"".join(payloads), dtype=np.uint8).reshape(-1, _REC_LEN)
    ts = _hex_columns(chars, 0, 16)
    t_raw = _hex_columns(chars, 16, 22)
    c_raw = _hex_columns(chars, 22, 28)
    p_raw = _hex_columns(chars, 28, 34)
    pt_raw = _hex_columns(chars, 34, 38)
    t, c, p = sbe49_to_physical(t_raw, c_raw, p_raw, pt_raw, cal)

    # The Matlab reader treats a median timestamp above 1e9 as milliseconds
    # since 1970 and anything smaller as milliseconds since power on.
    if np.nanmedian(ts) < 1e9:
        raise ValueError(
            "SB49 packets carry timestamps relative to power on, which are "
            "not supported yet"
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
    ds.attrs = dict(n_bad_length=n_bad_length)
    return ds


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
    `load_ctd` frames only `read_body`'s output, not the whole file (see the
    caution in `modfish.modraw.framer` about the header length declared in
    this format sometimes falling inside the block stream). This keeps its
    output byte-for-byte compatible with earlier versions; a whole-file
    reader is a separate concern.

    `n_bad_length` counts blocks whose payload length is not a multiple of
    the 40-character SBE49 record length. `n_bad_checksum` now comes from
    `framer.FrameStats` and counts trailing XOR checksum failures across
    *all* tags in the framed body, not just `$SB49` as before. Both being
    zero means no length- or checksum-anomalous SB49 records were found.
    It does not mean the framing was clean end to end: a corrupted length
    field that never produces a valid trailer yields no `Packet` at all, so
    the scanner resyncs past it without either counter seeing it (see
    `framer.FrameStats.n_resync`, not surfaced here).
    """
    file = Path(file)
    head = read_header(file)
    cal = sbe49_cal(head)
    if not cal:
        raise ValueError(f"no SBE49 calibration coefficients in the header of {file}")
    packets, stats = frame(read_body(file))
    ds = decode_sb49([p for p in packets if p.tag == "SB49"], cal)
    ds.attrs = dict(
        file=file.name,
        n_block=stats.tag_counts.get("SB49", 0),
        n_bad_length=ds.attrs.get("n_bad_length", 0),
        n_bad_checksum=stats.n_bad_checksum,
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
