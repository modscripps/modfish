#!/usr/bin/env python
# coding: utf-8
"""
Reading `.modraw` files written by the MOD acquisition software.

A `.modraw` file is ASCII. It opens with a header, whose length in bytes is
given on its very first line, and continues with a stream of blocks, one per
sensor, of the form

```
T<host ticks>$<TAG><16 hex timestamp><8 hex length>*<chk><data>*<chk>\\r\\n
```

Only the SBE49 CTD stream (`$SB49`) is decoded here. Its payload is a whole
number of 40-character records: a 16-hex millisecond timestamp followed by the
SBE49 output-format-24 frame, `TTTTTTCCCCCCPPPPPPtttt`, plus two trailing
characters that the Matlab reader discards.

This module follows
`MOD_fish_lib/EPSILOMETER/epsilib/mod_som_read_epsi_files_v4.m`, including its
SBE49 calibration polynomials, and adds a per-block checksum check that the
Matlab reader does not do.

Rudimentary on purpose; see
[issue #13](https://github.com/modscripps/modfish/issues/13).
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

#: One SBE49 block, from the tag to the trailing checksum.
_SB49 = re.compile(rb"\$SB49([\s\S]+?)\*([0-9A-Fa-f]{2})\r\n")

#: `$GPZDA,hhmmss.ss,dd,mm,yyyy`, the GPS UTC time sentence.
_ZDA = re.compile(rb"\$GPZDA,(\d{6}\.\d+),(\d{2}),(\d{2}),(\d{4})")

#: SBE49 calibration coefficients as they are named in the header.
_CAL_KEYS = (
    "TA0 TA1 TA2 TA3 CG CH CI CJ CTCOR CPCOR PA0 PA1 PA2 "
    "PTCA0 PTCA1 PTCA2 PTCB0 PTCB1 PTCB2 PTEMPA0 PTEMPA1 PTEMPA2"
).split()

#: Length in characters of one SBE49 record: 16 timestamp + 24 data.
_REC_LEN = 40

#: Character offset of the payload within a block, past `$SB49`, the 16
#: character timestamp, the 8 character length and the 3 character `*XX`.
_DATA_OFFSET = 27


def read_header(file):
    """Read the header of a .modraw file.

    The header length in bytes is given on the first line of the file.

    Parameters
    ----------
    file : Path or str
        Path to a .modraw file.

    Returns
    -------
    head : str
        Header text.
    """
    with open(file, "rb") as f:
        nbytes = int(f.readline().split(b"=")[1])
        f.seek(0)
        return f.read(nbytes).decode("latin-1")


def read_body(file):
    """Read everything in a .modraw file after the header.

    Parameters
    ----------
    file : Path or str
        Path to a .modraw file.

    Returns
    -------
    body : bytes
        File contents past the header.
    """
    head = read_header(file)
    with open(file, "rb") as f:
        f.seek(len(head.encode("latin-1")))
        return f.read()


def header_setup(head):
    """Extract the acquisition setup fields from a .modraw header.

    Parameters
    ----------
    head : str
        Header text as returned by `read_header`.

    Returns
    -------
    setup : dict
        Survey, vehicle, instrument serial number and the like. Keys are
        lowercased and stripped of their `CTD.` prefix.
    """
    setup = {}
    for key in ("survey", "experiment", "cruise", "vehicle", "fishflag", "SerialNum"):
        m = re.search(rf"CTD\.{key}\s*=\s*'([^']*)'", head)
        if m:
            setup[key.lower()] = m.group(1)
    m = re.search(r"GM_TIME\s*=\s*'([^']+)'", head)
    if m:
        setup["gm_time"] = m.group(1)
    return setup


def sbe49_cal(head):
    """Extract the SBE49 calibration coefficients from a .modraw header.

    Parameters
    ----------
    head : str
        Header text as returned by `read_header`.

    Returns
    -------
    cal : dict
        Calibration coefficients, keys lowercased.
    """
    cal = {}
    for key in _CAL_KEYS:
        m = re.search(rf"^{key}\s*=\s*(\S+)", head, re.MULTILINE)
        if m:
            cal[key.lower()] = float(m.group(1))
    return cal


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


def load_gps_time(file):
    """Read the GPS UTC timestamps interleaved in a .modraw file.

    Useful as an absolute reference against the acquisition computer's clock,
    which is what the file names and the header `GM_TIME` are based on.

    Parameters
    ----------
    file : Path or str
        Path to a .modraw file.

    Returns
    -------
    time : pd.DatetimeIndex
        UTC times from the `$GPZDA` sentences.
    """
    out = []
    for match in _ZDA.finditer(read_body(file)):
        hms, day, month, year = (g.decode() for g in match.groups())
        out.append(pd.Timestamp(f"{year}-{month}-{day} {hms[:2]}:{hms[2:4]}:{hms[4:]}"))
    return pd.DatetimeIndex(out)


def block_counts(file, tags=("SB49", "EFE4", "VNAV", "VNMAR", "ECOP", "ALTI", "GPZDA")):
    """Count the blocks of each stream in a .modraw file.

    A quick way to see which sensors were writing to a file, and at what rate.

    Parameters
    ----------
    file : Path or str
        Path to a .modraw file.
    tags : iterable of str, optional
        Stream tags to count.

    Returns
    -------
    counts : dict
        Number of occurrences of each tag.
    """
    body = read_body(file)
    return {tag: len(re.findall(rb"\$" + tag.encode(), body)) for tag in tags}
