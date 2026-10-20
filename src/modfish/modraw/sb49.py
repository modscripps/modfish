#!/usr/bin/env python
# coding: utf-8
"""
SBE49 CTD data parsing and conversion for `.modraw` files.

Handles SBE49 output-format-24 frames. `modfish.modraw.read` is the entry
point for a file.
"""

import numpy as np
import xarray as xr

from ._util import hex_columns, ms_to_datetime
from .framer import Packet


#: Length in characters of one SBE49 record: 16 timestamp + 24 data.
_REC_LEN = 40


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
        Attrs count what was dropped: `n_bad_checksum` packets whose
        trailing checksum failed, `n_bad_length` payloads whose length is
        not a multiple of the 40-character record length, and
        `n_bad_time` records whose timestamp is not hexadecimal or lies
        outside the datetime64[ns] range (a flipped high digit).

    Raises
    ------
    ValueError
        If no packet yields a usable record, or if the median timestamp
        looks like uptime rather than epoch milliseconds (not yet supported).
    """
    n_bad_length = 0
    n_bad_checksum = 0
    payloads = []
    for packet in packets:
        if not packet.chk_ok:
            n_bad_checksum += 1
            continue
        data = packet.payload
        if len(data) % _REC_LEN != 0:
            n_bad_length += 1
            continue
        payloads.append(data)

    if not payloads:
        raise ValueError("no usable SB49 blocks")

    # Stack every record of the stream first and decode each field once;
    # see `hex_columns` for why.
    chars = np.frombuffer(b"".join(payloads), dtype=np.uint8).reshape(-1, _REC_LEN)
    ts = hex_columns(chars, 0, 16)
    t_raw = hex_columns(chars, 16, 22)
    c_raw = hex_columns(chars, 22, 28)
    p_raw = hex_columns(chars, 28, 34)
    pt_raw = hex_columns(chars, 34, 38)
    t, c, p = sbe49_to_physical(t_raw, c_raw, p_raw, pt_raw, cal)

    # The Matlab reader treats a median timestamp above 1e9 as milliseconds
    # since 1970 and anything smaller as milliseconds since power on.
    if np.nanmedian(ts) < 1e9:
        raise ValueError(
            "SB49 packets carry timestamps relative to power on, which are "
            "not supported yet"
        )
    time = ms_to_datetime(ts)
    keep = ~np.isnat(time)
    n_bad_time = int((~keep).sum())
    time, t, c, p = time[keep], t[keep], c[keep], p[keep]
    t_raw, c_raw, p_raw, pt_raw = t_raw[keep], c_raw[keep], p_raw[keep], pt_raw[keep]

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
    ds.attrs = dict(n_bad_checksum=n_bad_checksum, n_bad_length=n_bad_length,
                    n_bad_time=n_bad_time)
    return ds

