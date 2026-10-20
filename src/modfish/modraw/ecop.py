#!/usr/bin/env python
# coding: utf-8
"""Decode ECOP Tridente fluorometer packets from `.modraw` files.

The ECOP records (Tridente fluorometer) are 28-character hex payloads
containing a millisecond timestamp and three 16-bit backscatter and
fluorescence measurements.

Normalized values are instrument-relative, pending physical calibration.
The scaling follows the Matlab reader `mod_som_read_epsi_files_v4.m`. That
reader defines Tridente calibration coefficients `cal0` and `cal1` but never
applies them, so these normalized quantities are instrument-native counts,
scaled to a standard range but not physically calibrated.

MOTIVE 2025 files record ECOP blocks at approximately 32 blocks/s, not the
16 Hz assumed by the Matlab reader. Some packets share timestamps with
earlier packets (duplicates), and occasional payloads carry binary garbage
instead of ASCII hex characters; these decode to NaN rows via the hex-column
path, with raw values preserved as NaN (no records are dropped).
"""

import numpy as np
import xarray as xr

from ._util import hex_columns, ms_to_datetime
from .framer import Packet


#: Length in characters of one ECOP record.
_REC_LEN = 28


def decode_ecop(packets: list[Packet]) -> xr.Dataset:
    """Decode ECOP Tridente fluorometer packets into a time series Dataset.

    Parameters
    ----------
    packets : list of Packet
        `$ECOP` packets as returned by `modfish.modraw.framer.frame` (already
        filtered to `tag == "ECOP"`; passing packets of other tags produces
        garbage, since their payload does not follow the ECOP record layout).

    Returns
    -------
    ds : xr.Dataset
        Fluorometer time series. Carries the raw counts alongside normalized
        variables: `bb`, `chla`, `fdom` (normalized) and `bb_raw`, `chla_raw`,
        `fdom_raw` (counts, float to preserve NaN). Dimension `time` has length
        equal to the number of ECOP packets with a valid 28-character payload.
        Attrs count what was dropped: `n_bad_checksum` packets whose
        trailing checksum failed, `n_bad_length` payloads whose length is
        not 28, and `n_bad_time` records whose timestamp is not hexadecimal
        or lies outside the datetime64[ns] range. Payloads with binary garbage instead of
        ASCII hex produce NaN data fields (`bb_raw`, `chla_raw`, `fdom_raw`
        and the normalized `bb`, `chla`, `fdom`); the timestamp bytes can
        still happen to decode as valid hex, so `time` for a garbage row is
        not reliably NaT. Filter garbage rows on the NaN data values, not
        on `time`.

    Raises
    ------
    ValueError
        If no packet yields a usable record.

    Notes
    -----
    Normalized values as in the Matlab reader: `(raw/65535 - 0.5)/scale`,
    with scales bb 0.05, chla 50, fDOM 1000. The Matlab reader defines
    Tridente cal0/cal1 coefficients but never applies them, so these
    quantities are instrument-relative and lack physical calibration.
    """
    n_bad_length = 0
    n_bad_checksum = 0
    payloads = []
    for packet in packets:
        if not packet.chk_ok:
            n_bad_checksum += 1
            continue
        data = packet.payload
        if len(data) != _REC_LEN:
            n_bad_length += 1
            continue
        payloads.append(data)

    if not payloads:
        raise ValueError("no usable ECOP blocks")

    # Stack every record of the stream first and decode each field once;
    # see `hex_columns` for why.
    chars = np.frombuffer(b"".join(payloads), dtype=np.uint8).reshape(-1, _REC_LEN)
    ts = hex_columns(chars, 0, 16)
    bb_raw = hex_columns(chars, 16, 20)
    chla_raw = hex_columns(chars, 20, 24)
    fdom_raw = hex_columns(chars, 24, 28)

    time = ms_to_datetime(ts)
    keep = ~np.isnat(time)
    n_bad_time = int((~keep).sum())
    time, bb_raw, chla_raw, fdom_raw = time[keep], bb_raw[keep], chla_raw[keep], fdom_raw[keep]

    # normalized values as in the Matlab reader
    bb = (bb_raw / 65535 - 0.5) / 0.05
    chla = (chla_raw / 65535 - 0.5) / 50
    fdom = (fdom_raw / 65535 - 0.5) / 1000

    ds = xr.Dataset(
        coords={"time": ("time", time)},
        data_vars={
            "bb": ("time", bb),
            "chla": ("time", chla),
            "fdom": ("time", fdom),
            "bb_raw": ("time", bb_raw),
            "chla_raw": ("time", chla_raw),
            "fdom_raw": ("time", fdom_raw),
        },
    )
    ds.bb.attrs = dict(long_name="backscatter", units="normalized")
    ds.chla.attrs = dict(long_name="chlorophyll-a fluorescence", units="normalized")
    ds.fdom.attrs = dict(long_name="fDOM fluorescence", units="normalized")
    ds.bb_raw.attrs = dict(long_name="backscatter counts", units="counts")
    ds.chla_raw.attrs = dict(long_name="chlorophyll-a fluorescence counts", units="counts")
    ds.fdom_raw.attrs = dict(long_name="fDOM fluorescence counts", units="counts")
    ds.attrs = dict(n_bad_checksum=n_bad_checksum, n_bad_length=n_bad_length,
                    n_bad_time=n_bad_time)
    return ds
