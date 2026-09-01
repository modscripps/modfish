#!/usr/bin/env python
# coding: utf-8
"""Decode ECOP Tridente fluorometer packets from `.modraw` files.

The ECOP records (Tridente fluorometer) are 28-character hex payloads
containing a millisecond timestamp and three 16-bit backscatter and
fluorescence measurements.

Normalized values are instrument-relative, pending physical calibration.
See Matlab reference `mod_som_read_epsi_files_v4.m` lines 2001-2003 for the
scaling formulas. Note that the Matlab reader (lines 1918-1923) defines
Tridente calibration coefficients `cal0` and `cal1` but never applies them,
so these normalized quantities are produced as instrument-native counts,
scaled to a standard range but not physically calibrated.

MOTIVE 2025 files record ECOP blocks at approximately 32 blocks/s, not the
16 Hz assumed by the Matlab reader. Some packets share timestamps with
earlier packets (duplicates), and occasional payloads carry binary garbage
instead of ASCII hex characters; these decode to NaN rows via the hex-column
path, with raw values preserved as NaN (no records are dropped).
"""

import numpy as np
import pandas as pd
import xarray as xr

from .framer import Packet
from .sb49 import _hex_columns


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
        `ds.attrs["n_bad_length"]` counts payloads whose length is not 28;
        those payloads are skipped. Payloads with binary garbage instead of
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
    Normalized values per Matlab v4 lines 2001-2003: `(raw/65535 - 0.5)/scale`.
    Scales are: bb 0.05, chla 50, fDOM 1000. The Matlab reader defines
    Tridente cal0/cal1 coefficients (lines 1918-1923) but never applies them,
    so these quantities are instrument-relative and lack physical calibration.
    """
    n_bad_length = 0
    ts, bb_raw, chla_raw, fdom_raw = [], [], [], []

    for packet in packets:
        data = packet.payload
        if len(data) != _REC_LEN:
            n_bad_length += 1
            continue

        # Stack this 28-character record as a (1, 28) row of uint8
        chars = np.frombuffer(data, dtype=np.uint8).reshape(1, _REC_LEN)
        ts.append(_hex_columns(chars, 0, 16))
        bb_raw.append(_hex_columns(chars, 16, 20))
        chla_raw.append(_hex_columns(chars, 20, 24))
        fdom_raw.append(_hex_columns(chars, 24, 28))

    if not ts:
        raise ValueError("no usable ECOP blocks")

    ts = np.concatenate(ts)
    bb_raw = np.concatenate(bb_raw)
    chla_raw = np.concatenate(chla_raw)
    fdom_raw = np.concatenate(fdom_raw)

    # Convert timestamp to datetime64[ns]. NaN timestamps produce NaT.
    time = pd.to_datetime(ts, unit="ms").values.astype("datetime64[ns]")

    # Compute normalized values per Matlab v4 lines 2001-2003
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
    ds.attrs = dict(n_bad_length=n_bad_length)
    return ds
