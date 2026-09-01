#!/usr/bin/env python
# coding: utf-8
"""
Altimeter distance decoder for `.modraw` files.

Decodes ALTI (altimeter) frames into distance measurements. The layout is
UNVERIFIED against real bytes: a search of ~700 `.modraw` files across two
MOTIVE cruises turned up no `$ALTI` frames. This implementation is based
purely on Matlab v4 semantics (lines 674-709 of `mod_som_read_epsi_files_v4.m`).
"""

import numpy as np
import pandas as pd
import xarray as xr

from .framer import Packet


def decode_alti(packets: list[Packet]) -> xr.Dataset:
    """Decode framed ALTI packets into an altimeter distance time series.

    Conversion follows Matlab v4 (lines 674-709): `dst = str2double(length_chars)
    * 1e-5 * 1500`. This uses 10 microsecond time units and assumes a sound
    speed of 1500 m/s, with no factor of 1/2 (the Matlab treats it this way;
    any geometry correction is owned by the later pipeline).

    Parameters
    ----------
    packets : list of Packet
        `$ALTI` packets as returned by `modfish.modraw.framer.frame` (already
        filtered to `tag == "ALTI"`; for ALTI frames, the framer leaves
        `payload` empty and puts the raw 8 reading characters in `length_field`).

    Returns
    -------
    ds : xr.Dataset
        Altimeter distance at the ALTI sample rate, dim `time`, var `dst`.
        Unreadable readings (where the 8 characters cannot be parsed as a
        float) are stored as NaN.

    Notes
    -----
    The frame layout is unverified against real ALTI bytes; no ALTI frames
    have been found in any sampled MOTIVE modraw file.
    """
    times = []
    distances = []

    for packet in packets:
        # Convert timestamp from milliseconds to datetime64[ns]
        time_ns = np.datetime64(packet.timestamp_ms, "ms").astype("datetime64[ns]")
        times.append(time_ns)

        # Try to convert the length_field (8 ASCII characters) to float
        try:
            reading_float = float(packet.length_field)
        except (ValueError, TypeError):
            reading_float = np.nan

        # Convert to distance: multiply by 1e-5 and sound speed (1500 m/s)
        distance = reading_float * 1e-5 * 1500
        distances.append(distance)

    # Create xarray Dataset with time coordinate and dst variable
    times_array = np.array(times, dtype="datetime64[ns]")
    distances_array = np.array(distances, dtype=float)

    ds = xr.Dataset(
        coords={"time": ("time", times_array)},
        data_vars={"dst": ("time", distances_array)},
    )

    # Add attributes to the dst variable
    ds.dst.attrs = dict(
        units="m",
        long_name="altimeter distance",
        comment="Assumes 1500 m/s sound speed per Matlab v4 implementation"
    )

    return ds
