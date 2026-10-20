#!/usr/bin/env python
# coding: utf-8
"""
Altimeter distance decoder for `.modraw` files.

Decodes ALTI (altimeter) frames into distance measurements. The layout is
unverified against real bytes: no `$ALTI` frame turned up in about 700
`.modraw` files from two MOTIVE cruises. The decoding follows the Matlab
reader `mod_som_read_epsi_files_v4.m`.
"""

import numpy as np
import xarray as xr

from ._util import ms_to_datetime
from .framer import Packet


def decode_alti(packets: list[Packet]) -> xr.Dataset:
    """Decode framed ALTI packets into an altimeter distance time series.

    Conversion as in the Matlab reader: `dst = reading * 1e-5 * 1500`, the
    reading in units of 10 microseconds and a sound speed of 1500 m/s. The
    Matlab reader applies no factor of 1/2 and treats `dst` as the slant
    distance to the bottom, from which it computes height above bottom for
    the 10 degree mounting angle. Whether the instrument reports one-way or
    round-trip time is not verified; `dst` follows the Matlab convention.

    Parameters
    ----------
    packets : list of Packet
        `$ALTI` packets as returned by `modfish.modraw.framer.frame` (already
        filtered to `tag == "ALTI"`; for ALTI frames, the framer leaves
        `payload` empty and puts the raw 8 reading characters in `length_field`).

    Returns
    -------
    ds : xr.Dataset
        Altimeter distance at the ALTI sample rate, dim `time`, var `dst`
        (m). Unreadable readings (where the 8 characters cannot be parsed
        as a float) are stored as NaN, as are readings whose timestamp
        lies outside the datetime64[ns] range (time NaT).

    Notes
    -----
    The frame layout is unverified against real ALTI bytes; no ALTI frames
    have been found in any sampled MOTIVE modraw file.
    """
    distances = []
    for packet in packets:
        # Try to convert the length_field (8 ASCII characters) to float
        try:
            reading_float = float(packet.length_field)
        except (ValueError, TypeError):
            reading_float = np.nan

        # Convert to distance: multiply by 1e-5 and sound speed (1500 m/s)
        distance = reading_float * 1e-5 * 1500
        distances.append(distance)

    times_array = ms_to_datetime(np.array([p.timestamp_ms for p in packets], dtype="uint64"))
    distances_array = np.array(distances, dtype=float)

    ds = xr.Dataset(
        coords={"time": ("time", times_array)},
        data_vars={"dst": ("time", distances_array)},
    )

    # Add attributes to the dst variable
    ds.dst.attrs = dict(
        units="m",
        long_name="altimeter distance",
        comment="Reading x 1e-5 s x 1500 m/s, no factor 1/2, as in the Matlab "
                "reader. Frame layout unverified against real bytes."
    )

    return ds
