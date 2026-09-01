#!/usr/bin/env python
# coding: utf-8
"""
GPS time extraction from `.modraw` files.

Parses $GPZDA sentences (and $INZDA, the talker prefix 2024-format files use
for the same sentence) from the GPS stream to extract absolute UTC
timestamps.
"""

import re

import numpy as np
import pandas as pd
import xarray as xr

from .header import read_body


#: `$GPZDA,hhmmss.ss,dd,mm,yyyy`, the GPS UTC time sentence. 2024-format files
#: (e.g. `skq202417s`) use the `$INZDA` talker prefix instead of `$GPZDA` for
#: the same sentence (same fields, observed against
#: EPSI24_11_26_102923.modraw), so both are accepted here, mirroring `_GGA`
#: below which already accepted both prefixes.
_ZDA = re.compile(rb"\$(?:GP|IN)ZDA,(\d{6}\.\d+),(\d{2}),(\d{2}),(\d{4})")

#: `$GPGGA,hhmmss.ss,ddmm.mmmm,N/S,dddmm.mmmm,E/W`, the GPS position sentence.
#: NMEA payloads cannot contain `$`, so regexing the ASCII GPS stream directly
#: (rather than framing it like the binary SOM blocks) is safe here.
_GGA = re.compile(
    rb"\$(?:GP|IN)GGA,"
    rb"(\d{6}\.?\d*),(\d{2})(\d{2}\.\d+),([NS]),(\d{3})(\d{2}\.\d+),([EW])"
)


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
        UTC times from the `$GPZDA`/`$INZDA` sentences.
    """
    out = []
    for match in _ZDA.finditer(read_body(file)):
        hms, day, month, year = (g.decode() for g in match.groups())
        out.append(pd.Timestamp(f"{year}-{month}-{day} {hms[:2]}:{hms[2:4]}:{hms[4:]}"))
    return pd.DatetimeIndex(out)


def decode_gga(body: bytes) -> xr.Dataset:
    """Decode `$GPGGA` position sentences into a fix time series.

    FCTD GPS sentences carry the ordinary `T<10-digit>` laptop-clock prefix
    used throughout `.modraw` files, not the hex acquisition timestamp that
    precedes binary SOM blocks; a GGA sentence's own `hhmmss.ss` field gives
    only time-of-day, with no date. The date is instead taken from
    `$GPZDA`/`$INZDA` sentences interleaved in the same stream: each GGA is
    paired with the date of its nearest `$GPZDA`/`$INZDA` sentence by byte
    offset (so a GGA is dated from whichever ZDA it fell closest to in the
    file, before or after).

    Combining that date with the GGA's time-of-day can land up to a day off
    right around midnight, when the nearest ZDA carries the other side of
    the rollover; this is corrected by shifting the combined timestamp by
    one day whenever it differs from the paired ZDA's own timestamp by more
    than 12 hours. Cruise files are short (~40 min), so at most one rollover
    can occur per file.

    Parameters
    ----------
    body : bytes
        File bytes to search for GGA/ZDA sentences. `.modraw` files carry
        the GPS stream past the header, but a caller may pass full-file
        bytes here too, since the header region never contains a
        `$GPGGA`/`$GPZDA`/`$INZDA` sentence.

    Returns
    -------
    ds : xr.Dataset
        Position fixes, dim `time`, data vars `lat` and `lon` (decimal
        degrees, signed: negative for S/W). Empty (zero-length `time`) if
        `body` has no `$GPGGA` sentences.

    Raises
    ------
    ValueError
        If `body` has `$GPGGA` sentences but no `$GPZDA`/`$INZDA` sentence
        to date them from.
    """
    gga_matches = list(_GGA.finditer(body))
    if not gga_matches:
        return xr.Dataset(
            coords={"time": ("time", np.array([], dtype="datetime64[ns]"))},
            data_vars={
                "lat": ("time", np.array([], dtype=float)),
                "lon": ("time", np.array([], dtype=float)),
            },
        )

    zda_matches = list(_ZDA.finditer(body))
    if not zda_matches:
        raise ValueError("$GPGGA sentences present but no $GPZDA/$INZDA sentence to date them from")

    zda_offsets = np.array([m.start() for m in zda_matches])
    zda_timestamps = []
    for match in zda_matches:
        hms, day, month, year = (g.decode() for g in match.groups())
        zda_timestamps.append(
            pd.Timestamp(f"{year}-{month}-{day} {hms[:2]}:{hms[2:4]}:{hms[4:]}")
        )

    times, lats, lons = [], [], []
    for match in gga_matches:
        nearest = np.argmin(np.abs(zda_offsets - match.start()))
        zda_ts = zda_timestamps[nearest]

        hms, lat_deg, lat_min, lat_hem, lon_deg, lon_min, lon_hem = match.groups()
        hms = hms.decode()
        time_of_day = pd.Timedelta(
            hours=int(hms[:2]), minutes=int(hms[2:4]), seconds=float(hms[4:])
        )
        candidate = zda_ts.normalize() + time_of_day
        diff = candidate - zda_ts
        if diff > pd.Timedelta(hours=12):
            candidate -= pd.Timedelta(days=1)
        elif diff < -pd.Timedelta(hours=12):
            candidate += pd.Timedelta(days=1)
        times.append(candidate)

        lat = int(lat_deg) + float(lat_min) / 60
        if lat_hem == b"S":
            lat = -lat
        lon = int(lon_deg) + float(lon_min) / 60
        if lon_hem == b"W":
            lon = -lon
        lats.append(lat)
        lons.append(lon)

    ds = xr.Dataset(
        coords={"time": ("time", pd.DatetimeIndex(times).values)},
        data_vars={
            "lat": ("time", np.array(lats)),
            "lon": ("time", np.array(lons)),
        },
    )
    ds.lat.attrs = dict(long_name="latitude", units="degrees_north")
    ds.lon.attrs = dict(long_name="longitude", units="degrees_east")
    return ds
