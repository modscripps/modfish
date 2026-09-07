"""VectorNav decoder for `.modraw` files.

`$VNAV` SOM frames carry batches of `$VNMAR` ASCII sentences, the VectorNav
magnetic, acceleration and angular-rate register. Each sentence is preceded
by its own 16-hex millisecond timestamp and ends with the NMEA XOR checksum
over the bytes between `$` and `*`. Observed at 40.0 Hz on both MOTIVE
cruises.
"""

import re

import numpy as np
import xarray as xr

from .framer import Packet

_VNMAR = re.compile(
    rb"([0-9a-fA-F]{16})\$(VNMAR,"
    rb"([+-]?\d+\.\d+),([+-]?\d+\.\d+),([+-]?\d+\.\d+),"
    rb"([+-]?\d+\.\d+),([+-]?\d+\.\d+),([+-]?\d+\.\d+),"
    rb"([+-]?\d+\.\d+),([+-]?\d+\.\d+),([+-]?\d+\.\d+))"
    rb"\*([0-9A-Fa-f]{2})"
)

_FIELDS = (
    ("mag_x", "Gauss"), ("mag_y", "Gauss"), ("mag_z", "Gauss"),
    ("accel_x", "m s-2"), ("accel_y", "m s-2"), ("accel_z", "m s-2"),
    ("gyro_x", "rad s-1"), ("gyro_y", "rad s-1"), ("gyro_z", "rad s-1"),
)


# Group numbering, verified against the fixture's first sentence: 1 = the
# 16-hex millisecond timestamp, 2 = the bytes between `$` and `*` (what the
# checksum covers), 3 through 11 = the nine fields in order, 12 = the checksum.


def _checksum_ok(sentence: bytes, stated: bytes) -> bool:
    ck = 0
    for byte in sentence:
        ck ^= byte
    return ck == int(stated, 16)


def decode_vnmar_bytes(body: bytes) -> xr.Dataset:
    """Decode every well-formed `$VNMAR` sentence in a byte string."""
    times, cols = [], [[] for _ in _FIELDS]
    for m in _VNMAR.finditer(body):
        if not _checksum_ok(m.group(2), m.group(12)):
            continue
        times.append(np.datetime64(int(m.group(1), 16), "ms").astype("datetime64[ns]"))
        for i in range(9):
            cols[i].append(float(m.group(3 + i)))
    coords = {"time": np.array(times, dtype="datetime64[ns]")}
    data = {
        name: ("time", np.array(col, dtype=float), {"units": units})
        for (name, units), col in zip(_FIELDS, cols)
    }
    ds = xr.Dataset(data, coords=coords)
    ds.attrs["source"] = "VNMAR"
    return ds


def decode_vnmar(packets: list[Packet]) -> xr.Dataset:
    """Decode framed `$VNAV` packets into a VectorNav time series."""
    return decode_vnmar_bytes(b"".join(p.payload for p in packets))
