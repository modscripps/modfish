"""VectorNav decoder for `.modraw` files.

`$VNAV` SOM frames carry batches of `$VNMAR` ASCII sentences, the VectorNav
magnetic, acceleration and angular-rate register. Each sentence is preceded
by its own 16-hex millisecond timestamp and ends with the NMEA XOR checksum
over the bytes between `$` and `*`. Observed at 40.0 Hz (25 ms median
sentence spacing) on MOTIVE 2024 and 2025 files, ten sentences per frame.
Other VectorNav registers (`$VNYPR`) are not decoded.

Payloads are joined before scanning. Every `$VNAV` payload seen so far
starts with a full timestamp and `$VNMAR` and ends with `\\r\\n`, so no
sentence is split across frames.
"""

import re

import numpy as np
import xarray as xr

from ._util import ms_to_datetime, xor_checksum
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


# Group numbering: 1 = the 16-hex millisecond timestamp, 2 = the bytes
# between `$` and `*` (what the checksum covers), 3 through 11 = the nine
# fields in order, 12 = the checksum.


def decode_vnmar_bytes(body: bytes) -> xr.Dataset:
    """Decode every well-formed `$VNMAR` sentence in a byte string.

    Parameters
    ----------
    body : bytes
        Bytes holding `<16 hex ms>$VNMAR,...*hh` sentences.

    Returns
    -------
    xr.Dataset
        Dim `time`, the nine `_FIELDS` variables (magnetic field in Gauss,
        acceleration in m s-2, angular rate in rad s-1). Sentences with a
        bad NMEA checksum or a timestamp outside the datetime64[ns] range
        are dropped and counted in `attrs["n_bad_checksum"]` and
        `attrs["n_bad_time"]`. Empty when `body` holds no `$VNMAR`.
    """
    ms, cols = [], [[] for _ in _FIELDS]
    n_bad_checksum = 0
    for m in _VNMAR.finditer(body):
        if xor_checksum(m.group(2)) != int(m.group(12), 16):
            n_bad_checksum += 1
            continue
        ms.append(int(m.group(1), 16))
        for i in range(9):
            cols[i].append(float(m.group(3 + i)))
    time = ms_to_datetime(np.array(ms, dtype="uint64"))
    keep = ~np.isnat(time)
    data = {
        name: ("time", np.array(col, dtype=float)[keep], {"units": units})
        for (name, units), col in zip(_FIELDS, cols, strict=True)
    }
    ds = xr.Dataset(data, coords={"time": time[keep]})
    ds.attrs = dict(source="VNMAR", n_bad_checksum=n_bad_checksum,
                    n_bad_time=int((~keep).sum()))
    return ds


def decode_vnmar(packets: list[Packet]) -> xr.Dataset:
    """Decode framed `$VNAV` packets into a VectorNav time series.

    Parameters
    ----------
    packets : list of Packet
        `$VNAV` packets from `modfish.modraw.framer.frame`. The frame
        checksum is not used: each sentence carries its own.

    Returns
    -------
    xr.Dataset
        See `decode_vnmar_bytes`.
    """
    return decode_vnmar_bytes(b"".join(p.payload for p in packets))
