"""Decode EFE4 microconductivity and accelerometer packets."""

import numpy as np
import xarray as xr

from ._util import ms_to_datetime


def decode_efe4(packets, meta):
    """Decode EFE4 data packets into a time series Dataset.

    EFE4 record layout: 8-byte little-endian u64 millisecond timestamp +
    3-byte big-endian ADC counts per channel, `meta["recs_per_block"]`
    records per block (80 on MOTIVE FCTD files).

    Voltage conversion:
    - Unipolar: volts = full_range * counts / 2**24
    - Bipolar: volts = full_range * (counts / 2**23 - 1)

    Parameters
    ----------
    packets : list[Packet]
        List of Packet objects with tag="EFE4". Each Packet has
        attributes: tag, timestamp_ms, laptop_ts_cs, payload, length_field.
    meta : dict
        Metadata dict from parse_som3 with keys: n_channels, channels,
        adc_conf, full_range, recs_per_block.

    Returns
    -------
    xr.Dataset
        Dimension "time" (`recs_per_block` per kept packet, less dropped
        records) and one data variable per channel named as in
        `meta["channels"]`, units "V". Attrs count what was dropped:
        `n_bad_checksum` packets whose trailing checksum failed (binary
        counts give no other sign of corruption), `n_bad_length` packets
        with the wrong payload length, and `n_bad_time` records whose
        timestamp lies outside the datetime64[ns] range.

    Raises
    ------
    ValueError
        If no EFE4 packets have a valid payload length, or if timestamps
        appear to be power-on-relative (not yet supported).
    """
    n_ch = meta["n_channels"]
    rec = np.dtype([("ts", "<u8"), ("adc", "u1", (n_ch, 3))])
    block_len = meta["recs_per_block"] * rec.itemsize
    checked = [p for p in packets if p.chk_ok]
    n_bad_checksum = len(packets) - len(checked)
    good = [p.payload for p in checked if len(p.payload) == block_len]
    n_bad = len(checked) - len(good)
    if not good:
        raise ValueError("no EFE4 packets with a valid payload length")
    raw = np.frombuffer(b"".join(good), dtype=rec)
    ts = raw["ts"]
    if np.median(ts) < 1e9:
        raise ValueError("power-on-relative EFE timestamps are not supported yet")
    time = ms_to_datetime(ts)
    keep = ~np.isnat(time)
    n_bad_time = int((~keep).sum())
    time = time[keep]
    adc = raw["adc"][keep].astype("u4")
    counts = (adc[:, :, 0] << 16) | (adc[:, :, 1] << 8) | adc[:, :, 2]

    data_vars = {}
    for i, name in enumerate(meta["channels"]):
        fr = meta["full_range"][i]
        if meta["adc_conf"][i] == "unipolar":
            volt = fr * counts[:, i] / 2**24
        else:
            volt = fr * (counts[:, i] / 2**23 - 1.0)
        data_vars[name] = ("time", volt)
    ds = xr.Dataset(coords={"time": ("time", time)}, data_vars=data_vars)
    for name in meta["channels"]:
        ds[name].attrs = dict(long_name=f"EFE channel {name}", units="V")
    ds.attrs = dict(n_bad_checksum=n_bad_checksum, n_bad_length=n_bad,
                    n_bad_time=n_bad_time)
    return ds
