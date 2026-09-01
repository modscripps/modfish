"""Decode EFE4 microconductivity and accelerometer packets."""

import numpy as np
import pandas as pd
import xarray as xr


def decode_efe4(packets, meta):
    """Decode EFE4 data packets into a time series Dataset.

    EFE4 record layout: 8-byte little-endian u64 millisecond timestamp +
    3-byte big-endian ADC counts per channel. 80 records per block.

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
        Dataset with dimension "time" (length 80 * n_valid_packets) and
        one data variable per channel named as in meta["channels"], with
        units "V". Attribute "n_bad_length" counts packets with
        incorrect payload length.

    Raises
    ------
    ValueError
        If no EFE4 packets have a valid payload length, or if timestamps
        appear to be power-on-relative (not yet supported).
    """
    n_ch = meta["n_channels"]
    rec = np.dtype([("ts", "<u8"), ("adc", "u1", (n_ch, 3))])
    block_len = meta["recs_per_block"] * rec.itemsize
    good = [p.payload for p in packets if len(p.payload) == block_len]
    n_bad = len(packets) - len(good)
    if not good:
        raise ValueError("no EFE4 packets with a valid payload length")
    raw = np.frombuffer(b"".join(good), dtype=rec)
    ts = raw["ts"]
    if np.median(ts) < 1e9:
        raise ValueError("power-on-relative EFE timestamps are not supported yet")
    time = pd.to_datetime(ts, unit="ms").values.astype("datetime64[ns]")
    adc = raw["adc"].astype("u4")
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
    ds.attrs["n_bad_length"] = n_bad
    return ds
