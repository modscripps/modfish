import struct

import numpy as np
import pytest

from modfish.modraw.efe import decode_efe4
from modfish.modraw.framer import frame
from modfish.modraw.header import parse_som3, read_body


@pytest.fixture
def efe_setup(rootdir):
    # Full file bytes: read_body swallows SOM3 and early EFE4 blocks.
    packets, _ = frame((rootdir / "data/FCTD_modraw_excerpt.modraw").read_bytes())
    meta = parse_som3(next(p for p in packets if p.tag == "SOM3").payload)
    return [p for p in packets if p.tag == "EFE4"], meta


def test_decode_efe4_first_record_matches_struct_decode(efe_setup):
    packets, meta = efe_setup
    n_ch = meta["n_channels"]
    ds = decode_efe4(packets, meta)

    rec = packets[0].payload[: 8 + 3 * n_ch]
    ts_ms = struct.unpack("<Q", rec[:8])[0]
    assert ds.time[0].values == np.datetime64(ts_ms, "ms").astype("datetime64[ns]")
    for i, name in enumerate(meta["channels"]):
        b0, b1, b2 = rec[8 + 3 * i : 11 + 3 * i]
        count = (b0 << 16) | (b1 << 8) | b2
        fr = meta["full_range"][i]
        if meta["adc_conf"][i] == "unipolar":
            volt = fr * count / 2**24
        else:
            volt = fr * (count / 2**23 - 1)
        assert ds[name][0].values == pytest.approx(volt)


def test_decode_efe4_sizes_and_rate(efe_setup):
    packets, meta = efe_setup
    ds = decode_efe4(packets, meta)
    assert ds.sizes["time"] == 80 * len(packets)
    dt = np.diff(ds.time.values).astype("timedelta64[ns]").astype(float) / 1e9
    # Byte-verified: this fixture's EFE runs 7 channels at ~320 Hz.
    assert np.median(dt) == pytest.approx(1 / 320, rel=0.05)


def test_decode_efe4_wrong_length_payload_skipped_and_tallied(efe_setup):
    packets, meta = efe_setup
    import dataclasses
    bad = dataclasses.replace(packets[0], payload=packets[0].payload[:-1])
    ds = decode_efe4([bad] + packets[1:], meta)
    assert ds.sizes["time"] == 80 * (len(packets) - 1)
    assert ds.attrs["n_bad_length"] == 1
