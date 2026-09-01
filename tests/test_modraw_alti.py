import numpy as np
import pytest

from modfish.modraw.alti import decode_alti
from modfish.modraw.framer import Packet


def _alti_packet(ts_ms, reading):
    return Packet(tag="ALTI", timestamp_ms=ts_ms, laptop_ts_cs=None,
                  payload=b"", length_field=reading)


def test_decode_alti_hand_computed():
    ds = decode_alti([_alti_packet(1723252616326, b"00006670")])
    assert ds.dst[0].values == pytest.approx(6670 * 1e-5 * 1500)  # 100.05 m
    assert ds.time[0].values == np.datetime64(1723252616326, "ms").astype("datetime64[ns]")


def test_decode_alti_unparseable_reading_is_nan():
    ds = decode_alti([_alti_packet(1723252616326, b"++++++++")])
    assert np.isnan(ds.dst[0])
