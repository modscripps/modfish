import numpy as np
import pytest

from modfish.modraw.ecop import decode_ecop
from modfish.modraw.framer import frame
from modfish.modraw.header import read_body


@pytest.fixture
def ecop_packets(rootdir):
    # Full file bytes: read_body swallows the earliest ECOP blocks, and the
    # hand-verified first record below is the file's true first ECOP.
    packets, _ = frame((rootdir / "data/FCTD_modraw_excerpt.modraw").read_bytes())
    return [p for p in packets if p.tag == "ECOP"]


def test_decode_ecop_first_record_hand_computed(ecop_packets):
    assert ecop_packets[0].payload == b"0000019afe3391257FFF80098087"
    ds = decode_ecop(ecop_packets)
    assert ds.time[0].values == np.datetime64(0x0000019AFE339125, "ms").astype("datetime64[ns]")
    assert ds.bb_raw[0] == 0x7FFF
    assert ds.bb[0].values == pytest.approx((0x7FFF / 65535 - 0.5) / 0.05)
    assert ds.chla[0].values == pytest.approx((0x8009 / 65535 - 0.5) / 50)
    assert ds.fdom[0].values == pytest.approx((0x8087 / 65535 - 0.5) / 1000)


def test_decode_ecop_all_blocks_one_record_each(ecop_packets):
    # The Matlab reader hardcodes sample_freq = 16 Hz for ECOP, but the
    # MOTIVE 2025 fixture carries ~32 blocks/s with some duplicated
    # timestamps; assert the observed data behavior, not the Matlab constant.
    ds = decode_ecop(ecop_packets)
    assert ds.sizes["time"] == len(ecop_packets)  # 960 in the fixture
    dt = np.diff(ds.time.values).astype("timedelta64[ns]").astype(float) / 1e9
    assert np.median(dt) == pytest.approx(0.031, rel=0.1)
    assert (dt >= 0).all()  # sorted, duplicates allowed


def test_decode_ecop_garbage_payload_produces_nan(ecop_packets):
    # Regression test: garbage payloads (binary instead of ASCII hex) should
    # produce NaN rows, not zeros, so they are distinguishable from legitimate
    # low readings downstream.
    ds = decode_ecop(ecop_packets)
    # Packet at index 165 is known to have binary garbage data
    assert np.isnan(ds.bb_raw[165].values)
    assert np.isnan(ds.chla_raw[165].values)
    assert np.isnan(ds.fdom_raw[165].values)
    assert np.isnan(ds.bb[165].values)
    assert np.isnan(ds.chla[165].values)
    assert np.isnan(ds.fdom[165].values)
