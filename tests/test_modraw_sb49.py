#!/usr/bin/env python

"""Tests for `modfish.modraw.sb49.decode_sb49`, the framer-based SB49 decoder."""

import numpy as np
import pytest

from modfish.modraw.framer import Packet, frame
from modfish.modraw.header import read_body, read_header, sbe49_cal
from modfish.modraw.sb49 import _hex_columns, decode_sb49


def test_decode_sb49_matches_load_ctd(rootdir):
    import modfish

    file = rootdir / "data/FCTD_modraw_excerpt.modraw"
    packets, _ = frame(read_body(file))
    cal = sbe49_cal(read_header(file))
    ds = decode_sb49([p for p in packets if p.tag == "SB49"], cal)
    ref = modfish.modraw.load_ctd(file)
    assert ds.sizes["time"] == ref.sizes["time"]
    np.testing.assert_array_equal(ds.time.values, ref.time.values)
    np.testing.assert_allclose(ds.p.values, ref.p.values)
    np.testing.assert_allclose(ds.t.values, ref.t.values)
    np.testing.assert_allclose(ds.c.values, ref.c.values)


def _chars(*rows):
    return np.frombuffer(b"".join(rows), dtype=np.uint8).reshape(len(rows), -1)


def test_hex_columns_decodes_upper_and_lower_case():
    out = _hex_columns(_chars(b"00FF", b"00ff", b"0a1B"), 0, 4)
    np.testing.assert_array_equal(out, [255.0, 255.0, 0x0A1B])


def test_hex_columns_nan_only_on_rows_with_a_non_hex_character():
    out = _hex_columns(_chars(b"0010", b"00G0", b"\x00\xff10", b"0020"), 0, 4)
    assert np.isnan(out[1]) and np.isnan(out[2])
    np.testing.assert_array_equal(out[[0, 3]], [16.0, 32.0])


def test_hex_columns_sixteen_digit_timestamp_is_exact():
    out = _hex_columns(_chars(b"0000019afe339125"), 0, 16)
    assert out[0] == float(0x0000019AFE339125)


def _sb49_packet(*records):
    return Packet(
        tag="SB49",
        timestamp_ms=0,
        laptop_ts_cs=None,
        payload=b"".join(records),
        length_field=b"",
    )


def test_decode_sb49_stacks_records_across_packets(rootdir):
    file = rootdir / "data/FCTD_modraw_excerpt.modraw"
    packets, _ = frame(read_body(file))
    cal = sbe49_cal(read_header(file))
    sb49 = [p for p in packets if p.tag == "SB49"]
    # Every record of the stream in one packet against one packet per block.
    records = b"".join(p.payload for p in sb49)
    one = decode_sb49([_sb49_packet(records)], cal)
    many = decode_sb49(sb49, cal)
    assert one.sizes["time"] == many.sizes["time"]
    for v in ("time", "t_raw", "c_raw", "p_raw", "pt_raw"):
        np.testing.assert_array_equal(one[v].values, many[v].values)


def test_decode_sb49_bad_length_payload_is_skipped_and_counted(rootdir):
    file = rootdir / "data/FCTD_modraw_excerpt.modraw"
    packets, _ = frame(read_body(file))
    cal = sbe49_cal(read_header(file))
    sb49 = [p for p in packets if p.tag == "SB49"]
    bad = _sb49_packet(sb49[0].payload[:-1])
    ds = decode_sb49([sb49[0], bad, sb49[1]], cal)
    assert ds.attrs["n_bad_length"] == 1
    ref = decode_sb49(sb49[:2], cal)
    np.testing.assert_array_equal(ds.time.values, ref.time.values)


def test_decode_sb49_no_usable_packets_raises(rootdir):
    cal = sbe49_cal(read_header(rootdir / "data/FCTD_modraw_excerpt.modraw"))
    with pytest.raises(ValueError, match="no usable SB49"):
        decode_sb49([_sb49_packet(b"x" * 39)], cal)
