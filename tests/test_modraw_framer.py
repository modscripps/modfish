#!/usr/bin/env python

"""Tests for the `modfish.modraw.framer` single-pass binary scanner."""

import numpy as np
import pytest

from modfish.modraw.framer import frame


def make_frame(tag, timestamp_ms, payload, t_prefix=None):
    """Build one SOM frame with a correct trailing XOR checksum."""
    chk = 0
    for byte in payload:
        chk ^= byte
    head = b"" if t_prefix is None else b"T%010d" % t_prefix
    return (
        head
        + b"$" + tag
        + b"%016x" % timestamp_ms
        + b"%08x" % len(payload)
        + b"*00"                      # header checksum, hex-validated only
        + payload
        + b"*%02X\r\n" % chk
    )


def test_frame_single_block_decodes_fields():
    body = make_frame(b"SB49", 1723252616326, b"hello world!", t_prefix=12345)
    packets, stats = frame(body)
    assert len(packets) == 1
    p = packets[0]
    assert p.tag == "SB49"
    assert p.timestamp_ms == 1723252616326
    assert p.laptop_ts_cs == 12345
    assert p.payload == b"hello world!"
    assert stats.n_frames == 1
    assert stats.n_bad_checksum == 0


def test_frame_without_t_prefix_gives_none():
    body = make_frame(b"SOM3", 5, b"abc")
    packets, _ = frame(body)
    assert packets[0].laptop_ts_cs is None


def test_frame_binary_payload_with_dollar_inside_not_split():
    # A '$SB49' byte sequence inside a binary payload must not start a frame.
    evil = b"\x00$SB49" + b"\x01" * 30
    body = make_frame(b"EFE4", 7, evil) + make_frame(b"SB49", 8, b"x" * 40)
    packets, stats = frame(body)
    assert [p.tag for p in packets] == ["EFE4", "SB49"]
    assert stats.n_resync == 0


def test_frame_resyncs_after_garbage():
    body = b"garbage$notaframe" + make_frame(b"SB49", 9, b"y" * 40)
    packets, stats = frame(body)
    assert [p.tag for p in packets] == ["SB49"]
    assert stats.n_resync > 0


def test_frame_bad_trailing_checksum_is_tallied_not_dropped():
    body = bytearray(make_frame(b"SB49", 9, b"y" * 40))
    body[-4] = ord("9") if chr(body[-4]) != "9" else ord("8")  # corrupt checksum
    packets, stats = frame(bytes(body))
    assert len(packets) == 1
    assert stats.n_bad_checksum == 1


def test_frame_truncated_trailing_frame_is_skipped():
    body = make_frame(b"SB49", 9, b"y" * 40)
    packets, stats = frame(body[:-10])
    assert packets == []


@pytest.mark.skip(reason="ALTI layout unverified against real bytes")
def test_frame_alti_length_field_is_reading_not_length():
    # TODO: verify this hypothesis against a real $ALTI frame and un-skip.
    # Searched ~700 files across two MOTIVE cruises (skq202521s:
    # Raw_full_cruise, all 98 EPSI25 files plus the first ~300 FCTD25 files;
    # skq202417s: RAW_full_cruise, first 300 EPSI24 files) without finding a
    # single "$ALTI" occurrence. ALTI is documented as "on demand, 1
    # value/block" in modraw_tag_format.md, so it may simply be rare/absent
    # in these particular deployments. The ALTI branch in framer.py
    # implements the planning-time hypothesis below but is unverified.
    #
    # ALTI: the 8-char length slot holds the reading as ASCII decimal,
    # payload is empty, frame ends right after the header checksum.
    body = b"T0000012345" + b"$ALTI" + b"%016x" % 42 + b"00006670" + b"*1A\r\n"
    packets, stats = frame(body)
    assert len(packets) == 1
    assert packets[0].tag == "ALTI"
    assert packets[0].length_field == b"00006670"
    assert packets[0].payload == b""


def test_frame_tag_counts_accumulate():
    body = make_frame(b"SB49", 1, b"a") + make_frame(b"SB49", 2, b"b") + make_frame(b"ECOP", 3, b"c")
    _, stats = frame(body)
    assert stats.tag_counts == {"SB49": 2, "ECOP": 1}


def test_frame_fixture_matches_known_tag_counts(rootdir):
    # Deliberately reads the whole file, not `read_body()`'s output: the
    # fixture's own `header_file_size_inbytes` (7069) declares a boundary
    # that falls *inside* the block stream. `$SOM3` and `$DCAL` (the two
    # metadata blocks) plus the first two `$SB49`, one `$EFE4`, and eight
    # `$ECOP` data frames are physically written before that boundary, so
    # `read_body()` excludes real frames. Confirmed by hand: a plain
    # `re.findall(rb"\$SB49", ...)` over the raw file gives 238, over
    # `read_body()`'s output only 236, with the difference landing inside the
    # first 7069 bytes. See the framer.py module docstring.
    body = (rootdir / "data/FCTD_modraw_excerpt.modraw").read_bytes()
    _, stats = frame(body)
    assert stats.tag_counts["SB49"] == 238
    assert stats.tag_counts["EFE4"] == 120
    assert stats.tag_counts["ECOP"] == 960
    assert stats.tag_counts["SOM3"] == 1
    assert stats.tag_counts["DCAL"] == 1
    assert stats.n_bad_checksum == 0
