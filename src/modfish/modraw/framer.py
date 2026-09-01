#!/usr/bin/env python
# coding: utf-8
"""Binary frame scanner for `.modraw` block streams.

A single-pass scanner over the "SOM tag frame" family described in
`modraw_tag_format.md`:

```
$<TAG><16 hex timestamp><8 hex length>*<hh><payload>*<hh>\\r\\n
```

optionally preceded by a `T<10 ASCII digits>` laptop-clock prefix
(`SYSTEM_TIME` in centiseconds). Replaces the old per-tag regex counting in
favor of one scanner that decodes every field once: tag, timestamp, payload,
and the trailing XOR checksum, while tallying resyncs and bad checksums
instead of silently dropping them.

`ALTI` is a documented exception (see `modraw_tag_format.md`, "Per-tag DATA
payload"): the 8-char field that is normally a hex byte-length instead holds
the distance reading as ASCII decimal text, the payload is empty, and the
frame ends with a single `*hh\\r\\n` directly after that field (matching
Matlab's non-greedy regex, which stops at the first `*XX\\r\\n` it finds).

This layout is a planning-time hypothesis, UNVERIFIED against real bytes: a
search of ~700 `.modraw` files across two MOTIVE cruises turned up no `$ALTI`
frame (see `test_frame_alti_length_field_is_reading_not_length`, skipped, in
`tests/test_modraw_framer.py`). ALTI is documented as "on demand, 1
value/block", so it may simply be absent from the deployments searched so
far.

Caution for callers: `header.read_body()` is not a safe substitute for the
whole file when you want every SOM frame. In `tests/data/FCTD_modraw_excerpt.
modraw`, `header_file_size_inbytes` (7069) declares a boundary that falls
*inside* the block stream, not before it: `$SOM3` and `$DCAL` (the two
one-off metadata blocks), plus the first two `$SB49`, one `$EFE4`, and eight
`$ECOP` data frames, are all written before that boundary and so are excluded
by `read_body()`. `frame(read_body(file))` on this fixture finds SB49=236,
EFE4=119, ECOP=952, and no SOM3/DCAL at all (152 fewer, when you add up
those 5 tags, than the whole-file counts of 238/120/960/1/1); `frame()` on
the raw file bytes finds every one of them with zero bad checksums.
Confirmed by hand (`re.findall(rb"\\$SB49", raw)` == 238 vs. 236 on
`read_body()`'s output). Anyone parsing `$SOM3`/`$DCAL` metadata via
`frame()` should scan the raw file, not `read_body()`'s output, or should
first confirm the file's header truly ends before its first block.
"""

import dataclasses

_HEADER_LEN = 32  # $ + 4 tag + 16 ts + 8 length + * + 2 checksum
_TRAILER_LEN = 5  # * + 2 checksum + \r\n
_T_PREFIX_LEN = 11  # 'T' + 10 digits
_HEXDIGITS = frozenset(b"0123456789abcdefABCDEF")


@dataclasses.dataclass
class Packet:
    tag: str  # e.g. "SB49", "EFE4"
    timestamp_ms: int  # 16-hex header timestamp
    laptop_ts_cs: int | None  # T-prefix, centiseconds, None if absent
    payload: bytes  # empty for ALTI
    length_field: bytes  # raw 8 length chars; holds the reading for ALTI


@dataclasses.dataclass
class FrameStats:
    n_frames: int = 0
    n_resync: int = 0  # failed frame attempts (rejected $ anchors), not bytes skipped
    n_bad_checksum: int = 0  # trailing XOR mismatches
    tag_counts: dict[str, int] = dataclasses.field(default_factory=dict)


def _is_hex(chunk):
    return all(b in _HEXDIGITS for b in chunk)


def _detect_t_prefix(body, dollar, pos):
    """Look back 11 bytes from `dollar` for a `T<10-digit>` prefix.

    Only accepted if it lies entirely at or past `pos` (the previous frame's
    end), so a `T` byte inside a binary payload already consumed by an
    earlier frame is never misread as a prefix.
    """
    start = dollar - _T_PREFIX_LEN
    if start < pos:
        return None
    prefix = body[start:dollar]
    if prefix[0:1] != b"T":
        return None
    digits = prefix[1:]
    if not all(48 <= b <= 57 for b in digits):
        return None
    return int(digits)


def _try_frame_at(body, dollar, pos):
    n = len(body)
    if dollar + _HEADER_LEN > n:
        return None

    tag_bytes = body[dollar + 1 : dollar + 5]
    if not all(
        (65 <= b <= 90) or (97 <= b <= 122) or (48 <= b <= 57) or b == 32
        for b in tag_bytes
    ):
        return None
    tag = tag_bytes.decode("ascii")

    ts_bytes = body[dollar + 5 : dollar + 21]
    if not _is_hex(ts_bytes):
        return None
    timestamp_ms = int(ts_bytes, 16)

    length_bytes = body[dollar + 21 : dollar + 29]
    header_chk_start = dollar + 29

    if tag.startswith("ALT"):
        # ALTI: the length field holds the ASCII reading, not a hex length.
        # Payload is empty; the frame ends right after the header checksum
        # slot with a single trailing `*hh\r\n`.
        trailer_start = header_chk_start
        if trailer_start + _TRAILER_LEN > n:
            return None
        if body[trailer_start : trailer_start + 1] != b"*":
            return None
        chk_bytes = body[trailer_start + 1 : trailer_start + 3]
        if not _is_hex(chk_bytes):
            return None
        if body[trailer_start + 3 : trailer_start + 5] != b"\r\n":
            return None
        frame_end = trailer_start + _TRAILER_LEN
        laptop_ts_cs = _detect_t_prefix(body, dollar, pos)
        packet = Packet(
            tag=tag,
            timestamp_ms=timestamp_ms,
            laptop_ts_cs=laptop_ts_cs,
            payload=b"",
            length_field=length_bytes,
        )
        return packet, frame_end, True

    if not _is_hex(length_bytes):
        return None
    if body[header_chk_start : header_chk_start + 1] != b"*":
        return None
    if not _is_hex(body[header_chk_start + 1 : header_chk_start + 3]):
        return None
    length = int(length_bytes, 16)

    payload_start = dollar + _HEADER_LEN
    payload_end = payload_start + length
    if payload_end + _TRAILER_LEN > n:
        return None
    payload = body[payload_start:payload_end]

    if body[payload_end : payload_end + 1] != b"*":
        return None
    trailer_chk_bytes = body[payload_end + 1 : payload_end + 3]
    if not _is_hex(trailer_chk_bytes):
        return None
    if body[payload_end + 3 : payload_end + 5] != b"\r\n":
        return None

    chk = 0
    for byte in payload:
        chk ^= byte
    chk_ok = trailer_chk_bytes.upper() == b"%02X" % chk

    frame_end = payload_end + _TRAILER_LEN
    laptop_ts_cs = _detect_t_prefix(body, dollar, pos)
    packet = Packet(
        tag=tag,
        timestamp_ms=timestamp_ms,
        laptop_ts_cs=laptop_ts_cs,
        payload=payload,
        length_field=length_bytes,
    )
    return packet, frame_end, chk_ok


def frame(body):
    """Scan a `.modraw` block stream and decode every SOM frame in it.

    Parameters
    ----------
    body : bytes
        File contents past the header, as returned by
        `modfish.modraw.header.read_body`.

    Returns
    -------
    packets : list of Packet
        Every frame decoded, in stream order. Frames with a bad trailing
        checksum are still included; see `stats.n_bad_checksum`.
    stats : FrameStats
        Counters describing the scan: frames found, failed frame attempts
        (rejected `$` anchors, `n_resync`), bad checksums, and a per-tag
        count.
    """
    packets = []
    stats = FrameStats()
    pos = 0
    n = len(body)
    while True:
        dollar = body.find(b"$", pos)
        if dollar < 0 or dollar + _HEADER_LEN > n:
            break
        parsed = _try_frame_at(body, dollar, pos)  # pos guards the T-prefix lookback
        if parsed is None:
            stats.n_resync += 1
            pos = dollar + 1
            continue
        packet, end, chk_ok = parsed
        if not chk_ok:
            stats.n_bad_checksum += 1
        packets.append(packet)
        stats.n_frames += 1
        stats.tag_counts[packet.tag] = stats.tag_counts.get(packet.tag, 0) + 1
        pos = end
    return packets, stats


def block_counts(file, tags=("SB49", "EFE4", "VNAV", "VNMAR", "ECOP", "ALTI", "GPZDA")):
    """Count the blocks of each stream in a .modraw file.

    A quick way to see which sensors were writing to a file, and at what rate.

    Scans only `header.read_body()`'s output, i.e. past the file's declared
    header span. That declared span can fall *inside* the block stream
    rather than before it (see the caution in this module's docstring), in
    which case frames written before the boundary are excluded here. Counts
    returned by this function can therefore come out lower than the
    `n_blocks_<TAG>` attrs `modfish.modraw.reader.read()` reports for the
    same file, since `read()` frames the whole file instead of
    `read_body()`'s output.

    Parameters
    ----------
    file : Path or str
        Path to a .modraw file.
    tags : iterable of str, optional
        Stream tags to count.

    Returns
    -------
    counts : dict
        Number of occurrences of each tag.
    """
    from modfish.modraw.header import read_body

    body = read_body(file)
    _, stats = frame(body)
    counts = {tag: stats.tag_counts.get(tag, 0) for tag in tags}
    # NMEA-style sentences do not use the SOM frame; count them by search.
    for tag in tags:
        if tag in ("VNAV", "VNMAR", "GPZDA", "GPGGA", "INGGA"):
            counts[tag] = body.count(b"$" + tag.encode())
    return counts
