#!/usr/bin/env python
# coding: utf-8
"""Binary frame scanner for `.modraw` block streams.

A single-pass scanner over the SOM tag frame family described in the
[`.modraw` tag reference](https://github.com/modscripps/modfish/blob/main/modraw_tag_format.md):

```
$<TAG><16 hex timestamp><8 hex length>*<hh><payload>*<hh>\\r\\n
```

optionally preceded by a `T<10 ASCII digits>` laptop-clock prefix
(`SYSTEM_TIME` in centiseconds). The scanner decodes tag, timestamp,
payload and trailing checksum of every frame once. The trailing checksum is
the XOR of the payload bytes and is verified. The header checksum `*<hh>`
after the length field is checked for hex digits only. Resyncs and bad
checksums are tallied in `FrameStats`.

`ALTI` is an exception: the 8-char field that is normally a hex byte length
holds the distance reading as ASCII decimal text, the payload is empty, and
the frame ends with a single `*hh\\r\\n` directly after that field. This
matches the Matlab reader's non-greedy regex, which stops at the first
`*XX\\r\\n` it finds. The layout comes from the Matlab reader and is
unverified against real bytes, since no sampled MOTIVE `.modraw` file has
an `$ALTI` frame.

Pass the whole file to `frame`. A file's declared header length can fall
inside the block stream, so the bytes past it can miss leading `$SOM3`,
`$DCAL` and data frames.
"""

import dataclasses

from ._util import xor_checksum

_HEADER_LEN = 32  # $ + 4 tag + 16 ts + 8 length + * + 2 checksum
_TRAILER_LEN = 5  # * + 2 checksum + \r\n
_T_PREFIX_LEN = 11  # 'T' + 10 digits
_HEXDIGITS = frozenset(b"0123456789abcdefABCDEF")


@dataclasses.dataclass
class Packet:
    """One decoded SOM frame.

    Parameters
    ----------
    tag : str
        Four-character tag, e.g. "SB49", "EFE4".
    timestamp_ms : int
        The 16-hex header timestamp, ms (since 1970 for values above 1e9,
        since power-on below).
    laptop_ts_cs : int or None
        `T`-prefix laptop clock, centiseconds, None if absent.
    payload : bytes
        Frame payload. Empty for ALTI.
    length_field : bytes
        The raw 8 length characters; for ALTI they hold the reading.
    chk_ok : bool
        True when the trailing XOR checksum matches the payload. ALTI
        frames carry no payload checksum and are always True.
    """

    tag: str
    timestamp_ms: int
    laptop_ts_cs: int | None
    payload: bytes
    length_field: bytes
    chk_ok: bool = True


@dataclasses.dataclass
class FrameStats:
    """Counters from one `frame` scan.

    Parameters
    ----------
    n_frames : int
        Frames decoded.
    n_resync : int
        `$` anchors that did not start a valid frame. Interleaved NMEA
        sentences and header lines holding a `$` count here as well as
        frames with damaged framing, so a healthy file has many. `$`
        bytes inside the payload of a decoded frame are skipped and not
        counted.
    n_bad_checksum : int
        Frames whose trailing XOR checksum did not match.
    tag_counts : dict of str to int
        Frames per tag.
    """

    n_frames: int = 0
    n_resync: int = 0
    n_bad_checksum: int = 0
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

    chk_ok = trailer_chk_bytes.upper() == b"%02X" % xor_checksum(payload)

    frame_end = payload_end + _TRAILER_LEN
    laptop_ts_cs = _detect_t_prefix(body, dollar, pos)
    packet = Packet(
        tag=tag,
        timestamp_ms=timestamp_ms,
        laptop_ts_cs=laptop_ts_cs,
        payload=payload,
        length_field=length_bytes,
        chk_ok=chk_ok,
    )
    return packet, frame_end, chk_ok


def frame(body):
    """Scan bytes for SOM frames and decode every one found.

    Parameters
    ----------
    body : bytes
        Bytes to scan, normally the whole file including its ASCII header.
        The declared header length can fall inside the block stream, so
        scanning only the bytes past it can miss frames.

    Returns
    -------
    packets : list of Packet
        Every frame decoded, in stream order. Frames with a bad trailing
        checksum are included with `chk_ok` False; the decoders drop them.
    stats : FrameStats
        Counters describing the scan: frames found, rejected `$` anchors
        (`n_resync`), bad trailing checksums, and a per-tag count.
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

