"""Builders for synthetic `.modraw` byte streams, shared by the modraw tests."""

import functools
import pathlib

from modfish.modraw.framer import frame

FIXTURE_FCTD = pathlib.Path(__file__).parent / "data/FCTD_modraw_excerpt.modraw"


def make_frame(tag, timestamp_ms, payload, t_prefix=None, chk=None):
    """Build one SOM frame.

    The trailing XOR checksum is computed from `payload` unless `chk` is
    given, which lets a test write a frame whose checksum does not match.
    """
    if chk is None:
        chk = 0
        for byte in payload:
            chk ^= byte
    head = b"" if t_prefix is None else b"T%010d" % t_prefix
    return (
        head
        + b"$" + tag
        + b"%016x" % timestamp_ms
        + b"%08x" % len(payload)
        + b"*00"  # header checksum, hex-validated only
        + payload
        + b"*%02X\r\n" % chk
    )


def minimal_header(nbytes=200, text=b""):
    """A header `read_header`/`header_setup`/`sbe49_cal` can parse.

    Declares its own length on the first line, then `text`, then filler
    with no `$`, no calibration keys and no `SOM3`/`DCAL` tag. The real
    fixture's declared header span falls inside its block stream and so
    carries genuine frames and calibration, which would defeat the tests
    that need them absent.
    """
    first_line = f"header_file_size_inbytes = {nbytes}\n".encode()
    filler = b"X" * (nbytes - len(first_line) - len(text))
    header = first_line + text + filler
    assert len(header) == nbytes
    return header


@functools.lru_cache(maxsize=1)
def fixture_packets():
    """Packets of the FCTD excerpt grouped by tag, `{tag: [Packet, ...]}`."""
    packets, _ = frame(FIXTURE_FCTD.read_bytes())
    out = {}
    for p in packets:
        out.setdefault(p.tag, []).append(p)
    return out


def reframe(packets):
    """Concatenate `packets` back into frame bytes with valid checksums."""
    return b"".join(make_frame(p.tag.encode(), p.timestamp_ms, p.payload) for p in packets)
