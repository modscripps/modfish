"""
Assemble every decodable stream in a `.modraw` file into one `xr.DataTree`.

`read()` frames the whole file: the header's declared length can fall inside
the block stream rather than before it (see `modfish.modraw.framer`), so
scanning only past that boundary would drop leading `$SOM3`, `$DCAL`,
`$SB49`, `$EFE4` and `$ECOP` frames.
"""

import collections
import logging
from pathlib import Path

import numpy as np
import xarray as xr

from ._util import ms_to_datetime
from .alti import decode_alti
from .ecop import decode_ecop
from .efe import decode_efe4
from .framer import frame
from .gps import decode_gga
from .header import (
    cal_complete,
    header_setup,
    parse_dcal,
    parse_som3,
    read_header,
    sbe49_cal,
)
from .sb49 import decode_sb49
from .vnav import decode_vnmar

logger = logging.getLogger(__name__)


def read(file):
    """Read every decodable stream in a `.modraw` file into an `xr.DataTree`.

    Frames the full file so that no block is excluded by a header length
    that falls inside the block stream; see `modfish.modraw.framer`. A
    decoder that fails on its stream (no usable payload, timestamps
    relative to power-on, GPS fixes without a date) logs a warning and its
    group is left out; the other streams are still returned.

    Parameters
    ----------
    file : Path or str
        Path to a `.modraw` file.

    Returns
    -------
    tree : xr.DataTree
        Root dataset (dim `block`) plus one group per stream with data:

        - `ctd` : SBE49 CTD time series (`decode_sb49`); calibration
          coefficients and header setup fields (see "Root attrs" below)
          are stamped onto `tree["ctd"].attrs`, so a consumer that opens
          the `ctd` group on its own (as the L0 pipeline does) still sees
          the instrument's `serialnum` and the rest.
        - `efe` : EFE4 microconductivity/accelerometer channels
          (`decode_efe4`), present only when both `$EFE4` and `$SOM3`
          blocks were found (the channel setup comes from `$SOM3`).
        - `ecop` : ECOP Tridente fluorometer time series (`decode_ecop`).
          Its time axis is not sorted or deduplicated here; the fixture's
          ~32 blocks/s stream carries duplicate timestamps, and decoders
          own their own axes.
        - `gps` : `$GPGGA`/`$INGGA` position fixes dated from
          `$GPZDA`/`$INZDA` (`decode_gga`), present only when the file has
          at least one GGA sentence.
        - `alti` : altimeter distance (`decode_alti`), present only when the
          file has `$ALTI` blocks.
        - `vnav` : VectorNav mag/accel/gyro time series (`decode_vnmar`),
          decoded from `$VNMAR` sentences carried in `$VNAV` blocks, present
          only when at least one `$VNMAR` sentence decoded.

        A group is present only when its stream has data. Each group's
        attrs carry its decoder's drop counts (`n_bad_checksum`,
        `n_bad_length`, `n_bad_time`). A file with none of the above
        raises `ValueError`.

        Root data variables, dim `block`, one entry per frame found by the
        scanner, in stream order, unsorted, undeduped, for clock forensics:

        - `laptop_time` : int64 centiseconds from the frame's `T`-prefix
          laptop-clock stamp, or -1 where the frame had none (kept integer
          rather than float so the missing marker cannot be confused with a
          real value near zero; see `comment` in its attrs).
        - `block_time` : datetime64[ns], the frame's own hex header
          timestamp; NaT where it lies outside the datetime64[ns] range (a
          corrupt but otherwise well-framed block, e.g. one flipped hex
          digit).
        - `block_tag` : str, the frame's SOM tag (`"SB49"`, `"EFE4"`, ...).

        Root attrs:

        - `file` : the input file's name.
        - `n_frames`, `n_resync`, `n_bad_checksum` : from `FrameStats`.
          `n_resync` counts failed frame attempts, not corruption: a
          whole-file scan legitimately includes every interleaved NMEA
          sentence and header text line, none of which are SOM frames, so
          the scanner correctly rejects and skips past them. File health is
          judged from `n_bad_checksum` and the decoders' own
          `n_bad_length` tallies, not from `n_resync`.
        - `n_blocks_<TAG>` : per-tag frame counts, flattened from
          `FrameStats.tag_counts`.
        - header setup fields from `header_setup` (survey, experiment,
          cruise, vehicle, fishflag, serialnum, gm_time; whichever are
          present in the header).

    Raises
    ------
    ValueError
        If none of the known streams (`ctd`, `efe`, `ecop`, `gps`, `alti`,
        `vnav`) yielded any data.
    """
    file = Path(file)
    head = read_header(file)
    body = file.read_bytes()
    packets, stats = frame(body)
    by_tag = collections.defaultdict(list)
    for p in packets:
        by_tag[p.tag].append(p)

    cal = sbe49_cal(head)
    if not cal_complete(cal) and "DCAL" in by_tag:
        cal = parse_dcal(by_tag["DCAL"][0].payload)
    setup = header_setup(head)

    decoders = {}
    if by_tag.get("SB49"):
        if cal_complete(cal):
            decoders["ctd"] = lambda: decode_sb49(by_tag["SB49"], cal).assign_attrs(cal, **setup)
        else:
            logger.warning(
                "file has %d SB49 blocks but no complete set of calibration "
                "coefficients; ctd group omitted",
                len(by_tag["SB49"]),
            )
    if by_tag.get("EFE4"):
        if by_tag.get("SOM3"):
            decoders["efe"] = lambda: decode_efe4(
                by_tag["EFE4"], parse_som3(by_tag["SOM3"][0].payload))
        else:
            logger.warning(
                "file has %d EFE4 blocks but no SOM3 channel setup; efe group omitted",
                len(by_tag["EFE4"]),
            )
    if by_tag.get("ECOP"):
        decoders["ecop"] = lambda: decode_ecop(by_tag["ECOP"])
    decoders["gps"] = lambda: decode_gga(body)
    if by_tag.get("ALTI"):
        decoders["alti"] = lambda: decode_alti(by_tag["ALTI"])
    if by_tag.get("VNAV"):
        decoders["vnav"] = lambda: decode_vnmar(by_tag["VNAV"])

    groups = {}
    for name, decode in decoders.items():
        try:
            ds = decode()
        except ValueError as exc:
            logger.warning("%s: %s group omitted: %s", file.name, name, exc)
            continue
        if ds.sizes.get("time", 0):
            groups[name] = ds

    if not groups:
        raise ValueError(f"no decodable data streams in {file}")

    root = xr.Dataset(
        data_vars=dict(
            laptop_time=(
                "block",
                np.array(
                    [p.laptop_ts_cs if p.laptop_ts_cs is not None else -1 for p in packets],
                    dtype="int64",
                ),
            ),
            block_time=(
                "block",
                ms_to_datetime(np.array([p.timestamp_ms for p in packets], dtype="uint64")),
            ),
            block_tag=("block", np.array([p.tag for p in packets])),
        ),
    )
    root.laptop_time.attrs = dict(
        long_name="laptop clock time",
        units="centiseconds",
        comment="-1 where the frame had no T-prefix laptop-clock stamp",
    )
    root.block_time.attrs = dict(
        long_name="block header timestamp",
        comment="NaT where the hex timestamp lies outside the datetime64[ns] range",
    )
    root.block_tag.attrs = dict(long_name="SOM frame tag")

    tree = xr.DataTree.from_dict({"/": root, **{f"/{k}": v for k, v in groups.items()}})
    tree.attrs = dict(
        file=file.name,
        n_frames=stats.n_frames,
        n_resync=stats.n_resync,
        n_bad_checksum=stats.n_bad_checksum,
        **{f"n_blocks_{k}": v for k, v in stats.tag_counts.items()},
        **setup,
    )
    return tree
