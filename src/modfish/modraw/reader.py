#!/usr/bin/env python
# coding: utf-8
"""
Assemble every decodable stream in a `.modraw` file into one `xr.DataTree`.

Unlike `modfish.modraw.sb49.load_ctd`, which frames only `read_body`'s
output, `read()` frames the *whole file*: the header's declared length can
fall inside the block stream rather than before it (see the caution in
`modfish.modraw.framer`), so scanning only past that boundary would silently
drop leading `$SOM3`/`$DCAL`/`$SB49`/`$EFE4`/`$ECOP` frames. `read()`'s CTD
group is therefore a superset of `load_ctd`'s: same records at the tail, plus
whatever sat before the (wrongly) declared header end.
"""

import collections
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from .alti import decode_alti
from .ecop import decode_ecop
from .efe import decode_efe4
from .framer import frame
from .gps import decode_gga
from .header import header_setup, parse_dcal, parse_som3, read_header, sbe49_cal
from .sb49 import decode_sb49


def read(file):
    """Read every decodable stream in a `.modraw` file into an `xr.DataTree`.

    Frames the full file (not `read_body`'s output) so that no block is
    excluded by a header length that ends up falling inside the block
    stream; see the module docstring and `modfish.modraw.framer` for why
    that matters.

    Parameters
    ----------
    file : Path or str
        Path to a `.modraw` file.

    Returns
    -------
    tree : xr.DataTree
        Root dataset (dim `block`) plus one group per stream with data:

        - `ctd` : SBE49 CTD time series (`decode_sb49`); calibration
          coefficients are stamped onto `tree["ctd"].attrs`.
        - `efe` : EFE4 microconductivity/accelerometer channels
          (`decode_efe4`), present only when both `$EFE4` and `$SOM3`
          blocks were found (the channel setup comes from `$SOM3`).
        - `ecop` : ECOP Tridente fluorometer time series (`decode_ecop`).
          Its time axis is not sorted or deduplicated here; the fixture's
          ~32 blocks/s stream carries duplicate timestamps, and decoders
          own their own axes.
        - `gps` : `$GPGGA` position fixes dated from `$GPZDA` (`decode_gga`),
          present only when the file has at least one GGA sentence.
        - `alti` : altimeter distance (`decode_alti`), present only when the
          file has `$ALTI` blocks.

        A group is present only when its stream has data; a file with none
        of the above raises `ValueError`.

        Root data variables, dim `block`, one entry per frame found by the
        scanner, in stream order, unsorted, undeduped, for clock forensics:

        - `laptop_time` : int64 centiseconds from the frame's `T`-prefix
          laptop-clock stamp, or -1 where the frame had none (kept integer
          rather than float so the missing marker cannot be confused with a
          real value near zero; see `comment` in its attrs).
        - `block_time` : datetime64[ns], the frame's own hex header
          timestamp.
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
        If none of the known streams (`ctd`, `efe`, `ecop`, `gps`, `alti`)
        yielded any data.
    """
    file = Path(file)
    head = read_header(file)
    # Frame the FULL file, not read_body: the declared header length can
    # overlap the block stream (see module docstring / framer caution).
    body = file.read_bytes()
    packets, stats = frame(body)
    by_tag = collections.defaultdict(list)
    for p in packets:
        by_tag[p.tag].append(p)

    cal = sbe49_cal(head)
    if not cal and "DCAL" in by_tag:
        cal = parse_dcal(by_tag["DCAL"][0].payload)

    groups = {}
    if by_tag.get("SB49") and cal:
        groups["ctd"] = decode_sb49(by_tag["SB49"], cal)
        groups["ctd"].attrs.update(cal)
    if by_tag.get("EFE4") and by_tag.get("SOM3"):
        meta = parse_som3(by_tag["SOM3"][0].payload)
        groups["efe"] = decode_efe4(by_tag["EFE4"], meta)
    if by_tag.get("ECOP"):
        groups["ecop"] = decode_ecop(by_tag["ECOP"])
    gga = decode_gga(body)
    if gga.sizes.get("time"):
        groups["gps"] = gga
    if by_tag.get("ALTI"):
        groups["alti"] = decode_alti(by_tag["ALTI"])

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
            block_time=("block", pd.to_datetime([p.timestamp_ms for p in packets], unit="ms").values),
            block_tag=("block", np.array([p.tag for p in packets])),
        ),
    )
    root.laptop_time.attrs = dict(
        long_name="laptop clock time",
        units="centiseconds",
        comment="-1 where the frame had no T-prefix laptop-clock stamp",
    )
    root.block_time.attrs = dict(long_name="block header timestamp")
    root.block_tag.attrs = dict(long_name="SOM frame tag")

    tree = xr.DataTree.from_dict({"/": root, **{f"/{k}": v for k, v in groups.items()}})
    tree.attrs = dict(
        file=file.name,
        n_frames=stats.n_frames,
        n_resync=stats.n_resync,
        n_bad_checksum=stats.n_bad_checksum,
        **{f"n_blocks_{k}": v for k, v in stats.tag_counts.items()},
        **header_setup(head),
    )
    return tree
