#!/usr/bin/env python
# coding: utf-8
"""
Reading `.modraw` files written by the MOD acquisition software.

A `.modraw` file opens with an ASCII header, whose declared length in bytes
sits on its first line, and continues with a stream of blocks, one per
sensor. Most blocks follow the SOM tag frame:

```
T<host ticks>$<TAG><16 hex timestamp><8 hex length>*<chk><data>*<chk>\\r\\n
```

GPS sentences (`$GPGGA`/`$INGGA`, `$GPZDA`/`$INZDA`) are plain NMEA text
interleaved in the same stream, not SOM frames. Full byte-level layout for
every tag, including the corrections logged below, lives in
`modraw_tag_format.md` at the repo root.

The header's declared length is not a safe boundary to skip to: on a MOTIVE
2025 fixture its declared size (7069 bytes) falls inside the block stream,
ahead of `$SB49`, `$EFE4`, and `$ECOP` frames still to come. `read()`
therefore frames the whole file instead of trusting the header.

## Streams and groups

| Group | Tag(s) | Sensor | Rate |
|---|---|---|---|
| `ctd` | `$SB49` | SBE49 CTD (t, c, p) | 16 Hz |
| `efe` | `$EFE4` | shear/FP07/accelerometer, raw ADC | ~320 Hz on MOTIVE 2025 FCTD files (7 channels: `t1 t2 f1 c1 a1 a2 a3`, all unipolar) |
| `ecop` | `$ECOP` | Tridente fluorometer/backscatter | ~32 blocks/s on MOTIVE 2025 files, not the 16 Hz the Matlab reader assumes; timestamps duplicate and some payloads are binary garbage |
| `gps` | `$GPGGA`/`$INGGA` dated by `$GPZDA`/`$INZDA` | position fixes | on demand |
| `alti` | `$ALTI` | altimeter distance | on demand; frame layout is unverified, no `$ALTI` frame found in any sampled MOTIVE file |

A group is present only when its stream has data. `ctd` also needs
calibration coefficients, from the header or a `$DCAL` block.

## Example

```python
from modfish.modraw import read

tree = read("EPSI25_08_12_120000.modraw")
tree["ctd"].ds.p.plot()
```

## Quality tally

`read()` stamps scan-level counters onto `tree.attrs`: `n_frames`,
`n_resync`, `n_bad_checksum`, and one `n_blocks_<TAG>` per tag found.
`n_resync` counts bytes skipped while hunting for the next valid frame, not
corruption: a whole-file scan runs into every interleaved NMEA sentence and
header text line, none of which are SOM frames, and the scanner correctly
skips past them. Judge file health from `n_bad_checksum` and each decoder's
own `n_bad_length` (on the per-group dataset's `attrs`), not from
`n_resync`.

This module follows
`MOD_fish_lib/EPSILOMETER/epsilib/mod_som_read_epsi_files_v4.m`, including
its SBE49 calibration polynomials, and adds a per-block checksum check the
Matlab reader does not do. See `modraw_tag_format.md` for the full tag
reference this implementation is built from.
"""

from .header import read_header, read_body, header_setup, sbe49_cal
from .sb49 import sbe49_to_physical, load_ctd, load_ctd_time_series
from .gps import load_gps_time
from .framer import block_counts
from .reader import read
from .convert import convert

__all__ = [
    "read_header",
    "read_body",
    "header_setup",
    "sbe49_cal",
    "sbe49_to_physical",
    "load_ctd",
    "load_ctd_time_series",
    "load_gps_time",
    "block_counts",
    "read",
    "convert",
]
