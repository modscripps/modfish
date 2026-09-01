#!/usr/bin/env python
# coding: utf-8
"""
Reading `.modraw` files written by the MOD acquisition software.

A `.modraw` file is ASCII. It opens with a header, whose length in bytes is
given on its very first line, and continues with a stream of blocks, one per
sensor, of the form

```
T<host ticks>$<TAG><16 hex timestamp><8 hex length>*<chk><data>*<chk>\\r\\n
```

Only the SBE49 CTD stream (`$SB49`) is decoded here. Its payload is a whole
number of 40-character records: a 16-hex millisecond timestamp followed by the
SBE49 output-format-24 frame, `TTTTTTCCCCCCPPPPPPtttt`, plus two trailing
characters that the Matlab reader discards.

This module follows
`MOD_fish_lib/EPSILOMETER/epsilib/mod_som_read_epsi_files_v4.m`, including its
SBE49 calibration polynomials, and adds a per-block checksum check that the
Matlab reader does not do.

Rudimentary on purpose; see
[issue #13](https://github.com/modscripps/modfish/issues/13).
"""

from .header import read_header, read_body, header_setup, sbe49_cal
from .sb49 import sbe49_to_physical, load_ctd, load_ctd_time_series
from .gps import load_gps_time
from .framer import block_counts
from .reader import read

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
]
