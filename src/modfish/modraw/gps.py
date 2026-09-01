#!/usr/bin/env python
# coding: utf-8
"""
GPS time extraction from `.modraw` files.

Parses GPZDA sentences from the GPS stream to extract absolute UTC timestamps.
"""

import re

import pandas as pd

from .header import read_body


#: `$GPZDA,hhmmss.ss,dd,mm,yyyy`, the GPS UTC time sentence.
_ZDA = re.compile(rb"\$GPZDA,(\d{6}\.\d+),(\d{2}),(\d{2}),(\d{4})")


def load_gps_time(file):
    """Read the GPS UTC timestamps interleaved in a .modraw file.

    Useful as an absolute reference against the acquisition computer's clock,
    which is what the file names and the header `GM_TIME` are based on.

    Parameters
    ----------
    file : Path or str
        Path to a .modraw file.

    Returns
    -------
    time : pd.DatetimeIndex
        UTC times from the `$GPZDA` sentences.
    """
    out = []
    for match in _ZDA.finditer(read_body(file)):
        hms, day, month, year = (g.decode() for g in match.groups())
        out.append(pd.Timestamp(f"{year}-{month}-{day} {hms[:2]}:{hms[2:4]}:{hms[4:]}"))
    return pd.DatetimeIndex(out)
