#!/usr/bin/env python
# coding: utf-8
"""
Header parsing and setup extraction from `.modraw` files.

This module handles the ASCII header format of MOD `.modraw` files and extracts
calibration coefficients for the SBE49 CTD.
"""

import re
from pathlib import Path


#: SBE49 calibration coefficients as they are named in the header.
_CAL_KEYS = (
    "TA0 TA1 TA2 TA3 CG CH CI CJ CTCOR CPCOR PA0 PA1 PA2 "
    "PTCA0 PTCA1 PTCA2 PTCB0 PTCB1 PTCB2 PTEMPA0 PTEMPA1 PTEMPA2"
).split()


def read_header(file):
    """Read the header of a .modraw file.

    The header length in bytes is given on the first line of the file.

    Parameters
    ----------
    file : Path or str
        Path to a .modraw file.

    Returns
    -------
    head : str
        Header text.
    """
    with open(file, "rb") as f:
        nbytes = int(f.readline().split(b"=")[1])
        f.seek(0)
        return f.read(nbytes).decode("latin-1")


def read_body(file):
    """Read everything in a .modraw file after the header.

    Parameters
    ----------
    file : Path or str
        Path to a .modraw file.

    Returns
    -------
    body : bytes
        File contents past the header.
    """
    head = read_header(file)
    with open(file, "rb") as f:
        f.seek(len(head.encode("latin-1")))
        return f.read()


def header_setup(head):
    """Extract the acquisition setup fields from a .modraw header.

    Parameters
    ----------
    head : str
        Header text as returned by `read_header`.

    Returns
    -------
    setup : dict
        Survey, vehicle, instrument serial number and the like. Keys are
        lowercased and stripped of their `CTD.` prefix.
    """
    setup = {}
    for key in ("survey", "experiment", "cruise", "vehicle", "fishflag", "SerialNum"):
        m = re.search(rf"CTD\.{key}\s*=\s*'([^']*)'", head)
        if m:
            setup[key.lower()] = m.group(1)
    m = re.search(r"GM_TIME\s*=\s*'([^']+)'", head)
    if m:
        setup["gm_time"] = m.group(1)
    return setup


def sbe49_cal(head):
    """Extract the SBE49 calibration coefficients from a .modraw header.

    Parameters
    ----------
    head : str
        Header text as returned by `read_header`.

    Returns
    -------
    cal : dict
        Calibration coefficients, keys lowercased.
    """
    cal = {}
    for key in _CAL_KEYS:
        m = re.search(rf"^{key}\s*=\s*(\S+)", head, re.MULTILINE)
        if m:
            cal[key.lower()] = float(m.group(1))
    return cal
