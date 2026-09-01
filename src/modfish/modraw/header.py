#!/usr/bin/env python
# coding: utf-8
"""
Header parsing and setup extraction from `.modraw` files.

This module handles the ASCII header format of MOD `.modraw` files and extracts
calibration coefficients for the SBE49 CTD.
"""

import re


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


#: The $DCAL block spells conductivity coefficients G/H/I/J rather than the
#: ASCII header's CG/CH/CI/CJ; `parse_dcal` normalizes to the latter via this
#: map (target `_CAL_KEYS` name -> name as it appears in $DCAL text).
_DCAL_KEY_ALIASES = {"CG": "G", "CH": "H", "CI": "I", "CJ": "J"}


def _cal_from_text(text, key_aliases=None):
    """Extract SBE49 calibration coefficients from cal-sheet-style text.

    Shared by `sbe49_cal` (ASCII header) and `parse_dcal` ($DCAL block),
    which spell the coefficients slightly differently: the header keys are
    flush left with no space before `=` (`TA0= ...`), the $DCAL block is
    indented with a space before `=` (`    TA0 = ...`), and the $DCAL block
    additionally uses G/H/I/J instead of CG/CH/CI/CJ.

    Parameters
    ----------
    text : str
        Text to search, decoded latin-1.
    key_aliases : dict, optional
        Maps a `_CAL_KEYS` name to the name it is spelled as in `text`, for
        keys where the two differ (see `_DCAL_KEY_ALIASES`).

    Returns
    -------
    cal : dict
        Calibration coefficients found, keys lowercased.
    """
    key_aliases = key_aliases or {}
    cal = {}
    for key in _CAL_KEYS:
        source_key = key_aliases.get(key, key)
        m = re.search(rf"^\s*{source_key}\s*=\s*(\S+)", text, re.MULTILINE)
        if m:
            cal[key.lower()] = float(m.group(1))
    return cal


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
    return _cal_from_text(head)


def parse_som3(payload):
    """Parse the EFE channel setup embedded in a $SOM3 block payload.

    The $SOM3 payload is a stream of submodules, each prefixed by a 4-byte
    little-endian size immediately before its ASCII tag. This extracts the
    `EFE`/`EFE3`/`EFE4` submodule, which describes the analog channel setup:
    channel count, records per block, and per-channel name/ADC config/full
    range. Layout follows the rust parser (`src/setup.rs:65-177`) and the
    Matlab reader (`mod_som_read_setup_from_raw.m:339-383`).

    Parameters
    ----------
    payload : bytes
        Raw payload of a $SOM3 `Packet` (see `modfish.modraw.framer.frame`).

    Returns
    -------
    meta : dict
        - n_channels : int
        - channels : list of str, channel names (e.g. "t1", "c1", "a1")
        - adc_conf : list of str, "unipolar" or "bipolar" per channel
        - full_range : list of float, full-scale range per channel (volts)
        - recs_per_block : int

    Raises
    ------
    ValueError
        If no EFE submodule is found, or its channel count is implausible.
    """
    for tag in (b"EFE4", b"EFE3", b"EFE"):
        idx = payload.find(tag)
        if idx >= 4:
            break
    else:
        raise ValueError("no EFE submodule in $SOM3 payload")
    size = int.from_bytes(payload[idx - 4 : idx], "little")
    module = payload[idx - 4 : idx - 4 + size]

    n_channels = int.from_bytes(module[28:32], "little")
    recs_per_block = int.from_bytes(module[32:36], "little")
    if not 0 < n_channels <= 32:
        raise ValueError(f"implausible EFE channel count {n_channels}")

    channels, adc_conf, full_range = [], [], []
    for c in range(n_channels):
        entry = module[44 + c * 60 : 44 + (c + 1) * 60]
        name = entry[:4].split(b"\x00")[0].decode()
        config_0 = int.from_bytes(entry[40:42], "little")
        if config_0 == 0x01E0:
            conf = "unipolar"
        elif config_0 == 0x09E0:
            conf = "bipolar"
        else:  # fall back by channel name, epsiSetup_fill_meta_data.m:112-118
            conf = "bipolar" if name.startswith("s") else "unipolar"
        channels.append(name)
        adc_conf.append(conf)
        full_range.append(1.8 if name.startswith("a") else 2.5)
    return dict(
        n_channels=n_channels,
        channels=channels,
        adc_conf=adc_conf,
        full_range=full_range,
        recs_per_block=recs_per_block,
    )


def parse_dcal(payload):
    """Parse the SBE49 calibration coefficients in a $DCAL block payload.

    The $DCAL payload is the plain-text SBE49 calibration sheet, decoded
    latin-1. It spells conductivity coefficients G/H/I/J rather than the
    ASCII header's CG/CH/CI/CJ (see `_DCAL_KEY_ALIASES`); this normalizes to
    the same key set `sbe49_cal` returns.

    Parameters
    ----------
    payload : bytes
        Raw payload of a $DCAL `Packet` (see `modfish.modraw.framer.frame`).

    Returns
    -------
    cal : dict
        Calibration coefficients, keys lowercased, same key set as
        `sbe49_cal`.
    """
    text = payload.decode("latin-1")
    return _cal_from_text(text, _DCAL_KEY_ALIASES)
