#!/usr/bin/env python

"""Tests for `modfish.modraw.sb49.decode_sb49`, the framer-based SB49 decoder."""

import numpy as np

from modfish.modraw.framer import frame
from modfish.modraw.header import read_body, read_header, sbe49_cal
from modfish.modraw.sb49 import decode_sb49


def test_decode_sb49_matches_load_ctd(rootdir):
    import modfish

    file = rootdir / "data/FCTD_modraw_excerpt.modraw"
    packets, _ = frame(read_body(file))
    cal = sbe49_cal(read_header(file))
    ds = decode_sb49([p for p in packets if p.tag == "SB49"], cal)
    ref = modfish.modraw.load_ctd(file)
    assert ds.sizes["time"] == ref.sizes["time"]
    np.testing.assert_array_equal(ds.time.values, ref.time.values)
    np.testing.assert_allclose(ds.p.values, ref.p.values)
    np.testing.assert_allclose(ds.t.values, ref.t.values)
    np.testing.assert_allclose(ds.c.values, ref.c.values)
