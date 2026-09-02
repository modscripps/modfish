#!/usr/bin/env python

"""Tests for SOM3 setup and DCAL calibration block parsing."""

import pytest

from modfish.modraw.framer import frame
from modfish.modraw.header import (
    header_setup,
    parse_dcal,
    parse_som3,
    read_header,
    sbe49_cal,
)


@pytest.fixture
def fixture_packets(rootdir):
    # Frame the FULL file: read_body swallows SOM3/DCAL (header-boundary
    # caveat documented in framer.py's module docstring).
    packets, _ = frame((rootdir / "data/FCTD_modraw_excerpt.modraw").read_bytes())
    return packets


def test_parse_som3_fctd_channel_setup(fixture_packets):
    # Expected values byte-verified against the fixture's SOM3 block during
    # planning (EFE4 submodule at payload offset 180, size 468).
    som3 = next(p for p in fixture_packets if p.tag == "SOM3")
    meta = parse_som3(som3.payload)
    assert meta["n_channels"] == 7
    assert meta["channels"] == ["t1", "t2", "f1", "c1", "a1", "a2", "a3"]
    assert meta["recs_per_block"] == 80
    assert meta["adc_conf"] == ["unipolar"] * 7  # all config_0 == 0x01E0
    assert meta["full_range"] == [2.5, 2.5, 2.5, 2.5, 1.8, 1.8, 1.8]


def test_parse_dcal_matches_header_cal(fixture_packets, rootdir):
    # The FCTD ASCII header embeds the same coefficients as the $DCAL block.
    dcal = next(p for p in fixture_packets if p.tag == "DCAL")
    from_block = parse_dcal(dcal.payload)
    from_header = sbe49_cal(read_header(rootdir / "data/FCTD_modraw_excerpt.modraw"))
    assert from_block["ta0"] == pytest.approx(from_header["ta0"])
    assert from_block["cg"] == pytest.approx(from_header["cg"])


def test_parse_dcal_all_coefficients_present(fixture_packets):
    # The $DCAL block spells conductivity coefficients as G/H/I/J rather than
    # the header's CG/CH/CI/CJ; parse_dcal must normalize to the sbe49_cal
    # key set. Values below are read off the raw DCAL text during planning.
    dcal = next(p for p in fixture_packets if p.tag == "DCAL")
    cal = parse_dcal(dcal.payload)
    assert cal["ta0"] == pytest.approx(7.906934e-04)
    assert cal["ta1"] == pytest.approx(2.938320e-04)
    assert cal["ta2"] == pytest.approx(-3.317254e-06)
    assert cal["ta3"] == pytest.approx(2.369378e-07)
    assert cal["cg"] == pytest.approx(-1.005065)
    assert cal["ch"] == pytest.approx(1.325720e-01)
    assert cal["ci"] == pytest.approx(-9.989038e-05)
    assert cal["cj"] == pytest.approx(2.811205e-05)
    assert cal["cpcor"] == pytest.approx(-9.570000e-08)
    assert cal["ctcor"] == pytest.approx(3.250000e-06)
    assert cal["pa0"] == pytest.approx(-5.312621e-02)
    assert cal["pa1"] == pytest.approx(8.981492e-03)
    assert cal["pa2"] == pytest.approx(4.060338e-11)
    assert cal["ptca0"] == pytest.approx(5.252602e05)
    assert cal["ptca1"] == pytest.approx(-8.509373)
    assert cal["ptca2"] == pytest.approx(3.786065e-01)
    assert cal["ptcb0"] == pytest.approx(1.027653e02)
    assert cal["ptcb1"] == pytest.approx(4.177845e-03)
    assert cal["ptcb2"] == pytest.approx(0.0)
    assert cal["ptempa0"] == pytest.approx(-8.653358e01)
    assert cal["ptempa1"] == pytest.approx(4.111889e01)
    assert cal["ptempa2"] == pytest.approx(1.235626)


def test_header_setup_2025_format_serialnum():
    # Exact line copied from FCTD25_12_08_133647.modraw (skq202521s cruise,
    # server mount /mnt/mod-server/MOTIVE/Cruises/skq202521s/05_processed_data
    # /25_1208_d08_motiveb/raw/), the 2025 header format notebook 04 found
    # serialnum missing for downstream. header_setup itself matches this
    # line correctly (single-quoted, no space before `=`); the L0 defect is
    # in reader.read() not stamping header_setup fields onto the ctd group's
    # own attrs, fixed separately and covered in test_modraw_reader.py.
    head = "CTD.SerialNum='0537'\n"
    setup = header_setup(head)
    assert setup["serialnum"] == "0537"
