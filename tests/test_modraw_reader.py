import logging

import numpy as np
import pytest
import xarray as xr
from modraw_frames import make_frame as _make_frame
from modraw_frames import minimal_header as _minimal_header

import modfish


@pytest.fixture(scope="module")
def tree(rootdir):
    return modfish.modraw.read(rootdir / "data/FCTD_modraw_excerpt.modraw")


@pytest.fixture
def synthetic_body():
    """A minimal synthetic `.modraw` body for exercising reader.py edge cases.

    A from-scratch header (see `_minimal_header`; carries no calibration
    coefficients) followed by three hand-built frames:

    - one valid `$ECOP` record, so `read()` always has a group to return
      and never raises "no decodable data streams" for these tests;
    - one `$SB49` frame whose 16-hex header timestamp is `f`*16
      (0xFFFFFFFFFFFFFFFF ms), which cannot fit in an int64 regardless of
      the datetime unit pandas picks, so it reliably raises
      `OutOfBoundsDatetime` without the `errors="coerce"` fix. This is a
      larger value than the review's "~9.2e15 ms" example: under the
      installed pandas (3.0), `pd.to_datetime(unit="ms")` now auto-selects
      a coarser resolution for values like 9.2e15 or `0x1` + 15 zero
      hex-chars, so those no longer overflow and would not reproduce the
      bug here; the full-`f` value overflows unconditionally. Since the
      header carries no calibration coefficients and no `$DCAL` block is
      present, this frame also exercises the "SB49 without cal" path;
    - one `$EFE4` frame with an arbitrary payload (never decoded, since
      this file has no `$SOM3` block), exercising the "EFE4 without SOM3"
      path.
    """
    ecop_payload = b"0000019afe3391257FFF80098087"  # hand-verified valid record
    huge_ts = int("f" * 16, 16)
    return (
        _minimal_header()
        + _make_frame(b"ECOP", 1, ecop_payload)
        + _make_frame(b"SB49", huge_ts, b"x" * 40)
        + _make_frame(b"EFE4", 2, b"y" * 10)
    )


def test_read_returns_datatree_with_expected_groups(tree):
    assert isinstance(tree, xr.DataTree)
    assert set(tree.children) == {"ctd", "efe", "ecop", "gps"}  # no ALTI in fixture


def test_read_root_attrs_carry_quality_tallies(tree):
    # n_resync counts failed frame attempts. On a healthy full-file scan
    # that includes every interleaved NMEA sentence (not SOM-framed, so the
    # scanner correctly rejects them: 147 here) plus 5 header text lines.
    # Corruption is signaled by the checksum/length tallies, which the spec
    # requires to be zero on clean files.
    assert tree.attrs["n_resync"] == 152
    assert tree.attrs["n_bad_checksum"] == 0
    assert tree.attrs["n_blocks_SB49"] == 238
    assert tree.attrs["vehicle"] == "FCTD1"


def test_read_ctd_group_carries_serialnum(tree):
    # header_setup fields (survey, experiment, cruise, vehicle, fishflag,
    # serialnum, gm_time) were stamped only onto the DataTree root's attrs,
    # not onto the ctd group's own attrs (which got only the SBE49
    # calibration coefficients); a consumer opening the ctd group alone,
    # as the L0 pipeline does, never saw serialnum (notebook 04, real 2025
    # data). header_setup itself parses the field correctly; this pins the
    # ctd group carrying it too.
    assert tree["ctd"].attrs["serialnum"] == "0537"


def test_read_laptop_time_diagnostics_present(tree):
    assert tree.ds.sizes["block"] == tree.attrs["n_frames"]
    assert tree.ds.block_tag.values[0] in ("SOM3", "DCAL", "SB49", "EFE4", "ECOP", "ALTI")


def test_read_roundtrips_through_netcdf(tree, tmp_path):
    path = tmp_path / "excerpt.nc"
    tree.to_netcdf(path)
    back = xr.open_datatree(path)
    np.testing.assert_allclose(back["ctd"].ds.p.values, tree["ctd"].ds.p.values)


def test_read_survives_corrupt_block_timestamp(synthetic_body, tmp_path):
    # Regression test: a corrupt-but-well-framed block timestamp must not
    # kill read() for the whole file. It should surface as NaT in the
    # block_time diagnostic, not raise OutOfBoundsDatetime.
    path = tmp_path / "corrupt_timestamp.modraw"
    path.write_bytes(synthetic_body)

    tree = modfish.modraw.read(path)

    is_sb49 = tree.ds.block_tag.values == "SB49"
    assert is_sb49.any()
    assert np.isnat(tree.ds.block_time.values[is_sb49]).all()


def test_read_warns_when_sb49_present_without_cal(synthetic_body, tmp_path, caplog):
    path = tmp_path / "sb49_no_cal.modraw"
    path.write_bytes(synthetic_body)

    with caplog.at_level(logging.WARNING):
        tree = modfish.modraw.read(path)

    assert "ctd" not in tree.children
    assert any(
        "SB49" in record.message and "calibration" in record.message
        for record in caplog.records
    )


def test_read_warns_when_efe4_present_without_som3(synthetic_body, tmp_path, caplog):
    path = tmp_path / "efe4_no_som3.modraw"
    path.write_bytes(synthetic_body)

    with caplog.at_level(logging.WARNING):
        tree = modfish.modraw.read(path)

    assert "efe" not in tree.children
    assert any(
        "EFE4" in record.message and "SOM3" in record.message
        for record in caplog.records
    )


def _write(tmp_path, name, body):
    path = tmp_path / name
    path.write_bytes(body)
    return path


def _ctd_body(sb49_packets=None, header_text=b""):
    """Synthetic file: header, the fixture's $DCAL, its first 20 $SB49."""
    from modraw_frames import fixture_packets, minimal_header, reframe

    pk = fixture_packets()
    sb49 = pk["SB49"][:20] if sb49_packets is None else sb49_packets
    return minimal_header(400, header_text) + reframe(pk["DCAL"]) + reframe(sb49)


def test_read_overflowing_record_timestamp_drops_only_that_record(tmp_path):
    """One flipped high hex digit in a record timestamp is valid hex but far
    past datetime64[ns]. It used to raise OverflowError, losing the file."""
    from modraw_frames import fixture_packets

    pk = fixture_packets()
    bad = pk["SB49"][3]
    payload = bytearray(bad.payload)
    payload[2:3] = b"f"  # 0000019a... -> 00f0019a..., about year 135000
    bad = type(bad)(**{**bad.__dict__, "payload": bytes(payload)})
    packets = pk["SB49"][:3] + [bad] + pk["SB49"][4:20]
    tree = modfish.modraw.read(_write(tmp_path, "ovf.modraw", _ctd_body(packets)))
    ctd = tree["ctd"].ds
    n_rec = sum(len(p.payload) // 40 for p in pk["SB49"][:20])
    assert ctd.sizes["time"] == n_rec - 1
    assert ctd.attrs["n_bad_time"] == 1
    assert not np.isnat(ctd.time.values).any()


def test_read_decoder_failure_omits_only_that_group(tmp_path, caplog):
    """GGA sentences without a ZDA to date them used to raise and lose the
    CTD too. The gps group is now omitted with a warning."""
    body = _ctd_body() + b"T0000000001$GPGGA,120000.00,0100.0000,N,13930.0000,W,1,08\r\n"
    with caplog.at_level(logging.WARNING):
        tree = modfish.modraw.read(_write(tmp_path, "gga.modraw", body))
    assert "ctd" in tree.children and "gps" not in tree.children
    assert any("gps" in r.message and "ZDA" in r.message for r in caplog.records)


def test_read_drops_bad_checksum_packets(tmp_path):
    from modraw_frames import fixture_packets, make_frame

    pk = fixture_packets()
    sb = pk["SB49"][:20]
    corrupt = bytearray(sb[5].payload)
    corrupt[20] ^= 0x01  # one flipped bit in the counts
    stale = 0
    for byte in sb[5].payload:
        stale ^= byte  # checksum of the original payload, wrong for the corrupt one
    body = _ctd_body(sb[:5]) + make_frame(b"SB49", sb[5].timestamp_ms, bytes(corrupt), chk=stale)
    good = modfish.modraw.read(_write(tmp_path, "good.modraw", _ctd_body(sb[:5])))["ctd"].ds
    tree = modfish.modraw.read(_write(tmp_path, "badchk.modraw", body))
    assert tree.attrs["n_bad_checksum"] == 1
    assert tree["ctd"].ds.sizes["time"] == good.sizes["time"]
    assert tree["ctd"].attrs["n_bad_checksum"] == 1


def test_read_partial_header_cal_falls_back_to_dcal(tmp_path):
    """A header carrying only some coefficients (no CG/CH/CI/CJ) used to
    skip the $DCAL fallback and fail with KeyError: 'cg'."""
    text = b"TA0= 7.906934e-04\nTA1= 2.938320e-04\nPA0= -5.312621e-02\n"
    tree = modfish.modraw.read(_write(tmp_path, "partial.modraw", _ctd_body(header_text=text)))
    assert "ctd" in tree.children
    assert tree["ctd"].attrs["cg"] == pytest.approx(-1.005065)


def test_read_vnav_without_vnmar_adds_no_group(tmp_path):
    from modraw_frames import make_frame

    body = _ctd_body() + make_frame(b"VNAV", 1765000000000,
                                    b"0000019afe33908b$VNYPR,+010.0,+001.0,-002.0*00\r\n")
    tree = modfish.modraw.read(_write(tmp_path, "vnypr.modraw", body))
    assert "vnav" not in tree.children


def test_read_block_time_is_nanoseconds(tree):
    assert tree.ds.block_time.dtype == np.dtype("datetime64[ns]")
