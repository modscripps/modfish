import logging

import numpy as np
import pytest
import xarray as xr

import modfish


@pytest.fixture(scope="module")
def tree(rootdir):
    return modfish.modraw.read(rootdir / "data/FCTD_modraw_excerpt.modraw")


def _make_frame(tag, timestamp_ms, payload, t_prefix=None):
    """Build one SOM frame with a correct trailing XOR checksum.

    Mirrors `make_frame` in `tests/test_modraw_framer.py`; duplicated here
    (rather than imported) to keep this test file self-contained.
    """
    chk = 0
    for byte in payload:
        chk ^= byte
    head = b"" if t_prefix is None else b"T%010d" % t_prefix
    return (
        head
        + b"$" + tag
        + b"%016x" % timestamp_ms
        + b"%08x" % len(payload)
        + b"*00"  # header checksum, hex-validated only
        + payload
        + b"*%02X\r\n" % chk
    )


def _minimal_header(nbytes=200):
    """Build a header `read_header`/`header_setup`/`sbe49_cal` can parse.

    Not a slice of the real fixture: the fixture's own declared header span
    (`header_file_size_inbytes`) falls *inside* its block stream (see the
    caution in `modfish.modraw.framer`'s module docstring) and so contains
    genuine `$SOM3`/`$DCAL`/`$SB49`/... frames and real calibration
    coefficients. Reusing it here would defeat the "no cal"/"no SOM3" tests
    below, since those blocks would already be present. This header instead
    declares its own length on the first line and pads with filler text
    that contains no `$`, no calibration keys, and no `SOM3`/`DCAL` tag.
    """
    first_line = f"header_file_size_inbytes = {nbytes}\n".encode()
    filler = b"X" * (nbytes - len(first_line))
    header = first_line + filler
    assert len(header) == nbytes
    return header


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


def test_read_ctd_group_contains_load_ctd(tree, rootdir):
    # read() frames the full file and captures SB49 blocks that sit inside
    # the (wrongly) declared header span, so it holds a superset of what the
    # read_body-based load_ctd sees; the legacy records must match exactly.
    ref = modfish.modraw.load_ctd(rootdir / "data/FCTD_modraw_excerpt.modraw")
    n_extra = tree["ctd"].ds.sizes["time"] - ref.sizes["time"]
    assert n_extra >= 0
    np.testing.assert_array_equal(tree["ctd"].time.values[n_extra:], ref.time.values)
    np.testing.assert_allclose(tree["ctd"].ds.p.values[n_extra:], ref.p.values)


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
