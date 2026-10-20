import pathlib

import xarray as xr

import modfish
from modfish.modraw.convert import convert


def test_convert_writes_one_nc_per_file(rootdir, tmp_path):
    src = rootdir / "data/FCTD_modraw_excerpt.modraw"
    written = convert([src], tmp_path)
    assert written == [tmp_path / "FCTD_modraw_excerpt.nc"]
    tree = xr.open_datatree(written[0])
    assert "ctd" in tree.children


def test_convert_skips_existing_without_overwrite(rootdir, tmp_path):
    src = rootdir / "data/FCTD_modraw_excerpt.modraw"
    first = convert([src], tmp_path)
    second = convert([src], tmp_path)
    assert second == []
    third = convert([src], tmp_path, overwrite=True)
    assert third == first


def test_convert_bad_file_is_skipped_not_fatal(rootdir, tmp_path, caplog):
    bad = tmp_path / "empty.modraw"
    bad.write_bytes(b"header_file_size_inbytes = 30\nx\n")
    src = rootdir / "data/FCTD_modraw_excerpt.modraw"
    written = convert([bad, src], tmp_path / "out")
    assert [p.name for p in written] == ["FCTD_modraw_excerpt.nc"]
    assert "empty.modraw" in caplog.text


def test_convert_failed_write_leaves_no_output(rootdir, tmp_path, monkeypatch):
    """A write that dies partway used to leave a truncated .nc that the next
    run, with overwrite=False, skipped as done."""
    import sys

    conv = sys.modules["modfish.modraw.convert"]  # the package re-exports the function

    class Boom:
        def to_netcdf(self, path):
            pathlib.Path(path).write_bytes(b"partial")
            raise OSError("disk full")

    monkeypatch.setattr(conv, "read", lambda f: Boom())
    out = conv.convert([rootdir / "data/FCTD_modraw_excerpt.modraw"], tmp_path)
    assert out == []
    assert list(tmp_path.iterdir()) == []


def test_convert_parallel_matches_sequential(rootdir, tmp_path):
    files = [rootdir / "data/FCTD_modraw_excerpt.modraw", rootdir / "data/EPSI_modraw_excerpt_2024.modraw"]
    seq = modfish.modraw.convert(files, tmp_path / "seq")
    par = modfish.modraw.convert(files, tmp_path / "par", parallel=True)
    assert [p.name for p in seq] == [p.name for p in par] and len(seq) == 2
