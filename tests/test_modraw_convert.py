import xarray as xr

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
