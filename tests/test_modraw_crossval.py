"""Cross-validation: 2024-format fixture, Matlab spot-check, rust reader.

`EPSI_modraw_excerpt_2024.modraw` is the header plus the first ~400 kB (cut
at a `\\r\\n` block boundary, 618,819 bytes total) of
`EPSI24_11_26_102923.modraw`, MOTIVE 2024 cruise skq202417s. Tag counts
and root attrs from `modfish.modraw.read()` on the fixture:

    n_frames=1607, n_resync=75, n_bad_checksum=0
    n_blocks_DCAL=1, n_blocks_SOM3=1, n_blocks_SB49=277,
    n_blocks_ECOP=1049, n_blocks_VNAV=138, n_blocks_EFE4=141
    survey='24_1125_d18_fctd_longsectE', experiment='MOTIVE24',
    cruise='MOTIVE', vehicle='FCTD2', fishflag='FCTD', serialnum='0387',
    gm_time='11/26/2024 10:29:23'

n_resync (75) is exactly the interleaved ASCII lines the scanner rejects
as non-SOM frames: 35 `$INGGA` + 35 `$INZDA` + 5 header lines holding a
`$`. 1+1+277+1049+138+141 = 1607 = n_frames.

2024 files use the `$INZDA`/`$INGGA` talker prefix, 2025 files
`$GPZDA`/`$GPGGA`. No SB49/EFE4 timestamps in this fixture are
power-on-relative.
"""

import pathlib
import shutil
import subprocess

import numpy as np
import pytest
import xarray as xr

import modfish

FIXTURE_2024 = "data/EPSI_modraw_excerpt_2024.modraw"

RUST = shutil.which("modraw") or "/home/gunnar/Projects/rust/modraw/target/release/modraw"


@pytest.mark.slow
def test_read_2024_format(rootdir):
    # No n_resync == 0 assertion: n_resync on a healthy file counts the
    # interleaved NMEA sentences and header lines (accounting in the module
    # docstring). File health is n_bad_checksum == 0.
    tree = modfish.modraw.read(rootdir / FIXTURE_2024)
    assert tree.attrs["n_bad_checksum"] == 0
    assert "ctd" in tree.children
    assert "efe" in tree.children
    assert "ecop" in tree.children
    assert "gps" in tree.children
    # Observed tag counts, see module docstring.
    assert tree.attrs["n_frames"] == 1607
    assert tree.attrs["n_blocks_SB49"] == 277
    assert tree.attrs["vehicle"] == "FCTD2"


@pytest.mark.slow
def test_2024_first_ctd_record_matches_matlab(rootdir):
    # Reference values: first element of ctd.P, ctd.T, ctd.C, ctd.dnum in
    # the Matlab-processed EPSI24_11_26_102923.mat of the source file:
    #   dnum[0] = 739582.4370765  ->  2024-11-26 10:29:23.409603 UTC
    #   P[0] = 885.15723318
    #   T[0] = 5.3460743
    #   C[0] = 3.37604346
    # Our first record: time 2024-11-26T10:29:23.410000000 (matches to the
    # ms), p=885.1578417711407, t=5.346073558636533, c=3.3760434619029605.
    tree = modfish.modraw.read(rootdir / FIXTURE_2024)
    first = tree["ctd"].ds.isel(time=0)
    assert first.time.values == np.datetime64("2024-11-26T10:29:23.410000000")
    assert first.p.item() == pytest.approx(885.15723318, abs=1e-3)
    assert first.t.item() == pytest.approx(5.3460743, abs=1e-4)
    assert first.c.item() == pytest.approx(3.37604346, abs=1e-6)


@pytest.mark.slow
@pytest.mark.skipif(not pathlib.Path(RUST).exists(), reason="rust modraw binary not built")
def test_ctd_matches_rust_reader(rootdir, tmp_path):
    # The rust reader writes a flat NetCDF (no groups) with variables
    # `time`, `pressure`, `temperature`, `conductivity` on dim `time` for
    # the CTD stream. Skipped when the `modraw` binary is not available.
    src = rootdir / "data/FCTD_modraw_excerpt.modraw"
    subprocess.run([RUST, "convert", str(src), "--outdir", str(tmp_path)], check=True)
    rust_nc = next(tmp_path.glob("*.nc"))
    rust = xr.open_dataset(rust_nc)  # flat file, no netCDF groups
    ours = modfish.modraw.read(src)["ctd"].ds
    assert ours.sizes["time"] == rust.sizes["time"]
    # The header carries the cal sheet twice, and the copies disagree in the
    # last printed digit of TA0 and PA1. modfish (and Matlab) read the
    # compact copy, the instrument's own GetCC-style dump. The rust reader
    # appears to parse the re-printed spaced copy. The PA1 difference maps
    # to a systematic ~1e-4 dbar pressure offset, so the tolerance covers
    # the cal-copy divergence.
    np.testing.assert_allclose(ours.p.values, rust["pressure"].values, atol=5e-4)
