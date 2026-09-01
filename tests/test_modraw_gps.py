import numpy as np
import pytest

from modfish.modraw.gps import decode_gga


def test_decode_gga_synthetic_sentence_with_zda_date():
    body = (
        b"T2951178339$GPGGA,134325.00,0147.061045,N,13750.303062,W,"
        b"5,26,0.49,28.368,M,0.000,M,2.2,0000*68\r\n"
        b"T2951178357$GPZDA,134325.00,08,12,2025,00,00*6A\r\n"
    )
    ds = decode_gga(body)
    assert ds.sizes["time"] == 1
    assert ds.lat[0].values == pytest.approx(1 + 47.061045 / 60)
    assert ds.lon[0].values == pytest.approx(-(137 + 50.303062 / 60))
    assert ds.time[0].values == np.datetime64("2025-12-08T13:43:25.000")


def test_decode_gga_midnight_rollover_corrects_date():
    # GGA just after midnight, ZDA still dated the day before means the
    # nearest-ZDA date must roll forward.
    body = (
        b"T0000000001$GPZDA,235959.00,08,12,2025,00,00*6A\r\n"
        b"T0000000002$GPGGA,000001.00,0147.061045,N,13750.303062,W,"
        b"5,26,0.49,28.368,M,0.000,M,2.2,0000*68\r\n"
    )
    ds = decode_gga(body)
    assert ds.time[0].values == np.datetime64("2025-12-09T00:00:01.000")


def test_decode_gga_fixture_has_expected_fixes(rootdir):
    ds = decode_gga((rootdir / "data/FCTD_modraw_excerpt.modraw").read_bytes())
    assert ds.sizes["time"] == 30
    assert ds.time.to_index().is_monotonic_increasing
    # Fixture starts 2025-12-08 13:43:25 near 1.8°N, 137.8°W.
    assert ds.time[0].values == np.datetime64("2025-12-08T13:43:25.000")
    assert (abs(ds.lat - 1.8) < 0.5).all()
    assert (abs(ds.lon + 137.8) < 0.5).all()


def test_decode_gga_no_gps_returns_empty(rootdir):
    ds = decode_gga(b"no gps here")
    assert ds.sizes.get("time", 0) == 0
