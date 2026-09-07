import numpy as np
import pytest

from modfish.modraw.vnav import decode_vnmar_bytes


def test_decode_vnmar_synthetic_sentence():
    body = (
        b"0000019b04646b0c$VNMAR,-00.2176,-00.1406,-00.2092,"
        b"+00.414,-08.134,+04.416,-00.105419,-00.159902,-00.004134*55\r\n"
    )
    ds = decode_vnmar_bytes(body)
    assert ds.sizes["time"] == 1
    assert ds.mag_x[0].values == pytest.approx(-0.2176)
    assert ds.accel_y[0].values == pytest.approx(-8.134)
    assert ds.gyro_z[0].values == pytest.approx(-0.004134)


def test_decode_vnmar_rejects_bad_checksum():
    body = (
        b"0000019b04646b0c$VNMAR,-00.2176,-00.1406,-00.2092,"
        b"+00.414,-08.134,+04.416,-00.105419,-00.159902,-00.004134*00\r\n"
    )
    ds = decode_vnmar_bytes(body)
    assert ds.sizes["time"] == 0


def test_decode_vnmar_fixture_rate_and_gravity(rootdir):
    raw = (rootdir / "data/EPSI_modraw_excerpt_2024.modraw").read_bytes()
    ds = decode_vnmar_bytes(raw)
    assert ds.sizes["time"] == 1380
    assert ds.time[0].values == np.datetime64("2024-11-26T10:29:23.410")
    dt = np.diff(ds.time.values).astype("timedelta64[ms]").astype(float)
    assert np.median(dt) == pytest.approx(25.0)
    g = np.sqrt(ds.accel_x**2 + ds.accel_y**2 + ds.accel_z**2).median()
    assert 9.0 < float(g) < 10.5, float(g)


def test_read_exposes_vnav_group(rootdir):
    from modfish.modraw import read

    tree = read(rootdir / "data/EPSI_modraw_excerpt_2024.modraw")
    assert "vnav" in tree
    assert tree["vnav"].ds.sizes["time"] == 1380
