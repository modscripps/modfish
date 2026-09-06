import numpy as np
import pytest

from modfish.chi.noise import NoiseFloor, resolve
from modfish.chi.spectra import window_spectrum

FS = 325.52


def test_builtin_matches_the_measurement():
    nf = NoiseFloor.from_builtin("fctd_2026")
    for freq, want in ((4.0, 7.38e-11), (10.0, 1.12e-10),
                       (20.0, 2.40e-10), (37.5, 4.32e-10)):
        got = float(nf.at(np.array([freq]))[0])
        assert got == pytest.approx(want, rel=0.1), f"{freq} Hz: {got} vs {want}"
    assert nf.nu == pytest.approx(11.04, abs=0.1)
    assert nf.source and nf.records and nf.measured


def test_at_is_the_identity_on_the_native_grid():
    nf = NoiseFloor.from_builtin("fctd_2026")
    assert nf.at(nf.f) == pytest.approx(nf.n, rel=1e-9)


def test_at_holds_the_end_value_outside_the_grid():
    nf = NoiseFloor.from_builtin("fctd_2026")
    assert float(nf.at(np.array([-5.0]))[0]) == pytest.approx(nf.n[0], rel=1e-9)
    assert float(nf.at(np.array([1e4]))[0]) == pytest.approx(nf.n[-1], rel=1e-9)


def test_resolve_dispatches():
    assert resolve(None) is None
    assert isinstance(resolve("builtin:fctd_2026"), NoiseFloor)
    with pytest.raises(ValueError, match="unknown builtin"):
        resolve("builtin:nope")


def test_roundtrip_through_a_file(tmp_path):
    nf = NoiseFloor.from_builtin("fctd_2026")
    p = tmp_path / "floor.npz"
    np.savez(p, f=nf.f, n=nf.n, nu=nf.nu, source="test", records="r1", measured="2026-01-01")
    back = NoiseFloor.from_file(p)
    assert back.at(nf.f) == pytest.approx(nf.n, rel=1e-9)
    assert back.source == "test"


def test_construction_coerces_plain_lists():
    nf = NoiseFloor(f=[1.0, 2.0, 3.0], n=[1e-10, 2e-10, 3e-10],
                     nu=11.0, source="s", records="r", measured="m")
    assert isinstance(nf.f, np.ndarray)
    assert isinstance(nf.n, np.ndarray)


def test_construction_validates_lists_with_valueerror():
    with pytest.raises(ValueError, match="strictly increasing"):
        NoiseFloor(f=[2.0, 1.0, 3.0], n=[1e-10, 2e-10, 3e-10],
                   nu=11.0, source="s", records="r", measured="m")
    with pytest.raises(ValueError, match="positive"):
        NoiseFloor(f=[1.0, 2.0, 3.0], n=[1e-10, -2e-10, 3e-10],
                   nu=11.0, source="s", records="r", measured="m")


def test_welch_degrees_of_freedom():
    """nu underpins the floor estimator, so it is measured, not assumed."""
    rng = np.random.default_rng(0)
    level, nw = 1e-9, int(round(2.0 * FS))
    sigma = np.sqrt(level * FS / 2)
    P = np.array([window_spectrum(rng.normal(0.0, sigma, nw), FS, 0.5)[1]
                  for _ in range(2000)])
    interior = P[:, 1:-1]
    nu = 2 * interior.mean(axis=0) ** 2 / interior.var(axis=0)
    assert interior.mean() == pytest.approx(level, rel=0.02)
    assert float(np.median(nu)) == pytest.approx(11.04, abs=0.6)
