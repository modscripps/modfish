import numpy as np
import pytest

from modfish.chi.response import antialias, derivative, preemphasis_inverse, preemphasis_response

FS = 325.52
NET = dict(R24=1e6, R25=577e3, R22=266.1, C19=1e-6)


@pytest.mark.parametrize(
    "f, som, ap00",
    [(3.0, 0.9989, 0.9989), (18.0, 0.9605, 0.9604), (37.5, 0.8391, 0.8371), (50.0, 0.7313, 0.7256)],
)
def test_antialias_matches_spec_table(f, som, ap00):
    # spec Context table, 3 m/s, fs = 325.52 Hz; ap00 in f is sinc^2(2 pi f / fs)
    assert antialias(f, FS, "som_sinc4") == pytest.approx(som, abs=5e-4)
    assert antialias(f, FS, "ap00_sinc2") == pytest.approx(ap00, abs=5e-4)


def test_antialias_agree_to_second_order():
    f = np.array([0.5, 1.0, 2.0])
    x = np.pi * f / FS
    expansion = 1 - 4 / 3 * x**2
    assert antialias(f, FS, "som_sinc4") == pytest.approx(expansion, abs=1e-6)
    assert antialias(f, FS, "ap00_sinc2") == pytest.approx(expansion, abs=1e-6)


def test_antialias_is_one_at_zero_and_rejects_unknown_kind():
    assert antialias(0.0, FS, "som_sinc4") == 1.0
    with pytest.raises(ValueError):
        antialias(1.0, FS, "bessel")


def test_preemphasis_is_a_differentiator_in_band():
    # In the chi band H ~ i 2 pi f tau with tau = (R24 + R25) C19 = 1.577 s
    tau = (NET["R24"] + NET["R25"]) * NET["C19"]
    f = np.array([3.0, 10.0, 40.0])
    expected = 1.0 / (1.0 + (2 * np.pi * f * tau) ** 2)
    assert preemphasis_inverse(f, **NET) == pytest.approx(expected, rel=1e-2)
    assert tau == pytest.approx(1.577)


def test_preemphasis_unity_at_dc():
    assert preemphasis_inverse(np.array([0.0]), **NET)[0] == 1.0
    assert abs(preemphasis_response(np.array([1e-6]), **NET)[0]) == pytest.approx(1.0, rel=1e-3)


def test_derivative():
    assert derivative(np.array([0.0, 1.0])) == pytest.approx([0.0, (2 * np.pi) ** 2])
