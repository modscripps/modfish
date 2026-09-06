"""The microconductivity channel's instrument noise floor.

The floor is a property of the acquisition chain, so it is stored in raw
`c1` volts against frequency and pushed through the same transfer-function
factor as the signal. See
`plans/2026-09-06-chi-noise-floor-design.md` and the measurement note
`motive-cruise-proc/book/data/fctd_chi_noise_floor.md`.
"""

import dataclasses
import pathlib

import numpy as np

# Lower envelope of the per-record chi-square tail estimates over eighteen
# FCTD deployment segments spanning the 2024 and 2025 MOTIVE cruises,
# measured in motive-cruise-proc drafts/15 on 2026-09-06. Per frequency the
# value is the median of the records within 1.5x of the minimum. The 4 Hz
# cluster holds seven records across two cruises, both fish and four probes.
_DEFAULT_NU = 11.04
# 40 points, 3.994 to 159.8 Hz (the f = 0 DC bin is dropped; see the
# measurement note for why it is a detrending residual, not a floor).
_DEFAULT_F = (3.994, 7.988, 11.98, 15.98, 19.97, 23.96, 27.96, 31.95, 35.95, 39.94, 43.94, 47.93, 51.92, 55.92, 59.91, 63.91, 67.9, 71.89, 75.89, 79.88, 83.88, 87.87, 91.86, 95.86, 99.85, 103.8, 107.8, 111.8, 115.8, 119.8, 123.8, 127.8, 131.8, 135.8, 139.8, 143.8, 147.8, 151.8, 155.8, 159.8)
_DEFAULT_N = (7.38e-11, 1.05e-10, 1.38e-10, 1.83e-10, 2.4e-10, 2.73e-10, 2.88e-10, 3.7e-10, 4.17e-10, 4.62e-10, 4.83e-10, 4.76e-10, 5.44e-10, 5e-10, 6.58e-10, 5.94e-10, 5.53e-10, 4.81e-10, 4.73e-10, 4.69e-10, 4.28e-10, 4.26e-10, 3.91e-10, 3.69e-10, 3.35e-10, 3.14e-10, 2.85e-10, 2.96e-10, 2.26e-10, 1.86e-10, 1.34e-10, 1.24e-10, 1e-10, 9.08e-11, 8.29e-11, 6.85e-11, 6.56e-11, 5.82e-11, 5.32e-11, 5.12e-11)
_DEFAULT_RECORDS = ("d12, d07s4, d07s6, d07s2, d11, d19, d20 define the 4 Hz "
                    "cluster; eighteen segments contributed bounds")
_DEFAULT_SOURCE = ("motive-cruise-proc drafts/15_fctd_chi_noise_floor, "
                   "lower envelope at q=0.05, cluster 1.5x")
_DEFAULT_MEASURED = "2026-09-06"


@dataclasses.dataclass(frozen=True)
class NoiseFloor:
    """A measured noise spectrum of the raw `c1` channel.

    Parameters
    ----------
    f : numpy.ndarray
        Frequency, Hz, increasing.
    n : numpy.ndarray
        Noise power spectral density, V^2/Hz, positive.
    nu : float
        Degrees of freedom of the Welch estimator the floor was measured
        with. Carried so a consumer can reproduce the quantile correction.
    source, records, measured : str
        Provenance.
    """

    f: np.ndarray
    n: np.ndarray
    nu: float
    source: str
    records: str
    measured: str

    def __post_init__(self):
        if self.f.shape != self.n.shape or self.f.ndim != 1:
            raise ValueError("f and n must be 1-D and the same length")
        if not np.all(np.diff(self.f) > 0):
            raise ValueError("f must be strictly increasing")
        if not np.all(self.n > 0):
            raise ValueError("n must be positive; the floor is interpolated in log space")

    def at(self, f) -> np.ndarray:
        """The floor on `f`, log in PSD and linear in frequency.

        Beyond either end of the measured grid the end value is held,
        never extrapolated.
        """
        return 10 ** np.interp(np.asarray(f, dtype=float), self.f, np.log10(self.n))

    @classmethod
    def from_builtin(cls, name: str) -> "NoiseFloor":
        if name != "fctd_2026":
            raise ValueError(f"unknown builtin noise floor {name!r}")
        return cls(f=np.array(_DEFAULT_F, dtype=float),
                   n=np.array(_DEFAULT_N, dtype=float),
                   nu=_DEFAULT_NU, source=_DEFAULT_SOURCE,
                   records=_DEFAULT_RECORDS, measured=_DEFAULT_MEASURED)

    @classmethod
    def from_file(cls, path) -> "NoiseFloor":
        with np.load(pathlib.Path(path), allow_pickle=False) as z:
            return cls(f=np.asarray(z["f"], dtype=float),
                       n=np.asarray(z["n"], dtype=float),
                       nu=float(z["nu"]), source=str(z["source"]),
                       records=str(z["records"]), measured=str(z["measured"]))


def resolve(ref) -> "NoiseFloor | None":
    """Turn a `ChiParams.noise` reference into a `NoiseFloor`.

    `None` disables subtraction. `"builtin:<name>"` selects a packaged
    spectrum. Anything else is a path to an `.npz`.
    """
    if ref is None or ref == "":
        return None
    if isinstance(ref, NoiseFloor):
        return ref
    ref = str(ref)
    if ref.startswith("builtin:"):
        return NoiseFloor.from_builtin(ref[len("builtin:"):])
    return NoiseFloor.from_file(ref)
