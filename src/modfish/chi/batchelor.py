"""Batchelor temperature-gradient spectrum and the resolved band fraction.

The form follows `EPSILOMETER/EPSILON/process/batchelor.m` (Lien 1992) with
`erfc` in the bracket. The resolved fraction `r(eps)` of a window replaces
the `c eps^p` power law of Alford and Pinkel (2000).
`BandResponse` computes it as the fraction of the Batchelor variance that
the window's Welch band sum recovers on average, so it includes the
estimator's own response (window leakage, the per-segment detrend) along
with the band edges.
"""

import dataclasses

import numpy as np
from scipy.signal import get_window
from scipy.special import erfc

from modfish.chi.config import ChiParams
from modfish.chi.response import antialias, preemphasis_inverse
from modfish.chi.spectra import band_kmax


def spectrum(k, eps, chi, nu, D, q):
    """Batchelor gradient spectrum, per cpm, integrating to chi / (6 D).

    Parameters
    ----------
    k : array_like
        Wavenumber, cpm.
    eps : float
        Dissipation rate, W/kg.
    chi : float
        Temperature variance dissipation, K^2/s.
    nu : float
        Kinematic viscosity, m^2/s.
    D : float
        Thermal diffusivity, m^2/s.
    q : float
        Universal constant, dimensionless.

    Returns
    -------
    numpy.ndarray
        Spectrum, K^2/m^2 per cpm, zero where the bracket goes negative.
    """
    k = np.asarray(k, dtype=float)
    kb = (eps / nu / D**2) ** 0.25
    a = np.sqrt(2 * q) * 2 * np.pi * k / kb
    uppera = erfc(a / np.sqrt(2)) * np.sqrt(np.pi / 2)
    gfun = 2 * np.pi * a * (np.exp(-(a**2) / 2) - a * uppera)
    P = np.sqrt(q / 2) * (chi / kb / D) * gfun
    return np.where(P > 0, P, 0.0)


def band_fraction(eps, kmin, kmax, nu, D, q, n=4001):
    """Fraction of the Batchelor variance between kmin and kmax.

    Parameters
    ----------
    eps : float
        Dissipation rate, W/kg.
    kmin : float
        Minimum wavenumber, cpm.
    kmax : float
        Maximum wavenumber, cpm.
    nu : float
        Kinematic viscosity, m^2/s.
    D : float
        Thermal diffusivity, m^2/s.
    q : float
        Universal constant.
    n : int, optional
        Number of points for integration, default 4001.

    Returns
    -------
    float
        Fraction of variance in the band, dimensionless.
    """
    k = np.linspace(kmin, kmax, n)
    return float(6 * D * np.trapezoid(spectrum(k, eps, 1.0, nu, D, q), k))


def welch_kernels(fs, nsec, pad=16):
    """Expected response of each `window_spectrum` bin to a stationary input.

    For a stationary input with one-sided PSD `S(f)`, the expected value of
    `window_spectrum` at bin `i` is `sum_f K[i] S(f) df` over `f`. The
    kernel follows from the estimator: a periodic Hamming window of
    `M = round(nsec fs)` samples applied after a per-segment linear
    detrend (projection `P`), one-sided density scaling. With
    `v_i = P (w exp(-2 pi i i n / M))` and
    `V_i(f) = sum_n v_i[n] exp(2 pi i f n / fs)`,
    `K_i(f) = (|V_i(f)|^2 + |V_i(-f)|^2) / (fs sum w^2)`. Segment
    overlap does not change the expectation.

    Parameters
    ----------
    fs : float
        Sampling rate, Hz.
    nsec : float
        Welch segment length, s.
    pad : int, optional
        Zero-padding factor of the frequency grid, default 16.

    Returns
    -------
    f : numpy.ndarray
        Frequency grid, Hz, `0` to `fs/2` in steps of `fs / (pad M)`.
    f_bins : numpy.ndarray
        Bin frequencies of `window_spectrum`, Hz, `i fs / M` for
        `i = 0 .. M // 2`.
    K : numpy.ndarray
        Shape `(f_bins.size, f.size)`, per Hz. Row 0 (DC) is zero.
    """
    M = int(round(nsec * fs))
    L = M * pad
    n = np.arange(M)
    w = get_window("hamming", M)
    A = np.column_stack([np.ones(M), n - n.mean()])
    P = np.eye(M) - A @ np.linalg.solve(A.T @ A, A.T)
    nb = M // 2 + 1
    nf = L // 2 + 1
    K = np.zeros((nb, nf))
    scale = 1.0 / (fs * np.sum(w**2))
    for i in range(1, nb):
        v = P @ (w * np.exp(-2j * np.pi * i * n / M))
        plus = np.fft.ifft(v, L)[:nf] * L
        minus = np.fft.fft(v, L)[:nf]
        K[i] = (np.abs(plus) ** 2 + np.abs(minus) ** 2) * scale
    return np.arange(nf) * fs / L, np.arange(nb) * fs / M, K


@dataclasses.dataclass
class BandResponse:
    """Resolved fraction `r(eps)` of the Welch band sum, per fall speed.

    `G[j, b, e]` is the expected contribution of bin `bins[b]` to the band
    sum of a window falling at `spd_grid[j]` through a Batchelor spectrum
    at `eps_grid[e]` with unit chi. A window's `r` is the sum of `G` over
    the bins `run_range` sums for it (`band_bins`), linear in `spd`.

    Attributes
    ----------
    fs : float
        Hz, sampling rate the kernels were built for.
    M : int
        Samples per Welch segment.
    spd_grid : numpy.ndarray
        m/s, increasing.
    eps_grid : numpy.ndarray
        W/kg, increasing.
    bins : numpy.ndarray of int
        Welch bin indices covered by `G`.
    G : numpy.ndarray
        Shape `(spd_grid.size, bins.size, eps_grid.size)`.
    params : ChiParams
        The parameters the table was built from. Its band limits
        (`kmin`, `band_kmax`) select the bins in `band_bins`.
    """

    fs: float
    M: int
    spd_grid: np.ndarray
    eps_grid: np.ndarray
    bins: np.ndarray
    G: np.ndarray
    params: ChiParams

    @classmethod
    def build(cls, params: ChiParams, fs, spd_max, spd_min=None, dspd=0.01,
              n_eps=81, eps_min=1e-12, eps_max=1e-4):
        """Tabulate `G` for one sampling rate.

        Parameters
        ----------
        params : ChiParams
            Supplies `nsec`, `kmin`, `kmax_cap`, `fmax_cap`, `min_spd`, the
            preemphasis and antialias constants and `nu`, `D`, `q`.
        fs : float
            Sampling rate, Hz.
        spd_max : float
            m/s, upper end of the speed grid. `r_of` clamps above it.
        spd_min : float or None, optional
            m/s, lower end of the speed grid. Defaults to `params.min_spd`.
        dspd : float, optional
            m/s, speed grid step, default 0.01.
        n_eps, eps_min, eps_max : optional
            Log eps grid, default 81 points from 1e-12 to 1e-4 W/kg.

        Returns
        -------
        BandResponse
        """
        spd_min = params.min_spd if spd_min is None else spd_min
        n_spd = max(int(np.ceil((spd_max - spd_min) / dspd)) + 1, 2)
        spd_grid = np.linspace(spd_min, max(spd_max, spd_min + dspd), n_spd)
        eps_grid = np.logspace(np.log10(eps_min), np.log10(eps_max), n_eps)
        f, f_bins, K = welch_kernels(fs, params.nsec)
        M = int(round(params.nsec * fs))
        # one bin past fmax_cap, so a window at a slightly higher fs than
        # the table still finds every bin it sums
        bins = np.flatnonzero((f_bins > 0) & (f_bins < params.fmax_cap + fs / M))

        # factor(f_i) / factor(f): gain and spd cancel, leaving h
        def h(x):
            return (x**2 * preemphasis_inverse(x, params.R24, params.R25, params.R22, params.C19)
                    / antialias(x, fs, params.antialias))

        f = f[1:]
        Kw = K[bins][:, 1:] * (f[1] - f[0]) / h(f) * h(f_bins[bins])[:, None]
        G = np.empty((n_spd, bins.size, n_eps))
        for j, spd in enumerate(spd_grid):
            B = np.stack([spectrum(f / spd, eps, 1.0, params.nu, params.D, params.q)
                          for eps in eps_grid], axis=1)
            G[j] = 6 * params.D * (fs / M) / spd * (Kw @ B)
        return cls(fs=float(fs), M=M, spd_grid=spd_grid, eps_grid=eps_grid,
                   bins=bins, G=G, params=params)

    def band_bins(self, spd, fs):
        """Welch bin indices `run_range` sums for a window.

        Parameters
        ----------
        spd : float
            Fall rate, m/s.
        fs : float
            The window's sampling rate, Hz.

        Returns
        -------
        numpy.ndarray of int
            Bins with `kmin < i fs / (M spd) < kmax(spd)`, the selection of
            `modfish.chi.spectra.integrate`.
        """
        # the frequencies `welch` returns, so a bin on a limit lands on
        # the same side as in `integrate`
        k = np.fft.rfftfreq(self.M, 1 / float(fs))[self.bins] / spd
        return self.bins[(k > self.params.kmin) & (k < band_kmax(spd, self.params))]

    def r_of(self, spd, fs):
        """Resolved fraction at every `eps_grid` point for one window.

        Parameters
        ----------
        spd : float
            Fall rate, m/s. `G` is interpolated linearly and clamped to
            `spd_grid`; the bins summed always follow the unclamped `spd`.
        fs : float
            The window's sampling rate, Hz.

        Returns
        -------
        numpy.ndarray
            Shape `(eps_grid.size,)`.
        """
        s = float(np.clip(spd, self.spd_grid[0], self.spd_grid[-1]))
        j = int(np.searchsorted(self.spd_grid, s, side="right")) - 1
        j = min(max(j, 0), self.spd_grid.size - 2)
        w = (s - self.spd_grid[j]) / (self.spd_grid[j + 1] - self.spd_grid[j])
        sel = np.isin(self.bins, self.band_bins(spd, fs))
        return (1 - w) * self.G[j, sel].sum(axis=0) + w * self.G[j + 1, sel].sum(axis=0)
