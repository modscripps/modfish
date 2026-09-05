"""Batchelor temperature-gradient spectrum and the resolved band fraction.

The form follows `EPSILOMETER/EPSILON/process/batchelor.m` (Lien 1992) with
`erfc` in the bracket. The band fraction `r(eps, kmax)` replaces the
`c eps^p` power law of Alford and Pinkel (2000): spec decision 2.
"""

import dataclasses

import numpy as np
from scipy.special import erfc


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
    nu, D, q : float
        Kinematic viscosity, thermal diffusivity, universal constant.

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


@dataclasses.dataclass
class FractionTable:
    """`r(eps, kmax)` on a log grid of `eps` and a linear grid of `kmax`.

    Attributes
    ----------
    eps_grid : numpy.ndarray
        W/kg, increasing.
    kmax_grid : numpy.ndarray
        cpm, increasing.
    r : numpy.ndarray
        Shape `(eps_grid.size, kmax_grid.size)`.
    kmin : float
        cpm.
    """

    eps_grid: np.ndarray
    kmax_grid: np.ndarray
    r: np.ndarray
    kmin: float

    @classmethod
    def build(cls, kmin, kmax_max, nu, D, q, n_eps=81, n_kmax=24,
              eps_min=1e-12, eps_max=1e-4):
        """Build a FractionTable by evaluating band_fraction on a parameter grid.

        Parameters
        ----------
        kmin : float
            Minimum wavenumber, cpm.
        kmax_max : float
            Upper end of kmax_grid, cpm. The grid runs from `kmin + 0.5`
            (the narrowest band worth tabulating) to `kmax_max` and
            `r_of` clamps outside it, so pass the largest `kmax` the
            caller can produce, not the nominal cap
            (`modfish.chi.pipeline._fraction_table`).
        nu : float
            Kinematic viscosity, m^2/s.
        D : float
            Thermal diffusivity, m^2/s.
        q : float
            Universal constant.
        n_eps : int, optional
            Number of eps grid points, default 81.
        n_kmax : int, optional
            Number of kmax grid points, default 24.
        eps_min : float, optional
            Minimum eps for grid, W/kg, default 1e-12.
        eps_max : float, optional
            Maximum eps for grid, W/kg, default 1e-4.

        Returns
        -------
        FractionTable
            Table with eps_grid (log-spaced from eps_min to eps_max),
            kmax_grid (linear from kmin + 0.5 to kmax_max), and r array.
        """
        eps_grid = np.logspace(np.log10(eps_min), np.log10(eps_max), n_eps)
        kmax_grid = np.linspace(kmin + 0.5, kmax_max, n_kmax)
        r = np.empty((n_eps, n_kmax))
        for i, eps in enumerate(eps_grid):
            for j, kmax in enumerate(kmax_grid):
                r[i, j] = band_fraction(eps, kmin, kmax, nu, D, q)
        return cls(eps_grid=eps_grid, kmax_grid=kmax_grid, r=r, kmin=kmin)

    def r_of(self, kmax):
        """Column of r at kmax, with linear interpolation clamped to grid.

        Parameters
        ----------
        kmax : float
            Wavenumber at which to interpolate, cpm.

        Returns
        -------
        numpy.ndarray
            Array of shape (eps_grid.size,) with r values interpolated
            linearly in kmax, clamped to [kmax_grid[0], kmax_grid[-1]].
        """
        kmax = float(np.clip(kmax, self.kmax_grid[0], self.kmax_grid[-1]))
        j = np.searchsorted(self.kmax_grid, kmax, side="right") - 1
        j = min(max(j, 0), self.kmax_grid.size - 2)
        w = (kmax - self.kmax_grid[j]) / (self.kmax_grid[j + 1] - self.kmax_grid[j])
        return (1 - w) * self.r[:, j] + w * self.r[:, j + 1]
