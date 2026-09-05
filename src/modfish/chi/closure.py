"""Alford and Pinkel (2000b) closure, appendix (A2) to (A6), with the
resolved fraction from the Batchelor band fraction (spec "Closure")."""

import gsw
import numpy as np
import xarray as xr
from scipy.ndimage import uniform_filter1d

from modfish.chi.batchelor import FractionTable
from modfish.chi.config import FLAG_N2, FLAG_RCLIP, FLAG_RRHO, ChiParams
from modfish.utils import sampling_interval


def _at(centers_ns, time_ns, values):
    return np.interp(centers_ns, time_ns, values, left=np.nan, right=np.nan)


def stratification(ctd: xr.Dataset, centers, params: ChiParams) -> xr.Dataset:
    """n2, Tz, Sz, alpha, beta and Rrho over `closure_window` at `centers`.

    Boxcar means of `p`, `t`, `SP`, `sgth0` over the window, gradients of
    the smoothed series against the smoothed pressure (dbar taken as m),
    interpolated to the window centers. `alpha`, `beta` from gsw at the
    smoothed state.

    Parameters
    ----------
    ctd : xarray.Dataset
        CTD record on dim `time`, with data variables `p`, `t`, `SP`,
        `sgth0`, `lon`, `lat`.
    centers : numpy.ndarray
        datetime64, window centers at which the gradients and derived
        variables are evaluated.
    params : ChiParams
        Supplies `closure_window` (s), `g` and `rho_0`.

    Returns
    -------
    xarray.Dataset
        On dim `time` (coordinate `centers`), with data variables `n2`
        (1/s^2), `Tz` (K/m), `Sz` (1/m), `alpha` (1/K), `beta` (kg/g) and
        `Rrho` (dimensionless).
    """
    fs = 1.0 / sampling_interval(ctd["time"].values)
    size = max(int(round(params.closure_window * fs)), 3)
    time_ns = ctd["time"].values.astype("datetime64[ns]").astype("int64")
    centers_ns = np.asarray(centers).astype("datetime64[ns]").astype("int64")

    def smooth(name):
        return uniform_filter1d(ctd[name].values.astype(float), size, mode="nearest")

    ps, ts, SPs, sg = smooth("p"), smooth("t"), smooth("SP"), smooth("sgth0")
    dp = np.gradient(ps)
    with np.errstate(divide="ignore", invalid="ignore"):
        n2 = params.g / params.rho_0 * np.gradient(sg) / dp
        Tz = np.gradient(ts) / dp
        Sz = np.gradient(SPs) / dp
    small = np.abs(dp) < 1e-6
    n2[small] = Tz[small] = Sz[small] = np.nan

    p_c = _at(centers_ns, time_ns, ps)
    t_c = _at(centers_ns, time_ns, ts)
    SP_c = _at(centers_ns, time_ns, SPs)
    lon_c = _at(centers_ns, time_ns, ctd["lon"].values.astype(float))
    lat_c = _at(centers_ns, time_ns, ctd["lat"].values.astype(float))
    SA = gsw.SA_from_SP(SP_c, p_c, lon_c, lat_c)
    CT = gsw.CT_from_t(SA, t_c, p_c)
    alpha = gsw.alpha(SA, CT, p_c)
    beta = gsw.beta(SA, CT, p_c)
    Tz_c, Sz_c, n2_c = (_at(centers_ns, time_ns, v) for v in (Tz, Sz, n2))
    with np.errstate(divide="ignore", invalid="ignore"):
        Rrho = alpha * Tz_c / (beta * Sz_c)
    out = xr.Dataset(
        dict(n2=("time", n2_c), Tz=("time", Tz_c), Sz=("time", Sz_c),
             alpha=("time", alpha), beta=("time", beta), Rrho=("time", Rrho)),
        coords=dict(time=np.asarray(centers).astype("datetime64[ns]")))
    out["n2"].attrs = dict(long_name="buoyancy frequency squared", units="1/s^2")
    out["Tz"].attrs = dict(long_name="vertical temperature gradient", units="K/m")
    out["Sz"].attrs = dict(long_name="vertical practical salinity gradient", units="1/m")
    out["alpha"].attrs = dict(long_name="thermal expansion coefficient", units="1/K")
    out["beta"].attrs = dict(long_name="haline contraction coefficient", units="kg/g")
    out["Rrho"].attrs = dict(long_name="density ratio alpha Tz / (beta Sz)", units="1")
    return out


def solve_epsilon(chi_pe_hat, kmax, table: FractionTable, gamma):
    """Invert `2 gamma eps r(eps, kmax) = chi_pe_hat` per window.

    Parameters
    ----------
    chi_pe_hat : array_like
        W/kg, the potential-energy dissipation rate implied by the
        measured chi (`stratification` and `closure`'s `chi_pe`). NaN or
        non-positive entries return NaN.
    kmax : array_like
        cpm, integration upper limit at each window, same shape as
        `chi_pe_hat`.
    table : FractionTable
        Precomputed `r(eps, kmax)` used to invert the closure without
        integrating the Batchelor spectrum per window.
    gamma : float
        Mixing efficiency.

    Returns
    -------
    eps, r : numpy.ndarray
        W/kg and the resolved fraction (clipped at 1); NaN where the input
        is NaN or not positive.
    clipped : numpy.ndarray of bool
        True where `r` was clipped at 1, including inputs below the
        table's range (`eps` at the grid floor, `r` = 1).
    """
    chi_pe_hat = np.asarray(chi_pe_hat, dtype=float)
    kmax = np.asarray(kmax, dtype=float)
    eps = np.full(chi_pe_hat.shape, np.nan)
    r = np.full(chi_pe_hat.shape, np.nan)
    clipped = np.zeros(chi_pe_hat.shape, dtype=bool)
    log_eps = np.log10(table.eps_grid)
    for j in range(chi_pe_hat.size):
        x = chi_pe_hat[j]
        if not (np.isfinite(x) and x > 0 and np.isfinite(kmax[j])):
            continue
        r_col = table.r_of(kmax[j])
        g = 2 * gamma * table.eps_grid * r_col
        if x <= g[0]:
            # below the table: the band resolves everything, r = 1, eps at the floor
            eps[j], r[j], clipped[j] = table.eps_grid[0], 1.0, True
            continue
        if x >= g[-1]:
            eps[j], r[j] = table.eps_grid[-1], r_col[-1]
            continue
        le = np.interp(np.log10(x), np.log10(g), log_eps)
        eps[j] = 10**le
        r[j] = np.interp(le, log_eps, r_col)
        if r[j] > 1.0:
            r[j], clipped[j] = 1.0, True
    return eps, r, clipped


def closure(chi, kmax, strat: xr.Dataset, params: ChiParams, table: FractionTable) -> xr.Dataset:
    """chi_pe, eps_chi, r and chi_tot at every window, with flags 8, 16, 128.

    Parameters
    ----------
    chi : array_like
        K^2/s, measured temperature-variance dissipation per window.
    kmax : array_like
        cpm, integration upper limit per window, same shape as `chi`.
    strat : xarray.Dataset
        `stratification` output on dim `time`, with `n2`, `alpha` and
        `Rrho`; supplies the `time` coordinate of the returned Dataset.
    params : ChiParams
        Supplies `g`, `rho_0`, `rrho_factor_max` and `gamma`.
    table : FractionTable
        Passed through to `solve_epsilon`.

    Returns
    -------
    xarray.Dataset
        On dim `time`, with data variables `chi_pe` (W/kg), `eps_chi`
        (W/kg), `r` (dimensionless), `chi_tot` (K^2/s) and `flag`
        (uint8, bits `FLAG_RCLIP` (8), `FLAG_N2` (16) and `FLAG_RRHO`
        (128) only).
    """
    chi = np.asarray(chi, dtype=float)
    n2 = strat["n2"].values
    alpha = strat["alpha"].values
    flag = np.zeros(chi.shape, dtype=np.uint8)
    with np.errstate(divide="ignore", invalid="ignore"):
        factor = 1.0 + 1.0 / strat["Rrho"].values ** 2
    capped = ~np.isfinite(factor) | (factor > params.rrho_factor_max)
    factor = np.where(capped, params.rrho_factor_max, factor)
    flag[capped & np.isfinite(chi)] |= FLAG_RRHO
    bad_n2 = ~(n2 > 0)
    flag[bad_n2] |= FLAG_N2
    with np.errstate(divide="ignore", invalid="ignore"):
        chi_pe = params.g * alpha / params.rho_0 * chi / n2 * factor
    chi_pe[bad_n2] = np.nan
    eps, r, clipped = solve_epsilon(chi_pe, kmax, table, params.gamma)
    flag[clipped] |= FLAG_RCLIP
    chi_tot = chi / r
    out = xr.Dataset(
        dict(chi_pe=("time", chi_pe), eps_chi=("time", eps), r=("time", r),
             chi_tot=("time", chi_tot), flag=("time", flag)),
        coords=dict(time=strat["time"].values))
    out["chi_pe"].attrs = dict(long_name="measured potential energy dissipation", units="W/kg")
    out["eps_chi"].attrs = dict(long_name="dissipation rate from the chi closure", units="W/kg")
    out["r"].attrs = dict(long_name="resolved fraction of the Batchelor variance", units="1")
    out["chi_tot"].attrs = dict(long_name="chi corrected for the unresolved band", units="K^2/s")
    return out
