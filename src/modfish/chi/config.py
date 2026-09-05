"""Microconductivity chi parameters and the product flag bits."""

import dataclasses

ANTIALIAS_KINDS = ("som_sinc4", "ap00_sinc2")

FLAG_SLOW = 1  # fall rate below min_spd
FLAG_NOISE = 2  # kmax set by the noise cut, below both caps
FLAG_EMPTY = 4  # fewer than min_bins wavenumber bins survived
FLAG_RCLIP = 8  # closure solve at an edge of the eps table (r held at the floor value below it, NaN above it)
FLAG_N2 = 16  # n2 not positive, closure undefined
FLAG_RAIL = 32  # a c1 sample in the window sat at the ADC rail
FLAG_NOENV = 64  # window center outside the ctd record or NaN environment
FLAG_RRHO = 128  # (1 + 1/Rrho^2) capped at rrho_factor_max

FLAG_MEANINGS = (
    "1 slow, 2 noise_limited, 4 band_empty, 8 r_clipped, 16 n2_not_positive, "
    "32 rail, 64 no_closure_inputs, 128 rrho_capped"
)


@dataclasses.dataclass
class ChiParams:
    """Microconductivity chi parameters, see `plans/2026-09-04-chi-design.md`.

    Parameters
    ----------
    enabled : bool
        Run `add_chi` in `process_deployment`.
    gain : float or None
        S/m per V, the fitted product of network scale and gain. Required
        when enabled; no default so a missing fit fails loudly.
    gain_source : str
        Provenance stamped into the product attrs.
    antialias : str
        "som_sinc4" (SOM ADC, power sinc^8(pi f/fs)) or "ap00_sinc2"
        (Alford and Pinkel 2000, sinc^2(pi k/k_N)).
    noise_floor : float
        V^2/Hz of the raw channel (bench, 1e-9).
    snr : float
        Noise cut factor; 0 disables the cut.
    kmin, kmax_cap : float
        cpm, integration band limits.
    fmax_cap : float
        Hz, frequency cap; kmax <= fmax_cap / spd.
    min_spd : float
        m/s, windows below return NaN.
    window, step, nsec : float
        s, window length, window step, Welch segment length.
    min_bins : int
        Fewer surviving bins raise the band-empty flag.
    closure : bool
        Compute chi_tot, eps_chi, r and the stratification variables.
    closure_window : float
        s, boxcar over which n2, Tz, Sz are evaluated.
    rrho_factor_max : float
        Cap on (1 + 1/Rrho^2).
    gamma, g : float
        Closure constants (mixing efficiency, gravity).
    rho_0 : float
        kg/m^3, reference density. Used by `stratification` for
        `n2 = (g / rho_0) d(sigma_0)/dp`; the closure prefactor is
        `(g alpha)^2 / n2` and carries no `rho_0`.
    nu, D, q : float
        Batchelor constants (kinematic viscosity, thermal diffusivity,
        universal constant).
    R24, R25, R22, C19 : float
        SBE 7 preemphasis network (ohm, ohm, ohm, farad).
    gap : float
        s, a step above this splits ranges.
    rail_lo, rail_hi : float
        V, rail thresholds on c1.
    spd_smooth : float
        s, fall-rate smoothing window.
    """

    enabled: bool = False
    gain: float | None = None
    gain_source: str = ""
    antialias: str = "som_sinc4"
    noise_floor: float = 1e-9
    snr: float = 3.0
    kmin: float = 1.0
    kmax_cap: float = 12.5
    fmax_cap: float = 50.0
    min_spd: float = 0.5
    window: float = 2.0
    step: float = 0.25
    nsec: float = 0.5
    min_bins: int = 4
    closure: bool = True
    closure_window: float = 31.25
    rrho_factor_max: float = 5.0
    gamma: float = 0.2
    rho_0: float = 1026.0
    g: float = 9.8
    nu: float = 1.3e-6
    D: float = 1.4e-7
    q: float = 3.7
    R24: float = 1e6
    R25: float = 577e3
    R22: float = 266.1
    C19: float = 1e-6
    gap: float = 0.01
    rail_lo: float = 0.001
    rail_hi: float = 2.499
    spd_smooth: float = 1.0

    def __post_init__(self):
        if self.antialias not in ANTIALIAS_KINDS:
            raise ValueError(
                f"antialias must be one of {ANTIALIAS_KINDS}, got {self.antialias!r}"
            )
        if self.enabled and self.gain is None:
            raise ValueError("chi.enabled is true but chi.gain is not set")
