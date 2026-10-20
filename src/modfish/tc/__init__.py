"""
T-C sensor response corrections for SBE49-carrying MOD profilers.

Fish-agnostic: the sampling interval is read from the data and every
rate- or fall-speed-dependent parameter is an explicit argument. Running
`correct` with every argument at its default is a no-op, so a cruise
config opts into each step and no fitted value is buried in this module.
The values fitted for FCTD1 (SBE49 serial 0537) on the 2025 MOTIVE cruise
live in that cruise's processing config.

`correct` applies the chain in a fixed order, each step skippable on its
own: a zero-phase low-pass on `t` and `c` (`lowpassfilter`), the sensor
response on `t` as a whole-record transfer function
`H(f) = (1 + i 2 pi f tau_t) exp(i 2 pi f lag)` (`response_correction`),
the Lueck and Picklo thermal-mass recursion on `c`
(`thermal_mass_correction`), and an optional viscous-heating term on `t`
(`viscous_heating_temperature_correction`). See `correct` for the full
contract and the sign conventions.

The estimators that fit those parameters live here too. `find_lags`
cross-correlates differenced `t` and `c` in short windows and returns the
apparent T-C lag per window with its mean pressure and fall rate.
`phase_correct` fits the same transfer function per segment in the
frequency domain and returns the T-C coherence along with it, which is
what sets the low-pass cutoff. `lag_tau_cost_map` scans a `(lag, tau_t)`
grid scored by `salinity_roughness`, the rms second difference of
practical salinity. `thermal_mass_cost_map` scans an `(alpha, tau)` grid
under a supplied objective, either `downup_separation`, Morison's mean
absolute salinity difference between consecutive down and up casts in
temperature bins, or `rosette_rms`, the rms difference against a
co-located shipboard rosette cast.

The code is split by role. `modfish.tc.chain` holds `correct` and its
steps, `modfish.tc.estimate` the parameter estimators, and
`modfish.tc.phase` the spectral `phase_correct` diagnostic. Every public
name is importable from `modfish.tc` directly.

Lineage: consolidated from gvpy.mod (which extended ctdproc's dual-sensor
implementation to the single-sensor FCTD). ctdproc reference for the
NumPy 2 reshape fix: commit 5e75198.
"""

# private helpers kept importable from `modfish.tc`, where they lived
# before the split (`_fill_gaps` is imported by downstream analysis code)
from modfish.tc._grid import _fill_gaps as _fill_gaps
from modfish.tc._grid import _on_uniform_grid as _on_uniform_grid
from modfish.tc._grid import _uniform_slots as _uniform_slots
from modfish.tc.chain import (
    correct,
    lowpassfilter,
    response_correction,
    thermal_mass_correction,
    viscous_heating_temperature_correction,
)
from modfish.tc.estimate import (
    downup_separation,
    find_lags,
    lag_tau_cost_map,
    rosette_rms,
    salinity_roughness,
    thermal_mass_cost_map,
)
from modfish.tc.phase import add_tcfit_default, atanfit, phase_correct, plot_spectra

__all__ = [
    "add_tcfit_default",
    "atanfit",
    "correct",
    "downup_separation",
    "find_lags",
    "lag_tau_cost_map",
    "lowpassfilter",
    "phase_correct",
    "plot_spectra",
    "response_correction",
    "rosette_rms",
    "salinity_roughness",
    "thermal_mass_correction",
    "thermal_mass_cost_map",
    "viscous_heating_temperature_correction",
]
