"""FCTD pipeline configuration dataclasses and loading utilities."""

import dataclasses
from pathlib import Path

import yaml

from modfish.chi.config import ChiParams

#: `tc` keys that `TCParams` no longer has. The phase-matching correction
#: they configured was replaced by `lag`, `tau_t` and `lowpass`.
#: `FCTDConfig.from_dict` rejects them by name instead of falling through
#: to the generic unknown-key error, so a stale cruise config fails loudly
#: naming its replacement.
_REMOVED_TC_KEYS = {"phase_match", "N", "f0", "tcfit"}

#: Top-level keys no longer in `FCTDConfig`, with the reason given in the
#: error. `keep_counts=True` left count-typed `t_raw`/`c_raw` in the tree,
#: which `make_l1` always refused.
_REMOVED_TOP_KEYS = {
    "keep_counts": "keep_counts was removed: make_l1 cannot run on a tree "
                   "with kept counts. Call concat_l0(..., keep_counts=True) "
                   "directly to inspect raw counts.",
}


@dataclasses.dataclass
class CastParams:
    """Parameters for cast detection and preprocessing.

    Parameters
    ----------
    smooth : float
        s, length of the median filter and of the boxcar applied to
        pressure before differentiation (Matlab: 256 samples at 16 Hz).
        Time gaps longer than this split the record for cast detection.
    wlim : float
        dbar/s threshold (Matlab: 0.025 dbar/sample at 16 Hz)
    min_range : float
        dbar, minimum cast pressure range (Matlab: 10)
    min_duration : float
        s, minimum cast duration (new; Matlab had none)
    """

    smooth: float = 16.0  # s, dp/dt smoothing (Matlab: 256 samples at 16 Hz)
    wlim: float = 0.4  # dbar/s threshold (Matlab: 0.025 dbar/sample at 16 Hz)
    min_range: float = 10.0  # dbar, minimum cast pressure range (Matlab: 10)
    min_duration: float = 10.0  # s, minimum cast duration (new; Matlab had none)


@dataclasses.dataclass
class TCParams:
    """Temperature-conductivity correction parameters for `tc.correct`.

    Field names match `tc.correct`'s keyword arguments exactly; `_apply_tc`
    calls `tc.correct(ctd, **dataclasses.asdict(config.tc))`.

    Parameters
    ----------
    lag : float
        s, T advance (T lags C). SBE49 manual: 0.0625
    tau_t : float
        s, thermistor time constant; 0 disables
    lowpass : float | None
        Hz, zero-phase low-pass on t, c; None disables
    thermal_mass : bool
        Enable thermal mass correction
    alpha : float
        dimensionless, thermal anomaly amplitude. SBE49 manual
    beta : float
        1/s, inverse relaxation time constant. SBE49 manual
    viscous_heating : bool
        derivation is for unpumped UCTD
    pr : float
        Prandtl number, Larson and Pedersen 1996 at 2 degC
    thermal_mass_profile : dict | None
        Pressure profile of the thermal-mass product `alpha * tau`,
        `{"p": [...], "product": [...]}`; replaces the scalar `alpha` when
        set. See `tc.thermal_mass_correction`.
    """

    lag: float = 0.0  # s, T advance (T lags C). SBE49 manual: 0.0625
    tau_t: float = 0.0  # s, thermistor time constant; 0 disables
    lowpass: float | None = None  # Hz, zero-phase low-pass on t, c; None disables
    thermal_mass: bool = False
    alpha: float = 0.03  # SBE49 manual
    beta: float = 1 / 7  # 1/s, SBE49 manual
    viscous_heating: bool = False  # derivation is for unpumped UCTD
    pr: float = 12.4  # Prandtl number, Larson and Pedersen 1996 at 2 degC
    thermal_mass_profile: dict | None = None  # {"p": [...], "product": [...]}; replaces alpha


@dataclasses.dataclass
class GridParams:
    """Depth gridding parameters.

    Parameters
    ----------
    dz : float
        m (Matlab: 0.5)
    depth_min : float | None
        m, center of the shallowest bin. The bin reaches `dz / 2` above
        it. None: minimum finite depth of the cast samples, rounded down
        to a multiple of `dz`.
    depth_max : float | None
        m, center of the deepest bin. The bin reaches `dz / 2` below it.
        None: maximum finite depth of the cast samples, rounded up to a
        multiple of `dz` (Matlab clamped 0-2000). Samples outside the
        outer bin edges are dropped.
    closure_exclude_flags : int
        `chi_flag` bits that keep a window out of the closure-derived bin
        means (`chi_tot`, `eps_chi`, `r`) on top of the bits excluded from
        every chi mean. `chi`, `kmax` and `phi` are unaffected. Default 0.
        Set 128 (`rrho_capped`) where the cap is driven by salinity-gradient
        noise, as on MOTIVE SBE49 0664.
    """

    dz: float = 0.5  # m (Matlab: 0.5)
    depth_min: float | None = None  # None: from data
    depth_max: float | None = None  # None: from data (Matlab clamped 0-2000)
    closure_exclude_flags: int = 0


@dataclasses.dataclass
class FCTDConfig:
    """Complete FCTD pipeline configuration.

    Parameters
    ----------
    casts : CastParams
        Cast detection and preprocessing parameters
    tc : TCParams
        Temperature-conductivity correction parameters
    grid : GridParams
        Depth gridding parameters
    chi : ChiParams
        Microconductivity chi parameters
    latitude_fallback : float | None
        degrees_north, used, with a warning, when the record has no usable
        GPS fix (`gps` group absent, empty, or without a finite lat/lon
        pair). `lon` is then NaN throughout, and so are SA, CT and sgth0.
        SP and depth stay valid. None (default): `make_l1` raises
        `ValueError` when there is no usable fix.
    gps_max_gap : float
        s, samples further than this from a GPS fix get NaN `lon`; their
        `lat` is still interpolated (or held) from the nearest fixes
    dpdt_smooth : float
        s, dPdt smoothing window
    groups : tuple of str or None
        L0 group names `concat_l0` loads; None loads every group. A single
        string is taken as one group name. Set to
        ``("ctd", "gps")`` to build the CTD product without the `efe`
        stream, which dominates memory.
    """

    casts: CastParams = dataclasses.field(default_factory=CastParams)
    tc: TCParams = dataclasses.field(default_factory=TCParams)
    grid: GridParams = dataclasses.field(default_factory=GridParams)
    chi: ChiParams = dataclasses.field(default_factory=ChiParams)
    latitude_fallback: float | None = None  # used, with a warning, when GPS is absent
    gps_max_gap: float = 300.0  # s, lon NaN beyond this from a fix
    dpdt_smooth: float = 1.0  # s, dPdt smoothing window
    groups: tuple[str, ...] | None = None  # L0 groups concat_l0 loads; None = all

    def __post_init__(self):
        if isinstance(self.groups, str):
            # tuple("ctd") would split it into ("c", "t", "d")
            self.groups = (self.groups,)
        elif self.groups is not None:
            self.groups = tuple(str(g) for g in self.groups)

    @classmethod
    def from_dict(cls, d: dict) -> "FCTDConfig":
        """Create FCTDConfig from a nested dictionary.

        Accepts any subset of known keys and rejects unknown keys for typo protection.

        Parameters
        ----------
        d : dict
            Dictionary with optional keys: "casts", "tc", "grid", "chi", and any top-level fields.

        Returns
        -------
        FCTDConfig
            Configuration object with overrides applied.

        Raises
        ------
        ValueError
            If unknown keys are provided, if nested sections are not dicts,
            if a scalar (non-section) field is given a dict value, if `tc`
            carries a removed key (`phase_match`, `N`, `f0`, `tcfit`), or
            if `keep_counts` is set.
        """
        # Mapping of nested section names to their dataclass types and field sets
        nested_sections = {
            "casts": (CastParams, {f.name for f in dataclasses.fields(CastParams)}),
            "tc": (TCParams, {f.name for f in dataclasses.fields(TCParams)}),
            "grid": (GridParams, {f.name for f in dataclasses.fields(GridParams)}),
            "chi": (ChiParams, {f.name for f in dataclasses.fields(ChiParams)}),
        }

        fctd_config_fields = {f.name for f in dataclasses.fields(FCTDConfig)}

        for key, reason in _REMOVED_TOP_KEYS.items():
            if key in d:
                raise ValueError(reason)

        # Check for unknown keys at top level
        top_level_unknown = set(d.keys()) - fctd_config_fields
        if top_level_unknown:
            raise ValueError(
                f"Unknown keys in FCTDConfig: {', '.join(sorted(top_level_unknown))}"
            )

        # Process nested sections and check for unknown keys
        result_kwargs = {}

        for section_name, (dataclass_type, field_names) in nested_sections.items():
            if section_name in d:
                section_value = d[section_name]

                # If None, skip (acceptable for optional sections)
                if section_value is None:
                    continue

                # Must be a dict; anything else is an error
                if not isinstance(section_value, dict):
                    raise ValueError(
                        f"Expected {section_name} to be a dict, got {type(section_value).__name__}"
                    )

                if section_name == "tc":
                    for key in section_value:
                        if key in _REMOVED_TC_KEYS:
                            raise ValueError(
                                f"tc.{key} was removed; set lag, tau_t and "
                                "lowpass instead"
                            )

                # Check for unknown keys in this section
                unknown_keys = set(section_value.keys()) - field_names
                if unknown_keys:
                    raise ValueError(
                        f"Unknown keys in {section_name}: {', '.join(sorted(unknown_keys))}"
                    )

                result_kwargs[section_name] = dataclass_type(**section_value)

        # Process top-level scalar fields
        for field in dataclasses.fields(FCTDConfig):
            if field.name in nested_sections:
                continue  # Already processed above
            if field.name in d:
                if isinstance(d[field.name], dict):
                    raise ValueError(
                        f"{field.name} is a scalar field, got a dict: {d[field.name]!r}"
                    )
                result_kwargs[field.name] = d[field.name]

        return cls(**result_kwargs)

    @classmethod
    def from_yaml(cls, path) -> "FCTDConfig":
        """Load FCTDConfig from a YAML file.

        Reads a mapping with keys casts/tc/grid/... and creates a configuration object.

        Parameters
        ----------
        path : str or Path
            Path to the YAML file.

        Returns
        -------
        FCTDConfig
            Configuration object loaded from the YAML file.
        """
        path = Path(path)
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data if data is not None else {})
