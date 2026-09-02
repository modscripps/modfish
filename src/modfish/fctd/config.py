"""FCTD pipeline configuration dataclasses and loading utilities."""

import dataclasses
from pathlib import Path

import yaml

#: `tc` keys from before the T-C correction sub-project reshape (task 3);
#: `FCTDConfig.from_dict` rejects them by name instead of falling through
#: to the generic unknown-key error, so a stale cruise config fails loudly
#: naming its replacement.
_REMOVED_TC_KEYS = {"phase_match", "N", "f0", "tcfit"}


@dataclasses.dataclass
class CastParams:
    """Parameters for cast detection and preprocessing.

    Parameters
    ----------
    smooth : float
        s, dp/dt smoothing (Matlab: 256 samples at 16 Hz)
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
    """

    lag: float = 0.0  # s, T advance (T lags C). SBE49 manual: 0.0625
    tau_t: float = 0.0  # s, thermistor time constant; 0 disables
    lowpass: float | None = None  # Hz, zero-phase low-pass on t, c; None disables
    thermal_mass: bool = False
    alpha: float = 0.03  # SBE49 manual
    beta: float = 1 / 7  # 1/s, SBE49 manual
    viscous_heating: bool = False  # derivation is for unpumped UCTD
    pr: float = 12.4  # Prandtl number, Larson and Pedersen 1996 at 2 degC


@dataclasses.dataclass
class GridParams:
    """Depth gridding parameters.

    Parameters
    ----------
    dz : float
        m (Matlab: 0.5)
    depth_min : float | None
        m, grid lower bound. None: from data
    depth_max : float | None
        m, grid upper bound. None: from data (Matlab clamped 0-2000)
    """

    dz: float = 0.5  # m (Matlab: 0.5)
    depth_min: float | None = None  # None: from data
    depth_max: float | None = None  # None: from data (Matlab clamped 0-2000)


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
    latitude_fallback : float | None
        degrees_north, used, with a warning, when GPS is absent
    gps_max_gap : float
        s, max GPS gap to interpolate across
    dpdt_smooth : float
        s, dPdt smoothing window
    keep_counts : bool
        keep *_raw count variables through concat
    """

    casts: CastParams = dataclasses.field(default_factory=CastParams)
    tc: TCParams = dataclasses.field(default_factory=TCParams)
    grid: GridParams = dataclasses.field(default_factory=GridParams)
    latitude_fallback: float | None = None  # used, with a warning, when GPS is absent
    gps_max_gap: float = 300.0  # s, max GPS gap to interpolate across
    dpdt_smooth: float = 1.0  # s, dPdt smoothing window
    keep_counts: bool = False  # keep *_raw count variables through concat

    @classmethod
    def from_dict(cls, d: dict) -> "FCTDConfig":
        """Create FCTDConfig from a nested dictionary.

        Accepts any subset of known keys and rejects unknown keys for typo protection.

        Parameters
        ----------
        d : dict
            Dictionary with optional keys: "casts", "tc", "grid", and any top-level fields.

        Returns
        -------
        FCTDConfig
            Configuration object with overrides applied.

        Raises
        ------
        ValueError
            If unknown keys are provided, if nested sections are not dicts,
            if a scalar (non-section) field is given a dict value, or if
            `tc` carries a key removed in the T-C correction sub-project
            (`phase_match`, `N`, `f0`, `tcfit`).
        """
        # Mapping of nested section names to their dataclass types and field sets
        nested_sections = {
            "casts": (CastParams, {f.name for f in dataclasses.fields(CastParams)}),
            "tc": (TCParams, {f.name for f in dataclasses.fields(TCParams)}),
            "grid": (GridParams, {f.name for f in dataclasses.fields(GridParams)}),
        }

        fctd_config_fields = {f.name for f in dataclasses.fields(FCTDConfig)}

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
                                f"tc.{key} was removed in the T-C correction "
                                "sub-project; set lag, tau_t and lowpass instead"
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
