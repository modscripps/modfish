"""FCTD pipeline configuration dataclasses and loading utilities."""

import dataclasses
from pathlib import Path

import yaml


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
    """Temperature-conductivity correction parameters.

    Parameters
    ----------
    phase_match : bool
        Enable phase-matching correction
    N : int
        samples, segment length (cruise-1 notebook: 2**7)
    f0 : float
        LP cutoff Hz (gvpy; orphaned modfish copy had 9)
    tcfit : tuple[float, float] | None
        dbar, upper/lower pressure limit for the phase fit. None: add_tcfit_default
    thermal_mass : bool
        Enable thermal mass correction
    alpha : float
        dimensionless, thermal anomaly amplitude. SBE manual placeholder
    beta : float
        1/s, inverse relaxation time constant. SBE manual placeholder
    viscous_heating : bool
        derivation is for unpumped UCTD
    """

    phase_match: bool = True
    N: int = 128  # segment length (cruise-1 notebook: 2**7)
    f0: float = 6.0  # LP cutoff Hz (gvpy; orphaned modfish copy had 9)
    tcfit: tuple[float, float] | None = None  # None: add_tcfit_default
    thermal_mass: bool = False
    alpha: float = 0.03  # SBE manual placeholder
    beta: float = 1 / 7  # SBE manual placeholder
    viscous_heating: bool = False  # derivation is for unpumped UCTD


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
            If unknown keys are provided or if nested sections are not dicts.
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
            if field.name in d and not isinstance(d[field.name], dict):
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
