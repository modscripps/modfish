"""L1 stage: positions, depth, dPdt, cast labels, T-C corrections, and
derived variables.

Consumes the deployment-level DataTree `modfish.fctd.concat.concat_l0`
produces and builds the L1 product: the `ctd` group gains position, depth,
`dPdt`, per-sample cast labels, T-C-corrected `t`/`c` (alongside the raw
values), and TEOS-10 derived variables. `efe`/`ecop` are carried through
cast-tagged; `gps`/`alti` are carried through unchanged. See
`plans/2026-09-01-fctd-pipeline-design.md` for the stage order and
rationale.
"""

import logging

import gsw
import numpy as np
import xarray as xr
from scipy.ndimage import uniform_filter1d

from modfish import tc
from modfish.fctd.casts import _sampling_rate, casts_to_dataset, find_casts, label_casts
from modfish.fctd.config import FCTDConfig

logger = logging.getLogger(__name__)

#: Prandtl number and empirical scale factor passed to
#: `tc.viscous_heating_temperature_correction`, matching that function's
#: own defaults (`scale=2.0` is the undocumented empirical factor its
#: docstring flags as provenance-less; recorded here so `_apply_tc` can
#: stamp the actual values used into `attrs["corrections"]`).
_VISCOUS_PR = 15.0
_VISCOUS_SCALE = 2.0

#: Human-readable note stamped onto `ctd.attrs` when position falls back
#: to a fixed latitude: `lon` is NaN in that case, which propagates into
#: every gsw call that needs a real position.
_FALLBACK_SALINITY_NOTE = (
    "latitude_source=fallback leaves lon all-NaN; SA, CT, and sgth0 "
    "(which require a real position) come out all-NaN too. Only SP "
    "(computed from t, c, p alone) is valid among the derived salinity "
    "variables."
)


def _nearest_gap_seconds(time: np.ndarray, ref_time: np.ndarray) -> np.ndarray:
    """Time gap, seconds, from each `time` sample to the nearest `ref_time` sample.

    Parameters
    ----------
    time : np.ndarray
        Timestamps to evaluate, dtype datetime64.
    ref_time : np.ndarray
        Sorted reference timestamps, dtype datetime64, at least one entry.

    Returns
    -------
    np.ndarray
        Absolute gap in seconds to the nearest entry of `ref_time`, one per
        `time` sample.
    """
    idx = np.searchsorted(ref_time, time)
    idx_lo = np.clip(idx - 1, 0, ref_time.size - 1)
    idx_hi = np.clip(idx, 0, ref_time.size - 1)
    gap_lo = np.abs((time - ref_time[idx_lo]) / np.timedelta64(1, "s"))
    gap_hi = np.abs((time - ref_time[idx_hi]) / np.timedelta64(1, "s"))
    return np.minimum(gap_lo, gap_hi)


def _add_position(ctd: xr.Dataset, gps: xr.Dataset | None, config: FCTDConfig) -> xr.Dataset:
    """Interpolate GPS `lon`/`lat` onto ctd time, or fall back to a fixed latitude.

    Stretches of ctd time further than `config.gps_max_gap` from the
    nearest GPS fix are masked: `lon` is set to NaN, and `lat` is set to
    `config.latitude_fallback` if given, else NaN. Samples beyond the GPS
    record's own time span are extrapolated as a constant equal to the
    nearest endpoint (`np.interp`'s default clamping behavior); that
    clamp only matters up to `gps_max_gap` past the GPS range, since
    stretches beyond it are masked the same as any other gap.

    Parameters
    ----------
    ctd : xr.Dataset
        `ctd` group, with a `time` coordinate.
    gps : xr.Dataset or None
        `gps` group (`lat`, `lon` on `time`), or None if the group is
        absent from the input tree.
    config : FCTDConfig
        Uses `gps_max_gap` and `latitude_fallback`.

    Returns
    -------
    xr.Dataset
        `ctd` with `lon`, `lat` added and `attrs["latitude_source"]` set to
        "gps" or "fallback".

    Raises
    ------
    ValueError
        If there is no usable GPS fix (group absent, empty, or all-NaN) and
        `config.latitude_fallback` is None.
    """
    time = ctd["time"].values
    n = time.size

    has_gps = gps is not None and gps.sizes.get("time", 0) > 0
    valid = None
    if has_gps:
        gps_lat = np.asarray(gps["lat"].values, dtype=float)
        gps_lon = np.asarray(gps["lon"].values, dtype=float)
        valid = np.isfinite(gps_lat) & np.isfinite(gps_lon)
        has_gps = bool(valid.any())

    if not has_gps:
        if config.latitude_fallback is None:
            raise ValueError(
                "no GPS and no latitude_fallback: cannot position the ctd record"
            )
        logger.warning(
            "no usable GPS fix; falling back to latitude_fallback=%s "
            "(longitude set to NaN)",
            config.latitude_fallback,
        )
        lat = np.full(n, config.latitude_fallback, dtype=float)
        lon = np.full(n, np.nan, dtype=float)
        ctd = ctd.assign(lat=("time", lat), lon=("time", lon))
        ctd["lat"].attrs = dict(long_name="latitude", units="degrees_north")
        ctd["lon"].attrs = dict(long_name="longitude", units="degrees_east")
        ctd.attrs["latitude_source"] = "fallback"
        ctd.attrs["latitude_source_note"] = _FALLBACK_SALINITY_NOTE
        return ctd

    gps_time = gps["time"].values[valid]
    gps_lat = gps_lat[valid]
    gps_lon = gps_lon[valid]

    time_i8 = time.astype("datetime64[ns]").astype("int64")
    gps_time_i8 = gps_time.astype("datetime64[ns]").astype("int64")

    lat = np.interp(time_i8, gps_time_i8, gps_lat)
    lon = np.interp(time_i8, gps_time_i8, gps_lon)

    gap = _nearest_gap_seconds(time, gps_time)
    masked = gap > config.gps_max_gap
    if masked.any():
        if config.latitude_fallback is not None:
            lat = np.where(masked, config.latitude_fallback, lat)
        else:
            lat = np.where(masked, np.nan, lat)
        lon = np.where(masked, np.nan, lon)

    ctd = ctd.assign(lat=("time", lat), lon=("time", lon))
    ctd["lat"].attrs = dict(long_name="latitude", units="degrees_north")
    ctd["lon"].attrs = dict(long_name="longitude", units="degrees_east")
    ctd.attrs["latitude_source"] = "gps"
    return ctd


def _add_depth(ctd: xr.Dataset) -> xr.Dataset:
    """Add depth (m, positive down) from pressure and latitude.

    Parameters
    ----------
    ctd : xr.Dataset
        Must have `p` (dbar) and `lat` (degrees_north) on `time`.

    Returns
    -------
    xr.Dataset
        `ctd` with `depth` added.
    """
    depth = -gsw.z_from_p(ctd["p"].values, ctd["lat"].values)
    ctd = ctd.assign(depth=("time", depth))
    ctd["depth"].attrs = dict(long_name="depth", units="m", positive="down")
    return ctd


def _add_dpdt(ctd: xr.Dataset, config: FCTDConfig) -> xr.Dataset:
    """Add smoothed pressure rate of change `dPdt` (dbar/s).

    Parameters
    ----------
    ctd : xr.Dataset
        Must have `p` and `time`.
    config : FCTDConfig
        Uses `dpdt_smooth`, s.

    Returns
    -------
    xr.Dataset
        `ctd` with `dPdt` added.
    """
    fs = _sampling_rate(ctd["time"].values)
    dpdt = np.gradient(ctd["p"].values) * fs
    window = max(round(config.dpdt_smooth * fs), 1)
    dpdt = uniform_filter1d(dpdt, window, mode="nearest")
    ctd = ctd.assign(dPdt=("time", dpdt))
    ctd["dPdt"].attrs = dict(long_name="pressure rate of change", units="dbar/s")
    return ctd


def _apply_tc(ctd: xr.Dataset, config: FCTDConfig) -> xr.Dataset:
    """Apply the configured T-C sensor response corrections, in order.

    Stashes `t_raw`/`c_raw` before any correction. Phase matching
    (`config.tc.phase_match`) runs `tc.phase_correct`, whose output lands
    on a trimmed time axis (segmenting loses the edges) that is otherwise
    NaN-free. Thermal-mass (`config.tc.thermal_mass`) and viscous-heating
    (`config.tc.viscous_heating`) corrections, when also enabled, run on
    that same trimmed, NaN-free working dataset rather than on a
    pre-reindexed one: `thermal_mass_correction`'s recursive filter
    propagates the first NaN it sees through every later sample
    (`tc.py:632-633`), so feeding it the trimmed axis's NaN-padded
    reindexed form would silently turn the whole `c` record NaN. The
    working dataset is reindexed back onto the full ctd time axis once, at
    the end, only if phase matching ran (with `thermal_mass`/
    `viscous_heating` alone, the input is already the full, finite axis
    and no reindex is needed). `t`/`c` attrs, which xarray arithmetic and
    `reindex` are not reliably guaranteed to carry through every step, are
    saved before any correction and reattached at the end.

    Parameters
    ----------
    ctd : xr.Dataset
        Must have `t`, `c`, `p`, `lon`, `lat`, `dPdt` on `time`.
    config : FCTDConfig
        Uses `tc` (a `TCParams`).

    Returns
    -------
    xr.Dataset
        `ctd` with `t_raw`, `c_raw` added, `t`/`c` corrected per config
        (attrs preserved from the pre-correction `t`/`c`), and
        `attrs["corrections"]` set to a human-readable summary of the
        steps applied, each with its parameters (or "none").
        `attrs["tau1"]`/`attrs["L1"]` are set when phase matching runs.
    """
    ctd = ctd.assign(t_raw=ctd["t"].copy(deep=True), c_raw=ctd["c"].copy(deep=True))
    ctd["t_raw"].attrs = dict(ctd["t"].attrs)
    ctd["c_raw"].attrs = dict(ctd["c"].attrs)

    t_attrs = dict(ctd["t"].attrs)
    c_attrs = dict(ctd["c"].attrs)

    tc_cfg = config.tc
    corrections = []
    original_time = ctd["time"]

    if tc_cfg.phase_match:
        work = tc.phase_correct(ctd, N=tc_cfg.N, f0=tc_cfg.f0, tcfit=tc_cfg.tcfit)
        ctd.attrs["tau1"] = float(work.attrs["tau1"])
        ctd.attrs["L1"] = float(work.attrs["L1"])
        corrections.append(f"phase_correct(N={tc_cfg.N}, f0={tc_cfg.f0}, tcfit={tc_cfg.tcfit})")
    else:
        work = ctd

    if tc_cfg.thermal_mass:
        work = tc.thermal_mass_correction(work, alpha=tc_cfg.alpha, beta=tc_cfg.beta)
        corrections.append(
            f"thermal_mass_correction(alpha={tc_cfg.alpha}, beta={tc_cfg.beta})"
        )

    if tc_cfg.viscous_heating:
        dT = tc.viscous_heating_temperature_correction(
            work["dPdt"].values, Pr=_VISCOUS_PR, scale=_VISCOUS_SCALE
        )
        work = work.assign(t=work["t"] - dT)
        corrections.append(
            f"viscous_heating_temperature_correction(Pr={_VISCOUS_PR}, "
            f"scale={_VISCOUS_SCALE})"
        )

    if tc_cfg.phase_match:
        work = work.reindex(time=original_time)

    ctd["t"] = work["t"]
    ctd["c"] = work["c"]
    ctd["t"].attrs = t_attrs
    ctd["c"].attrs = c_attrs

    ctd.attrs["corrections"] = "; ".join(corrections) if corrections else "none"
    return ctd


def _add_derived(ctd: xr.Dataset) -> xr.Dataset:
    """Add TEOS-10 derived variables: `SP`, `SA`, `CT`, `sgth0`.

    Parameters
    ----------
    ctd : xr.Dataset
        Must have `t` [degC], `c` [S/m], `p` [dbar], `lon`, `lat` on `time`.

    Returns
    -------
    xr.Dataset
        `ctd` with `SP` (practical salinity), `SA` (absolute salinity),
        `CT` (conservative temperature), and `sgth0` (potential density
        anomaly referenced to 0 dbar) added.
    """
    SP = gsw.SP_from_C(ctd["c"].values * 10.0, ctd["t"].values, ctd["p"].values)
    SA = gsw.SA_from_SP(SP, ctd["p"].values, ctd["lon"].values, ctd["lat"].values)
    CT = gsw.CT_from_t(SA, ctd["t"].values, ctd["p"].values)
    sgth0 = gsw.sigma0(SA, CT)

    ctd = ctd.assign(
        SP=("time", SP), SA=("time", SA), CT=("time", CT), sgth0=("time", sgth0)
    )
    ctd["SP"].attrs = dict(long_name="practical salinity", units="1")
    ctd["SA"].attrs = dict(long_name="absolute salinity", units="g/kg")
    ctd["CT"].attrs = dict(long_name="conservative temperature", units="degC")
    ctd["sgth0"].attrs = dict(
        long_name="potential density anomaly referenced to 0 dbar", units="kg/m^3"
    )
    return ctd


def make_l1(tree: xr.DataTree, config: FCTDConfig | None = None) -> xr.DataTree:
    """Build the L1 product from a concatenated L0 DataTree.

    Order of operations on the `ctd` group: positions, depth, `dPdt`, cast
    labels, T-C corrections (needs positions and `dPdt` in place first),
    then TEOS-10 derived variables. See `_add_position`, `_add_depth`,
    `_add_dpdt`, `_apply_tc`, and `_add_derived` for each step's contract.

    Parameters
    ----------
    tree : xr.DataTree
        Output of `modfish.fctd.concat.concat_l0`: a `ctd` group plus any
        of `efe`, `ecop`, `gps`, `alti`.
    config : FCTDConfig or None, optional
        Pipeline configuration. Defaults to `FCTDConfig()`.

    Returns
    -------
    xr.DataTree
        Groups `ctd` (16 Hz product: `t`, `c`, `t_raw`, `c_raw`, `p`,
        `depth`, `lon`, `lat`, `dPdt`, `SP`, `SA`, `CT`, `sgth0`, coord
        `cast`) and `casts` (per-cast table), plus `efe`, `ecop`, `gps`,
        `alti` carried through when present in `tree` (`efe`/`ecop`
        cast-tagged, `gps`/`alti` unchanged). Root attrs and `ctd` attrs
        carry the input tree's concat provenance (`files`, `n_files`).

    Raises
    ------
    ValueError
        Propagated from `_add_position` when there is no usable GPS fix
        and `config.latitude_fallback` is None.

    Notes
    -----
    When positioning falls back to `config.latitude_fallback`
    (`ctd.attrs["latitude_source"] == "fallback"`), `lon` is left all-NaN
    rather than fabricated, since a fallback gives no real longitude. `SA`
    (and everything downstream of it: `CT`, `sgth0`) needs a real position
    and comes out all-NaN as a result; `SP` does not depend on position
    and stays valid. This is intended, not a bug: `_add_position` stamps
    the same explanation onto `ctd.attrs["latitude_source_note"]` for
    anyone inspecting the product without having read this docstring.
    """
    if config is None:
        config = FCTDConfig()

    ctd = tree["ctd"].to_dataset()
    ctd.attrs = {**tree.attrs, **ctd.attrs}
    gps = tree["gps"].to_dataset() if "gps" in tree.children else None
    efe = tree["efe"].to_dataset() if "efe" in tree.children else None
    ecop = tree["ecop"].to_dataset() if "ecop" in tree.children else None
    alti = tree["alti"].to_dataset() if "alti" in tree.children else None

    ctd = _add_position(ctd, gps, config)
    ctd = _add_depth(ctd)
    ctd = _add_dpdt(ctd, config)

    casts_df = find_casts(ctd["p"].values, ctd["time"].values, config.casts)
    ctd = label_casts(ctd, casts_df)
    if efe is not None:
        efe = label_casts(efe, casts_df, time_ref=ctd["time"].values)
    if ecop is not None:
        ecop = label_casts(ecop, casts_df, time_ref=ctd["time"].values)
    casts_ds = casts_to_dataset(casts_df, ctd["time"].values)

    ctd = _apply_tc(ctd, config)
    ctd = _add_derived(ctd)

    groups = {"ctd": ctd, "casts": casts_ds}
    if efe is not None:
        groups["efe"] = efe
    if ecop is not None:
        groups["ecop"] = ecop
    if gps is not None:
        groups["gps"] = gps
    if alti is not None:
        groups["alti"] = alti

    l1_tree = xr.DataTree.from_dict({f"/{k}": v for k, v in groups.items()})
    l1_tree.attrs = dict(tree.attrs)
    return l1_tree
