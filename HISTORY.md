# History

## unreleased

### New Features
-   Added `modfish.modraw` subpackage: a binary frame scanner plus per-tag
    decoders for the FCTD `.modraw` tag set (`SB49` CTD, `EFE4`
    microconductivity/accelerometer, `ECOP` Tridente fluorometer, `GPGGA`/
    `GPZDA` GPS, `ALTI` altimeter), `read()` assembling every decodable
    stream into one `xr.DataTree`, and `convert()` for batch `.modraw` to
    NetCDF conversion.
-   Added `modfish.fctd` subpackage: deployment-level L0 concatenation
    (`concat_l0`), cast detection (`find_casts`, `label_casts`,
    `casts_to_dataset`), the L1 stage (`make_l1`: positions, depth, dPdt,
    cast labels, T-C corrections, TEOS-10 derived variables), per-cast
    depth gridding (`grid_casts`), a typed YAML-loadable configuration
    (`FCTDConfig`), and a deployment driver (`process_deployment`)
    chaining concat -> L1 -> grid and writing both products.
-   Added `modfish.tc`: T-C sensor response corrections (phase matching,
    thermal-mass, viscous-heating) consolidated from `gvpy.mod`, which
    had extended `ctdproc`'s dual-sensor implementation to the
    single-sensor FCTD, carrying ctdproc's NumPy 2 reshape fix
    (ctdproc commit 5e75198).
-   Added `tc.correct` to `modfish.tc`: an explicit-parameter T-C
    correction chain (zero-phase low-pass on `t`/`c`, sensor response on
    `t` as a whole-record transfer function in `lag` and `tau_t`,
    time-domain thermal mass on `c`, viscous heating on `t`, each
    independently skippable) that preserves the input time axis and
    stamps `processing` and `corrections` attrs naming the steps
    applied. Every argument defaults to a no-op, so a cruise config opts
    into each step. `FCTDConfig.tc`/`make_l1` now drive this chain in
    place of the per-segment phase-matching path.
-   Added T-C parameter estimators to `modfish.tc`: `find_lags`,
    `lag_tau_cost_map`, `salinity_roughness`, `downup_separation`,
    `rosette_rms`, and `thermal_mass_cost_map`, for fitting `lag`,
    `tau_t`, `alpha`, and `beta` against a deployment record.

### Bug fixes
-   `response_correction` now subtracts the straight line through the
    record's first and last gap-filled sample before its whole-record
    `rfft` and adds it back analytically. The FFT treats the record as
    periodic, so a mismatch between the two ends (15 degC across a deep
    cast) wrapped into a discontinuity that the fractional-delay sinc
    kernel rang on through the whole record, distorting the
    salinity-roughness cost used to fit `lag` and `tau_t`.
-   `find_lags` no longer raises `IndexError` when a window's correlation
    peak lands on the first or last lag of the correlation array. The
    sub-sample quadratic refinement needs a neighbor on each side of the
    peak, and where one is missing the raw peak lag is returned unrefined.
    The case turned up on 1 of 152 real d09 casts.

### Documentation
-   Documented the `modfish.modraw` subpackage in its module docstring
    (stream/group table, example, quality-tally semantics) and corrected
    `modraw_tag_format.md` against byte-level findings from MOTIVE 2024/2025
    files: EFE4 channel count and rate on FCTD files, ECOP actual block rate
    and payload quirks, GPS timestamp prefix and talker naming, the
    unverified ALTI frame layout, and the header length field's unsafe use
    as a read boundary.
-   Documented the `modfish.fctd` subpackage and `modfish.tc` module in
    their docstrings, including the three parameter choices left open for
    the T-C correction analysis (FCTD reprocessing sub-project 3): the
    phase-matching low-pass cutoff `f0` (6 Hz in `gvpy`, 9 in the orphaned
    `modfish.utils` copy), the thermal-mass `alpha`/`beta` pair, and
    whether `t`/`c` should be renamed.
-   Documented `TCParams` and `tc.correct` for the explicit-parameter
    chain: field-by-field provenance comments on `TCParams` (`lag`,
    `tau_t`, `lowpass`, `alpha`, `beta`, `pr`), and the correction order,
    gap-fill behavior, and no-op defaults on `tc.correct`. `make_l1`'s docstring
    now states that `t`/`c` carry a `processing` attr and that the
    default `FCTDConfig()` applies no correction.

### Internal Changes
-   Added cross-validation fixtures and tests against the Matlab and rust
    `.modraw` readers.
-   Removed the orphaned ctdproc-derived T-C block from `modfish.utils`
    and the broken cast-finding stubs (`split_casts`,
    `smooth_pressure_derivative`) from the single-file `modfish.fctd`
    module, both superseded by the new `modfish.fctd` subpackage and
    `modfish.tc`.
-   Added cross-validation tests for the `modfish.fctd` pipeline against
    the Matlab-gridded products for one 2024
    (`24_1120_d11_fctd_to_mooringA`) and one 2025
    (`25_1205_d07_FCTD1_FrontStation`) deployment. Findings recorded in
    `fctd_validation_notes.md`.
-   Registered `-m "not slow"` as the default pytest deselection: the
    cross-validation tests run about 36 minutes with the server mount
    present (`fctd_validation_notes.md`), so they are opt-in via
    `uv run pytest -m slow`.
-   Reshaped `TCParams` around `tc.correct`'s chain: `lag`, `tau_t`,
    `lowpass`, and `pr` replace `phase_match`, `N`, `f0`, and `tcfit`. Every field now defaults to a no-op, so `FCTDConfig()`
    leaves `t`/`c` equal to `t_raw`/`c_raw`. `FCTDConfig.from_dict`
    raises `ValueError` naming the removed key and its replacement when
    a stale cruise config sets any of the four under `tc`.
-   Rewrote `_apply_tc` in `modfish.fctd.l1` as a thin wrapper around
    `tc.correct(ctd, **dataclasses.asdict(config.tc))`. The reindex back
    onto the full time axis is gone, since `tc.correct` never leaves it.
    `tau1`, `L1`, and `tcfit` are no longer stamped onto `ctd.attrs`.

<!-- ## unreleased -->
<!-- ### Breaking changes -->
 
<!-- ### New Features -->

<!-- ### Bug fixes -->

<!-- ### Documentation -->

<!-- ### Internal Changes -->


## 2024.12.0
-   Created package.
