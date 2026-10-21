# History

## unreleased

### New Features
-   `modfish.fctd.depth_edges` (formerly the private `grid._depth_edges`,
    which stays as an alias) and `modfish.utils.gap_runs`, the index
    slices of a time axis between gaps.
-   The `chi` group carries a per-window `direction` ("down", "up", or
    "none" outside every cast). `spd` is a speed magnitude, so upcast
    windows were only distinguishable by joining against `casts`.
-   `ChiParams` rejects a non-positive `window`, `step`, `nsec` or
    `min_spd`, an `nsec` longer than `window`, and `kmin >= kmax_cap`.
-   `GridParams.closure_exclude_flags` names `chi_flag` bits that keep a
    window out of the gridded `chi_tot`, `eps_chi` and `r` means while it
    still contributes to `chi`. Default 0, no change. The grid attrs carry
    the value. For windows whose `rrho_capped` bit is set by
    salinity-gradient noise (MOTIVE SBE49 0664).
-   `tc.downup_separation` takes a `pmax` so the down/up cost can be
    restricted to a pressure band.
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
-   Added `utils.nsqfcn`, buoyancy frequency squared of one profile
    after Gregg and Alford, ported from `gvpy.ocean` so that `modfish`
    no longer depends on it. `io.add_n2` computes `n2` again, so
    `load_fctd_grid` and `load_epsi_grid` carry it as they did before
    the gvpy dependency was cut.
-   `concat_l0` and `FCTDConfig` take a `groups` selection of the L0
    groups to load, e.g. `("ctd", "gps")`, so a deployment whose `efe`
    stream would not fit in memory can be processed for its CTD product
    alone. The L1 root attrs record the selection. Default unchanged:
    every group is loaded.
-   Added `modfish.chi` subpackage: microconductivity chi from the L0
    `efe/c1` stream (`add_chi`, a `/chi` group on 2 s windows at a 0.25 s
    step: band-limited `chi`, `kmax`, flags) with the Alford and Pinkel
    closure (`chi_tot`, `eps_chi`, resolved fraction `r` from the Batchelor
    band fraction, in-situ `n2`, `Tz`, `Sz`, `Rrho`) as an optional stage,
    the gain fit against the SBE 49 (`fit_gain`), a `ChiParams` section on
    `FCTDConfig`, `grid_casts` binning of the chi group, and
    `fctd.add_chi_to_products` for existing L1 files. Every correction is a
    named PSD factor (inverse preemphasis, SOM sinc^4 antialias, spectral
    derivative); no time-domain filtering, no first difference, no `c`/`p`
    constants. `gain` is S/m per volt of `c1`; the shipboard constant 22
    is per ADC fraction, 8.8 on this scale. Design:
    `plans/2026-09-04-chi-design.md`.
-   `modfish.chi` subtracts a measured instrument noise spectrum inside
    the band integral in place of the scalar cut that used to truncate
    the band. The old cut, raw PSD below `snr * noise_floor` with
    `noise_floor` = 1e-9 V^2/Hz read off a bench spectrum at 50 to 90
    Hz, a range the band never reaches, discarded 50.48 percent of the
    2025 windows. `ChiParams` drops `snr` and `noise_floor` and gains
    `noise` (a reference to the measured floor, resolved by
    `modfish.chi.noise.resolve`) and `phi_max` (the noise-fraction
    threshold for flagging a window noise-dominated). `/chi` and the
    depth-gridded product both gain `phi`, the noise fraction of the
    band integral per window. `chi_flag` bit 2 changes meaning from
    "kmax set by the noise cut" to "noise-dominated". Negative bins in
    the subtraction are summed as they stand, never clipped, so `chi`
    can now come out negative.

### Bug fixes
-   The chi closure's resolved fraction `r` is the fraction of the
    Batchelor variance the window's Welch band sum recovers on average
    (`chi.batchelor.BandResponse`). It replaces `FractionTable`, the exact
    integral of the Batchelor spectrum from `kmin` to the band's upper
    edge, which disagreed with the band sum in two ways: the lower edge of
    the summed bins is `k_first - dk/2`, not `kmin`, and the per-segment
    detrend and Hamming leakage take power out of the lowest bins (bin 1
    keeps 0.73 of a white spectrum). On synthetic c1 the old `r` was up to
    10 % above the band sum at 1e-10 W/kg and 0.5 m/s, 3 % at 1e-8 W/kg,
    under 1.5 % above 1 m/s. The new `r` matches it within 1.5 %. On the
    2025 d07 FrontStation cast 50 (median fall rate 3.5 m/s) the median of
    `chi_tot` and `eps_chi` is unchanged, with 98 % of windows within
    -2 % to +3 %. `closure` and `solve_epsilon` take each window's `spd`
    and `fs` in place of `kmax`. The chi group attrs carry
    `r_model = "welch_response"`.
-   `tc.find_lags` returned the edge lag (+4.875 s at 16 Hz with
    `window=80`) for every window when `t` or `c` held a single NaN:
    `filtfilt` spread the NaN over the whole record. NaN are now filled
    before the low-pass, and only windows holding one get a NaN lag.
-   `tc.phase_correct` raises a `ValueError` naming the problem when no
    sample lies inside `tcfit`, or when the fit range or the span deeper
    than 1 dbar holds fewer than 2 segments of `N` samples. It used to
    fail with `IndexError`, a LAPACK error or `ZeroDivisionError`. A
    2-segment record is corrected when `return_spectra` is False.
-   `utils.nsqfcn` (and so `io.add_n2`) returned all-NaN N^2 for
    profiles spaced between `dp/2` and `dp` (e.g. 6 dbar data with the
    default `dp=10`): the two-point Hanning window is all zeros. Such
    data are now used unsmoothed, as gvpy does since its 2b4bf2a.
-   `utils.datetime_linspace` returned a constant array for datetime64[ns]
    input, the unit `mattime_to_datetime64` returns. The fast-time
    fallback in `io.load_fctd_raw_mat` therefore stamped every
    microconductivity sample with the first time.
-   `utils.loadmat` raised `IndexError` on any top-level 2-D numeric
    variable (its struct check indexed `range(size)` along the first
    axis), e.g. a stacked `uConductivity` matrix.
-   `io.load_fctd_raw_time_series` dropped the file holding `start` and
    returned data past `end`. It now returns exactly `[start, end]`.
-   Along-track `dist` in the grid loaders went NaN after the first
    missing position. Missing positions are interpolated first.
-   Grid loader metadata: `SA` is g/kg (was kg/m^3), `sgth` is labeled
    sigma_0, and Epsifish `th` is potential temperature (was labeled
    Theta, the symbol for conservative temperature).
-   `io.load_fctd_grid(what=...)` raises on an unknown value instead of
    returning the full grid. `utils.mattime_to_datetime64` accepts a
    pandas Series.
-   `modraw.read` no longer loses a whole file to one bad stream. A record
    timestamp past the datetime64[ns] range (one flipped high hex digit)
    raised `OverflowError`; such records are now dropped and counted in
    the group's `n_bad_time`. A decoder that raises `ValueError` (GGA
    without a ZDA, no usable payload, power-on timestamps) now leaves only
    its group out, with a warning, as missing calibration already did.
-   Packets whose trailing checksum fails are dropped by the SB49, ECOP
    and EFE4 decoders and counted in each group's `n_bad_checksum`.
    `Packet` carries `chk_ok`. None of the 8.7 million frames in the 690
    MOTIVE 2025 d07 files fails its checksum.
-   A header with only part of the SBE49 coefficients no longer skips the
    `$DCAL` fallback and fails with `KeyError` (new
    `modraw.header.cal_complete`).
-   `modraw.convert` writes `<stem>.nc.tmp` and renames it, so an
    interrupted write leaves no truncated `.nc` for a later
    `overwrite=False` run to skip.
-   `read` adds no empty `vnav` group for `$VNAV` frames without `$VNMAR`,
    `vnav` counts sentences with a bad checksum, and the root
    `block_time` is datetime64[ns] as documented (it was ms under pandas
    3).
-   `fctd.find_casts` smooths and differentiates pressure per stretch
    between time gaps longer than `CastParams.smooth` (16 s). A pressure
    jump across a gap made a fake cast (a 250 dbar "downcast" across a
    600 s gap in a synthetic record), and casts next to a gap spanned it.
    The median filter now replicates edge samples (`medfilt` zero-padded,
    which bent the smoothed pressure for half a window at every record
    and gap edge). On MOTIVE d07 the count stays at 495; the five casts
    next to the three gaps over 16 s now end or start where the data do.
-   `fctd.concat._drop_misstamped` keeps the longest time-ordered
    subsequence after the leading-block rule. One record stamped far
    early dropped every record before it (500 of 1000 in a test file).
-   A GPS gap longer than `gps_max_gap` no longer sets `lat`, and with it
    `depth`, to NaN. `lat` is interpolated or held from the nearest
    fixes, `lon` stays NaN there, and the `ctd` attrs carry
    `n_lon_masked` and a `position_note`. `latitude_fallback` now applies
    only when the record has no usable fix.
-   Longitude is unwrapped before interpolating, so a track across 180
    degrees no longer interpolates through 0.
-   `fctd.grid.depth_edges` builds centers from an integer count, so a
    `dz` such as 0.1 no longer adds a bin past `depth_max`. It raises a
    clear error on all-NaN depth and on `depth_min > depth_max`.
-   `grid_casts` derives its depth range from cast samples only, leaves a
    cast without labeled samples NaN/NaT instead of raising, and no
    longer warns once per cast on all-NaN `lon`.
-   `FCTDConfig.keep_counts` is removed: `True` always failed in
    `make_l1`. A config that sets it gets an error naming
    `concat_l0(..., keep_counts=True)`. `groups: ctd` (a string) is read
    as one group instead of `("c", "t", "d")`.
-   `chi.fit.fit_gain_casts` takes the CTD sampling rate from the record
    span (`utils.sampling_interval`). The median step of 1 ms-quantized
    16 Hz stamps reads 62 ms, which biased the fitted gain 2 % high on a
    synthetic record and chi about twice that.
-   The `chi` record floor estimate (`record_floor_n`) inverts the
    chi-square quantile at the degrees of freedom of the current Welch
    setup (new `chi.spectra.welch_dof`) instead of the shipped floor's
    11.04. With `nsec = 0.25` the old value read the floor about 1.4x
    high. Default settings at 325.52 Hz give 11.03, so default runs are
    unchanged.
-   `FLAG_N2` is set only on windows with a finite `chi`, like
    `FLAG_RRHO`. Slow, empty-band and no-environment windows in inverted
    stratification no longer carry bit 16.
-   `concat_l0` removes two acquisition-restart defects from `ctd`
    (#41). Records at a file's start that are stamped later than records
    following them (0.2 to 2.3 s late on MOTIVE, two to four records) are
    dropped per file, and a first sample after a gap longer than 0.5 s is
    dropped when its pressure departs from the extrapolation of the next
    two samples by more than 5 dbar (stale readings 30 to 50 dbar off,
    with `t` and `c` equally stale). The counts are stamped as
    `n_misstamped` and `n_stale_after_gap` on the `ctd` attrs. On the 23
    MOTIVE products this removes 16 samples from five of them and leaves
    the other 18 unchanged.
-   `tc.correct`, `tc.response_correction` and `tc.thermal_mass_correction`
    lay the record on a uniform time grid before running, so a time gap in
    a concatenated record no longer feeds a step into the low-pass, the
    derivative term and the thermal-mass recursion. On the 2025 MOTIVE d09
    record a 276 s gap had produced spikes of several degrees in corrected
    temperature and salinity errors above 0.01 over about a minute after
    the gap. They also now raise `ValueError` on a time axis that is not
    strictly increasing. `correct` with every argument at its default
    used to return an unmodified copy regardless.
-   The sampling rate every `modfish.tc` function derives from the time
    axis, and the one `modfish.fctd` uses for cast detection and `dPdt`,
    now comes from one helper, `modfish.utils.sampling_interval`: the
    mean step over the gap-free stretches of `time`. It used to be the
    median step, which on the FCTD's 1 ms-quantized 16 Hz stamps reads
    63.00 ms against a 62.50 ms mean, so every rate came out 0.8 % low.
    The bias cancelled while estimation and application shared the
    convention; it stops cancelling the moment a fitted time constant is
    compared with a manufacturer value or one fitted elsewhere. The 2025
    MOTIVE T-C values (`lag` 0.03 s, `tau_t` 0.05 s, `alpha` 0.010,
    `beta` 1/12 s) were fitted under the median convention; applied
    through the new one they map to sample units 0.8 % differently,
    inside the widths of the fitted cost troughs (lag trough 0.08 to
    0.09 s, `alpha` pinned to a factor 1.4), so the cruise config values
    stand.
-   `response_correction` now subtracts the straight line through the
    record's first and last gap-filled sample before its whole-record
    `rfft` and adds it back analytically. The FFT treats the record as
    periodic, so a mismatch between the two ends (15 degC across a deep
    cast) wrapped into a discontinuity that the fractional-delay sinc
    kernel rang on through the whole record, distorting the
    salinity-roughness cost used to fit `lag` and `tau_t`.
-   `io.load_fctd_raw_mat` labels `c` as S/m. The attr read mS/cm while
    the values were S/m (a median of 3.3 S/m on files from both MOTIVE
    cruises, giving a practical salinity of 34.5 when read as S/m),
    which `modfish.modraw` had labelled correctly all along.
-   `find_lags` no longer raises `IndexError` when a window's correlation
    peak lands on the first or last lag of the correlation array. The
    sub-sample quadratic refinement needs a neighbor on each side of the
    peak, and where one is missing the raw peak lag is returned unrefined.
    The case turned up on 1 of 152 real d09 casts.
-   `modraw.read` now stamps the header setup fields (`survey`,
    `experiment`, `cruise`, `vehicle`, `fishflag`, `serialnum`,
    `gm_time`) onto the `ctd` group's own attrs, alongside the SBE49
    calibration coefficients that were already there. They had been
    written to the DataTree root only, so a consumer that opens the
    `ctd` group on its own, as the L0 pipeline does, never saw
    `serialnum`, `vehicle`, `cruise` or `gm_time`.
-   `modraw.read` converts a 5 MB MOTIVE `.modraw` file in about 0.3 s,
    down from about 7 s. `decode_sb49` and `decode_ecop` now stack the
    records of a whole stream and decode each hexadecimal field once,
    through a byte-to-digit lookup table, where they used to decode
    every block on its own (44,000 numpy calls on arrays of a few rows
    per file). Output is bit-identical, checked on files from both
    MOTIVE cruises.

### Documentation
-   Documented the `modfish.modraw` subpackage in its module docstring
    (stream/group table, example, quality-tally semantics) and corrected
    `modraw_tag_format.md` against byte-level findings from MOTIVE 2024/2025
    files: EFE4 channel count and rate on FCTD files, ECOP actual block rate
    and payload quirks, GPS timestamp prefix and talker naming, the
    unverified ALTI frame layout, and the header length field's unsafe use
    as a read boundary.
-   Documented the `modfish.fctd` subpackage and `modfish.tc` module in
    their docstrings, including three parameter choices left open at the
    time for the T-C correction analysis (FCTD reprocessing
    sub-project 3): the
    phase-matching low-pass cutoff `f0` (6 Hz in `gvpy`, 9 in the orphaned
    `modfish.utils` copy), the thermal-mass `alpha`/`beta` pair, and
    whether `t`/`c` should be renamed. All three were settled while the
    T-C correction chain was built, and the docstrings now record the
    outcome.
-   Documented `TCParams` and `tc.correct` for the explicit-parameter
    chain: field-by-field provenance comments on `TCParams` (`lag`,
    `tau_t`, `lowpass`, `alpha`, `beta`, `pr`), and the correction order,
    gap-fill behavior, and no-op defaults on `tc.correct`. `make_l1`'s docstring
    now states that `t`/`c` carry a `processing` attr and that the
    default `FCTDConfig()` applies no correction.
-   `tc.thermal_mass_correction` Notes: the coefficients in the code equal
    the SBE Data Processing manual's (0.029867 both at 16 Hz). Only
    `dc/dT` differs, which `dcdt` selects. The table of published values
    now quotes Lueck & Picklo (1990) correctly (`1/beta` 10 s, `alpha`
    0.028 at 0.85 m/s and 0.020 to 0.025 at 1.7 to 1.9 m/s) and drops an
    unsourced row.
-   `tc.atanfit` weights by coherence to the eighth power (as the MATLAB
    original does), not the fourth. `phase_correct`'s `tcfit` is
    `[lower, upper]`. The conductivity spectrum in `plot_spectra` is
    labeled in (S/m)^2/Hz. `tc.correct`, `response_correction` and
    `thermal_mass_correction` accept time gaps, which their parameter
    text denied. Pointers to a private processing repository are
    replaced by the values they referred to.

### Internal Changes
-   `tc.downup_separation` takes its default `pmin` from
    `_fit_range_default`, shared with the other estimators.
-   Removed the unused and partly broken `io.load_epsi_profile`,
    `load_epsi_raw`, `load_epsi_raw_mat`, `load_epsi_ctd_raw` (it returned
    the Epsi stream, not the CTD) and `plot_epsi_profile` (undefined
    `gv`), plus `utils.process_file_path` and `loadmat(onevar=...)`.
-   matplotlib is an optional extra, `modfish[plot]`, used only by
    `tc.plot_spectra`.
-   Removed `modraw.load_ctd`, `load_ctd_time_series`, `load_gps_time`,
    `block_counts` and `header.read_body`. They framed only the bytes past
    the declared header and so returned less data than `modraw.read`, the
    entry point. Hex, timestamp and checksum helpers moved to
    `modraw._util`.
-   `tc.downup_separation` indexes casts by their time runs and
    `tc.thermal_mass_correction` evaluates its recursion with
    `scipy.signal.lfilter`; a 242-point thermal-mass cost map on the full
    2025 d12 record drops from about 45 min to about 10 min.
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
    `lowpass`, and `pr` replace `phase_match`, `N`, `f0`, and `tcfit`.
    Every field now defaults to a no-op, so `FCTDConfig()` leaves `t`/`c`
    equal to `t_raw`/`c_raw`. `FCTDConfig.from_dict`
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
