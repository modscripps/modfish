# FCTD pipeline validation against the Matlab grids

Evidence behind `tests/test_fctd_crossval.py`. Every number here comes from runs on
samoan on 2026-09-01 with the mod server mounted. Design decisions referenced below
live in `plans/2026-09-01-fctd-pipeline-design.md`.

Two deployments, one per cruise, as the design doc's "Testing and validation" section
asks for. They are not equivalent baselines, and the difference decides how the numbers
should be read.

| | 2024 `24_1120_d11_fctd_to_mooringA` | 2025 `25_1205_d07_FCTD1_FrontStation` |
|---|---|---|
| Matlab chain | `concatenate_and_grid_fctd.m` -> `FastCTD_GridData.m` at `0d0d5e4` | `make_FCTDall_L0` -> `make_FCTDall_L1` -> `FastCTD_GridData` on `master` |
| T-C correction in the baseline | yes, applied inside the gridder | none, `response_match_applied` is 0 everywhere |
| depth from pressure | `sw_dpth(p, 20)` hardcoded | `sw_dpth(p, deployment mean latitude)` |
| what the comparison gives | an upper bound, dominated by the correction difference | a clean comparison of the two chains |
| median \|dt\| | 0.00214 K | 7.82e-6 K |
| median \|dSP\| | 0.00191 | 0.00160 as stored, 7.89e-6 after undoing a temperature-scale bug in the Matlab salinity |

The 2025 row is the one that says whether the python pipeline reproduces the Matlab
chain. It does, to five decimal places in temperature, conductivity, and pressure.

## 2024 d11 deployment and baseline

`24_1120_d11_fctd_to_mooringA`, cruise skq202417s, 2024-11-20 13:48 to 2024-11-21 09:52
UTC. 256 `.modraw` files, 1.2 GB, at 0.725 to 0.730 degN and 139.18 to 139.71 degW.

Baseline: `fctd_mat/FCTDgrid.mat`, written at sea (file mtime 2024-11-21 02:35), 187
profiles on a 4001-point depth axis, 0 to 2000 m at 0.5 m.

The Matlab code that produced it is `MOD_fish_lib` at commit `0d0d5e4` (2024-11-09), path
`FastCTD_MATLAB/fctd_formatting/concatenate_and_grid_fctd.m` calling
`FastCTD_GridData.m` once per direction. Two independent pieces of evidence pin that
version. The saved struct has no `salinity_despike` field, and the merge step copies every
`FCTDdown` field except `tGrid`, so the code that ran never created one.
`salinity_despike` and the Chebyshev depth smoothing of `salinity` entered
`FastCTD_GridData.m` at `2579cce` (2024-12-07), after the file was written. The baseline's
`salinity` is therefore the unsmoothed `sw_salt` of binned C/T/P (`FastCTD_GridData.m:578`
at `0d0d5e4`), which is what the salinity comparison uses.

## 2024 d11 pipeline run

`modfish.modraw.convert(raw, l0, parallel=True)` then `process_deployment` with
`phase_match=False`, `thermal_mass=False`, `viscous_heating=False`,
`latitude_fallback=30.0`, `dz=0.5`. Conversion of the 256 files took 579 s, the pipeline
22 s.

Products: 1,154,404 CTD samples at 16 Hz, 201 casts (96 down, 105 up), depth axis -0.5 to
1046.5 m at 0.5 m (2095 bins). `ctd.attrs["latitude_source"] == "gps"` and no sample has a
NaN longitude, so `latitude_fallback` never fired and the `LAT = 30` equivalent
(`make_FCTDall_L0.m:87`) has no effect on this comparison. Median in-cast fall rate is
3.12 dbar/s, p90 4.37 dbar/s.

## 2024 d11 cast matching

Casts are matched to Matlab grid columns by mutual-nearest profile mean time within 5 min.
The mutual test matters. One-sided nearest assigns 190 of our casts to only 186 distinct
Matlab columns.

- 186 pairs. 186 of 187 Matlab columns (99.5 %), 186 of 201 python casts (92.5 %).
- Paired mean times agree to 0.56 s (median), 2.14 s (p90), 101 s (max).

15 python casts have no Matlab counterpart, all explained:

- 12 full-depth casts (8 to 1023 dbar, ~2000 filled bins each) between 02:08 and 03:19 on
  2024-11-21. `fctd_mat/` holds 239 `EPSI*.mat` files against 256 raw files, and 16 of the
  17 missing conversions fall in 02:08:04 to 03:18:35. The cruise chain never read those
  raw files, so the Matlab grid has a 74 min hole there with a single profile at 02:42:37.
  The 17th missing conversion is at 09:44:03.
- 3 shallow bounces our detector keeps and the Matlab detector rejects: pressure ranges
  18.3-41.5, 40.3-50.1, and 13.2-26.9 dbar in the gridded columns.

One Matlab column (2024-11-21 09:07:37) has no mutual match on our side.

## 2024 d11 median differences

Median over matched casts of the per-cast median absolute difference, python profile
sampled on the Matlab depth axis. "Depth-matched" repeats the comparison after mapping the
Matlab bin centers through `sw_dpth(., 20)` and back out through `gsw.z_from_p` at the
deployment latitude, which removes the depth-axis mismatch described in the next section.

| variable | ours | theirs | median | p90 | max | depth-matched median |
|---|---|---|---|---|---|---|
| temperature [K] | `t` | `temperature` | 0.00214 | 0.00242 | 0.0179 | 0.00056 |
| conductivity [S/m] | `c` | `conductivity` | 3.21e-4 | 4.61e-4 | 4.58e-3 | 1.93e-4 |
| practical salinity | `SP` | `salinity` | 0.00191 | 0.00310 | 0.0183 | 0.00198 |
| sigma-0 [kg/m^3] | `sgth0` | `sgth` | 0.00181 | 0.00257 | 0.0141 | 0.00155 |
| pressure [dbar] | `p` | `pressure` | 0.308 | 0.325 | 0.642 | 0.0416 |

`sgth` is not a Matlab field. `modfish.io.load_fctd_grid` recomputes it with gsw from the
Matlab binned values, so that row compares gsw against gsw on two different input sets.

## 2024 d11 depth axis and the gridder's hardcoded latitude

The 0.31 dbar pressure difference is a depth-axis difference. Within a 0.5 m bin, binned
pressure is fixed by depth alone, so a systematic offset there means the two products put
the same water at different depths.

`FastCTD_GridData.m:286` at `0d0d5e4` sets `myFCTD.depth = sw_dpth(myFCTD.pressure,20)`.
The deployment is at 0.73 degN. `sw_dpth` depends on latitude through the gravity term, and
20 degN against 0.73 degN shifts the depth assigned to a given pressure by 0.65 m at
1046 m. Our `_add_depth` uses `gsw.z_from_p` with per-sample latitude (design doc, "Stage:
L1", step 1). At the true latitude the two formulas agree to 3e-4 m at 1000 dbar, so the
formula choice contributes nothing and the hardcoded 20 accounts for the whole offset.

Undoing it collapses the pressure difference from 0.308 to 0.0416 dbar and the temperature
difference from 0.00214 to 0.00056 K. `test_crossval_d11_depth_coordinate_convention`
asserts both.

The `LAT = 30` fallback in `make_FCTDall_L0.m:87` that the design doc calls out is a
separate site and is not on the code path that produced this baseline.

## 2024 d11 response matching in the Matlab gridder

The design doc records that the cruise-era L1 ships with `apply_response_matching_code = 0`
and concludes that no T-C correction reached any MOTIVE product. That flag lives in
`make_FCTDall_L1.m:14`, a file that did not yet exist at `0d0d5e4`, so this baseline never
went through it. `FastCTD_GridData` carries its own correction and applies it to every
profile before binning:

- `use_old_code = 1` at `FastCTD_GridData.m:245` selects San's legacy frequency-domain code.
- Conductivity gets a gain and phase correction from polynomial fits loaded out of
  `FCTD_SalinityCorrectionFactors_toCond.mat` (`FastCTD_GridData.m:215, 257-268`).
- Temperature, conductivity, and pressure then get a low pass
  `(cos(pi f / 2 f_Ny))**30` with `f_Ny = 8` Hz (`FastCTD_GridData.m:269-271`), half
  amplitude near 1 Hz, which is roughly 3 m at the observed fall rate.
- The downcast coefficients `GainPFit_Dn` / `PhsPFit_Dn` are assigned once
  (`FastCTD_GridData.m:221-222`) and used for upcasts too. `GainPFit_Up` / `PhsPFit_Up`
  exist in the `.mat` file and are never read.

The effect is measurable in the products. RMS of the second difference along depth, median
over the 177 matched casts with at least 200 common finite bins:

| scale | t, ours | t, Matlab | S, ours | S, ours (phase_match=True) | S, Matlab |
|---|---|---|---|---|---|
| 0.5 m | 0.01018 | 0.00651 | 0.01112 | 0.00426 | 0.00113 |
| 1.0 m | 0.02824 | 0.01914 | 0.02218 | 0.00732 | 0.00277 |
| 2.0 m | 0.06902 | 0.05599 | 0.03735 | 0.01214 | 0.00741 |
| 5.0 m | 0.17107 | 0.15812 | 0.05610 | 0.02433 | 0.02153 |

Our uncorrected salinity has as much 0.5 m structure as our temperature (0.0111 against
0.0102), the signature of salinity spiking from the uncorrected T-C mismatch. The Matlab
salinity has a tenth of that.

Rerunning the same comparison with `phase_match=True` (`tau1 = 0.0485`, `L1 = 0.0581`
fitted once for the deployment) moves every residual toward the Matlab product: t 0.00214
to 0.00169 K, `SP` 0.00191 to 0.00127, `sgth0` 0.00181 to 0.00165 kg/m^3. Conductivity is
unchanged at 3.21e-4 S/m. Our phase matching does not smooth temperature (0.5 m t roughness
0.01018 to 0.01127), so the remaining temperature gap is the Matlab low pass.

Consequence for the comparison contract. `phase_match=False` matches the Matlab L1 stage
and does not match the gridded baseline. The residuals in the table above are dominated by
that, so they are an upper bound on the disagreement between the two chains. Sub-project 3,
which chooses correction parameters, is where this gets settled.

## 2025 d07 deployment and baseline

`25_1205_d07_FCTD1_FrontStation`, cruise skq202521s, 2025-12-05 08:03 to 2025-12-07 14:30
UTC, at 1.473 to 1.475 degN and 140.814 to 140.817 degW.

Raw files are picked out of `03_raw_mod_data/Raw_full_cruise` by stem, one per
`FCTD25_*.mat` in `fctd_mat/`, so the python chain reads the file set the Matlab chain
consumed. 690 stems, 690 resolved in `Raw_full_cruise`, 0 skipped, 3.2 GB. The deployment
also has its own `raw/` copy, with 691 files, one of which
(`FCTD25_12_05_205935.modraw`) the Matlab chain never converted. Selecting by stem drops it
and keeps the two chains on the same inputs.

Baseline: `fctd_mat/FCTDgrid.mat`, written 2025-12-07 06:32, 494 profiles on the same
4001-point 0 to 2000 m axis. This one comes from the modern chain, `make_FCTDall_L0` ->
`make_FCTDall_L1` -> `FastCTD_GridData` on `master`, and it carries the fields the 2024
product lacks: `salinity` (Chebyshev depth-smoothed), `salinity_despike` (the unsmoothed
`sw_salt` of the binned c/t/p), `response_match_applied`, and `down`/`up` flags. Comparisons
below use `salinity_despike`. The two salinity fields differ by 0.00063 (median), 0.046
(p99), 11.9 (max), so the choice matters.

`response_match_applied` is 0 at every one of its 1,282,179 finite points, so this baseline
is genuinely correction-free. `test_crossval_d07_no_response_matching_in_baseline` asserts
the flag and, independently, that the two products carry the same amount of bin-scale
structure (0.5 m temperature roughness 0.01094 ours against 0.01096 Matlab, ratio 1.002,
where the same statistic on d11 is 0.64).

## 2025 d07 pipeline run

Same configuration as d11: `phase_match=False`, `thermal_mass=False`,
`viscous_heating=False`, `latitude_fallback=30.0`, `dz=0.5`. Conversion of the 690 files
took 1499 s. `concat_l0` 77 s, `make_l1` 2 s, `grid_casts` 6 s.

Products: 3,039,652 CTD samples, 495 casts (247 down, 248 up), depth axis -1.0 to 1399.0 m
at 0.5 m (2801 bins). `latitude_source` is `gps`, so `latitude_fallback` never fired here
either. Median in-cast fall rate 3.38 dbar/s, p90 4.98 dbar/s.

Consecutive 2025 L0 files overlap by roughly 14 to 15 s. `concat_l0` drops the duplicate
timestamps and logs each overlap. The overlap is in the acquisition, which writes
overlapping files.

## 2025 d07 cast matching and differences

All 494 Matlab columns pair up, against 494 of our 495 casts. Paired mean times agree to
0.88 s (median), 1.72 s (p90), 50.6 s (max). The one python cast without a counterpart is a
36 dbar upcast fragment (1290 to 1327 dbar, 74 filled bins) at the bottom of a turnaround,
which the Matlab detector keeps joined to the ascent that follows it.

| variable | ours | theirs | median | p90 | max |
|---|---|---|---|---|---|
| temperature [K] | `t` | `temperature` | 7.82e-6 | 7.84e-6 | 8.94e-6 |
| conductivity [S/m] | `c` | `conductivity` | 1.91e-9 | 8.74e-9 | 8.82e-9 |
| pressure [dbar] | `p` | `pressure` | 7.47e-5 | 7.54e-5 | 6.07e-4 |
| practical salinity | `SP` | `salinity_despike` | 0.00160 | 0.00165 | 0.00475 |
| practical salinity | `SP` | `salinity_despike` recomputed on ITS-90 | 7.89e-6 | 7.91e-6 | 8.04e-6 |
| sigma-0 [kg/m^3] | `sgth0` | gsw sigma-0 from `salinity_despike` | 0.00126 | 0.00130 | 0.00359 |

Temperature, conductivity, and pressure agree at the level of bin membership. Our L0 is
bit-identical to the Matlab per-file `.mat` (checked on `FCTD25_12_05_080305`: t, p, and c
all zero difference over 4444 samples), so nothing in the reader or the calibration
contributes. The residual comes from which samples land in which 0.5 m bin under the two
depth conventions, `sw_dpth` at the deployment mean latitude against `gsw.z_from_p` per
sample, which here differ by at most 0.0008 m. The sign is consistent with that: ours is
warmer by 8 microkelvin, and a slightly deeper depth axis pulls slightly warmer water into
each bin.

The `sw_dpth(p, 20)` hardcode that dominates the d11 pressure comparison is absent here.
`FastCTD_GridData` on `master` bins on `FCTD.depth` from `make_FCTDall_L0.m:87`,
`sw_dpth(pressure, nanmean(latitude))`, instead of recomputing it. The d11 remap therefore
has nothing to collapse, which
`test_crossval_d07_depth_convention_is_already_aligned` asserts.

## 2025 d07 salinity and the ITS-90 conversion

The 0.0016 salinity offset is a temperature-scale bug in the Matlab salinity, and it is
exact. `salinity_despike` reproduces to 2.8e-14 as
`gsw.SP_from_C(c*10, t/1.00024, p)`, which is PSS-78 evaluated with the ITS-90 temperature
passed straight through as if it were already IPTS-68. PSS-78 is defined on IPTS-68 and the
SBE49 reports ITS-90, so the 1.00024 conversion is required. `MOD_fish_lib` bundles its own
`FastCTD_MATLAB/seawater/sw_sals.m` with `del_T = T - 15` and a `sw_salt.m` header
documenting T as IPTS-68, present since the repo's initial commit, and that copy was first
on the 2025 MATLAB path.

Skipping the conversion leaves the assumed IPTS-68 temperature below the correct one, and
PSS-78 returns a higher salinity for a lower temperature at fixed conductivity ratio, so
the Matlab salinity is **high**. `salinity_despike` minus a correctly computed PSS-78 is
positive at all 1,282,179 finite points: median +0.00159, p1 +0.00089, p99 +0.00473,
minimum +0.00083. The error grows with temperature, from +0.00115 (mean, 3 to 6 degC) to
+0.00266 (10 to 14 degC) to +0.00458 (22 to 27 degC).

The 2024 product does not have the bug. Its `salinity` reproduces to 1.4e-14 as
`gsw.SP_from_C(c*10, t, p)`, so a seawater 3.3 copy with `del_T68 = T * 1.00024 - 15` took
precedence on that run. `test_crossval_d11_salinity_grid` asserts that reproduction, so the
two products' opposite behavior is pinned on both sides. The two cruises' products are on
different temperature scales for salinity. The median offset is 0.00160 at the 2025
deployment's temperatures and would be 0.00189 at the 2024 deployment's.

Recomputing `salinity_despike` from the Matlab binned c/t/p with the conversion brings
agreement with our `SP` to 7.89e-6, the same floor as the temperature comparison, so the
whole 0.0016 is the temperature scale and nothing else.

## Salinity and density formulation

Two differences the design doc anticipated turn out to be numerically irrelevant here, and
one real one shows up in density. Numbers in this section are from the 2024 product.

- **PSS-78 is PSS-78.** `gsw.SP_from_C(c*10, t, p)` and `sw_salt(c*10/sw_c3515, t, p)`
  agree on the 2024 product's own binned values to a maximum of 1.4e-14 over 350,116
  finite points, 350,112 of them bit-identical. The "TEOS-10 against EOS-80" framing does
  not apply to practical salinity. The agreement depends on which `sw_sals.m` is on the
  MATLAB path. See the 2025 section above, where a copy without the temperature-scale
  conversion puts the Matlab salinity 0.0016 high.
- **Per-sample against binned salinity is 1e-6.** Within our own product, `SP` computed per
  sample and then bin-averaged differs from `SP` recomputed from the binned c/t/p by 3.3e-9
  (median) and 2.8e-6 (p99). The design doc's preference for computing before binning
  ("Stage: L1", step 5) is correct in principle and worth about 1e-6 in practice on this
  deployment.
- **Potential density does differ.** The Matlab `density` field is `sw_pden(S,T,P,0)`
  (`FastCTD_GridData.m:579`). Against gsw `sigma0` on identical binned inputs it sits
  0.00514 kg/m^3 low (median), 0.00767 at p99, 0.00802 at most. That is larger than the
  0.00181 kg/m^3 total `sgth0` difference from every other cause combined, so anyone
  comparing our `sgth0` against the Matlab `density` field should expect the EOS offset to
  dominate.

## Intentional differences

Complete list, each with the design decision it comes from
(`plans/2026-09-01-fctd-pipeline-design.md`).

| difference | ours | Matlab | why |
|---|---|---|---|
| depth from pressure | `gsw.z_from_p`, per-sample latitude | `sw_dpth`, at 20 degN in 2024 and at the deployment mean latitude in 2025 | "Stage: L1" step 1: depth via `gsw.z_from_p` with per-sample latitude |
| missing position | `latitude_fallback` with a warning, `lon` left NaN | silent `LAT = 30` | "Stage: L1" step 1: replaces Matlab's silent fallback |
| salinity and density | per sample from c/t/p before binning | from binned C/T/P after gridding | "Stage: L1" step 5, and bug list item "salinity/density derived after binning" |
| equation of state | TEOS-10 (`SA`, `CT`, `sigma0`) | EOS-80 `sw_pden` for density. Practical salinity is PSS-78 on both sides | "Stage: L1" step 5: SP, SA, CT, sigma-0 via gsw |
| depth-smoothed salinity | absent, smoothing is analysis-side | Chebyshev-smoothed `salinity`, absent in the 2024 baseline and present in 2025 | "Stage: grid": dropped, gridded products carry unsmoothed binned values |
| empty bins | NaN, guaranteed | NaN in both products checked, no exact zeros in either `temperature` field | "Stage: grid": empty bins are NaN, never zero (bug B4) |
| out-of-range samples | dropped by the histogram | clamped into the first and last bin by `bindata.m` | "Stage: grid": rejected, never clamped (bug B3) |
| skipped casts | absent from the product | zero columns | "Stage: grid": skipped casts are absent, never zero columns |
| cast bookkeeping | one detector, chronological integer ids, `cast` coordinate on both the time series and the grid | detection run three times, no key linking grid columns to the time series | "Stage: cast detection": one implementation, run once, consumed by both L1 and grid |
| down and up | one grid with a `direction` coordinate | `FCTDgrid` / `FCTDdown` / `FCTDup`, no direction flag in `FCTDgrid` | "Decisions": one grid with a `direction` coordinate replaces the triple |
| minimum cast duration | 10 s in addition to the 10 dbar range | 10 dbar range only | `CastParams.min_duration`, new in this pipeline |
| count variables | dropped at concat, available in the per-file L0 | carried | "Data model": concat drops count variables so `t_raw`/`c_raw` unambiguously mean uncorrected physical values |
| time-gridded product | absent | `tGrid` | "Stage: grid": `tGrid` dropped, buggy and unused downstream |
| T-C correction | explicit config, off for both comparisons | legacy gain/phase plus low pass in the 2024 gridder, off in the 2025 chain | "Stage: L1" step 4: every switch and parameter explicit config |

The empty-bin row is a guarantee, not an observed difference. Neither Matlab `temperature`
field holds a single exact zero (0 of 350,116 finite points in 2024, 0 of 1,282,179 in
2025), because `bindata.m` divides by a count it has already set to NaN. The zero fills the
design doc catalogues do not appear in either product. The pipeline guarantees NaN anyway.
Clamping was not tested, since neither deployment reaches the 2000 m grid edge.

Defects found on the Matlab side are not in this table, because they are not choices. The
2024 gridder's hardcoded `sw_dpth(p, 20)`, its use of the downcast gain and phase
coefficients on upcasts, the 17 raw files the 2024 chain never converted, and the missing
ITS-90 conversion in the 2025 salinity are all listed in the sections above.

## Reproducing

```
uv run pytest tests/test_fctd_crossval.py -q -m slow
```

One module fixture per deployment, each converting its raw files and running the pipeline
once. About 36 minutes together: 10 min for d11 (256 files, 1.2 GB) and 26 min for d07 (690
files, 3.2 GB). `slow` is registered in `pyproject.toml` but not deselected by default, so
a plain `uv run pytest tests/` pays that cost too.

Intermediates run to about 3.7 GB for d11 (1.9 GB of L0 netCDF, 1.8 GB of L1) and 4.8 GB
for d07. pytest keeps the last three numbered tmp roots, which on samoan (`/tmp` is a 16 GB
tmpfs) is enough to fill the filesystem and make unrelated tests fail with
`PermissionError`. Each fixture therefore loads its grid into memory and deletes its tmp
tree before any test runs. The d07 fixture also calls `concat_l0` / `make_l1` /
`grid_casts` directly instead of `process_deployment`, which would write a 4.8 GB L1
netCDF that nothing reads. `test_fctd_pipeline.py` and the d11 fixture cover the driver.
Peak resident memory during the d07 `concat_l0` was about 16 GB.

## Open items

- The 17 raw files the 2024 cruise chain never converted are worth reporting to whoever owns
  the 2024 products. The python pipeline reads all 256.
- The 2025 Matlab salinity is 0.0016 high because `MOD_fish_lib`'s bundled `sw_sals.m`
  skips the ITS-90 conversion. Skipping it puts the assumed IPTS-68 temperature below the
  correct one, and PSS-78 returns a higher salinity for a lower temperature at fixed
  conductivity ratio, so the error is positive at every ocean temperature. Anyone using
  `FCTDgrid.salinity` or `salinity_despike` from the 2025 cruise, or comparing 2024 against
  2025 salinities, needs to know. Worth reporting upstream.
- Whether the 2024 gridder's legacy gain and phase correction is the right target for our
  `phase_correct` parameters is sub-project 3.
