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

<!-- ## unreleased -->
<!-- ### Breaking changes -->
 
<!-- ### New Features -->

<!-- ### Bug fixes -->

<!-- ### Documentation -->

<!-- ### Internal Changes -->


## 2024.12.0
-   Created package.
