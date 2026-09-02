"""Cross-validation of the python pipeline against the Matlab grids.

Two deployments, one per cruise, as the design doc's "Testing and
validation" section asks for. Skips unless the mod server is mounted.

- 2024 `24_1120_d11_fctd_to_mooringA` against `fctd_mat/FCTDgrid.mat`,
  written at sea 2024-11-21 by the cruise-era chain
  (`concatenate_and_grid_fctd.m` -> `FastCTD_GridData.m` at `MOD_fish_lib`
  `0d0d5e4`). That gridder applies its own T-C correction, so the
  comparison is an upper bound.
- 2025 `25_1205_d07_FCTD1_FrontStation` against its `fctd_mat/FCTDgrid.mat`,
  written 2025-12-07 by the modern chain (`make_FCTDall_L0` ->
  `make_FCTDall_L1` -> `FastCTD_GridData` on `master`). Its
  `response_match_applied` field is 0 everywhere, so this one is a genuinely
  correction-free comparison and carries the tight tolerances.

Runtime, measured on samoan 2026-09-01: the 2024 fixture converts 256
`.modraw` files (1.2 GB) in 579 s and runs `process_deployment` in 22 s; the
2025 fixture converts 690 files (3.2 GB) in 1499 s and runs the stage chain
in 84 s. About 36 minutes for the module. `slow` tests are not deselected by
default in `pyproject.toml`, so a plain `uv run pytest tests/` pays that
cost.

Comparison contract (task brief): corrections off (`phase_match=False`,
`thermal_mass=False`), `latitude_fallback=30.0` (a no-op on both, each
deployment has continuous GPS), `dz=0.5`, casts matched to Matlab grid
columns by profile mean time. Matching is mutual-nearest, not one-sided
nearest. On d11 the python product has 201 casts against the Matlab
product's 187, and a one-sided match assigns four python casts to Matlab
columns that are already claimed.

Findings behind the tolerances are written up in `fctd_validation_notes.md`
at the repo root. The ones that shape the assertions here:

- The 2024 gridder hardcodes `sw_dpth(pressure, 20)`
  (`FastCTD_GridData.m:286` at `0d0d5e4`), while that deployment sits at
  0.73 degN and the pipeline uses `gsw.z_from_p` with per-sample latitude.
  The two depth axes drift apart by 0.65 m at 1046 m, which is what
  `test_crossval_d11_depth_coordinate_convention` measures and removes. The
  2025 chain takes depth from `make_FCTDall_L0.m:87`,
  `sw_dpth(pressure, mean latitude)`, and the same remap moves nothing
  there.
- The 2024 gridder applies San's legacy frequency-domain response matching
  to every profile (`use_old_code = 1`, `FastCTD_GridData.m:245`): a gain
  and phase correction on conductivity plus a `(cos(pi f / 2 f_Ny))**30` low
  pass on t, c, and p with `f_Ny = 8` Hz. The comparison run has no T-C
  correction, so the 2024 Matlab profiles are the smoother of the two and
  the 2024 residuals are set by that.
- 17 of the 256 raw files were never converted by the 2024 cruise chain (239
  `EPSI*.mat` files in `fctd_mat/`), 16 of them between 02:08:04 and
  03:18:35 on 2024-11-21, so 12 full-depth python casts have no Matlab
  counterpart.
- The 2025 `salinity_despike` is PSS-78 evaluated with the ITS-90
  temperature passed straight through as if it were IPTS-68. MOD_fish_lib
  bundles its own `FastCTD_MATLAB/seawater/sw_sals.m` with `del_T = T - 15`,
  and that copy was first on the 2025 path. The 2024 run used a seawater 3.3
  copy that does convert. Skipping the conversion leaves the assumed IPTS-68
  temperature below the correct one, and PSS-78 returns a higher salinity
  for a lower temperature at fixed conductivity ratio, so the 2025 Matlab
  salinity is high by 0.0016, which `test_crossval_d07_salinity_grid`
  measures both ways.
"""

import pathlib
import shutil

import gsw
import numpy as np
import pytest
import xarray as xr

import modfish
from modfish.fctd import concat_l0, grid_casts, make_l1, process_deployment
from modfish.fctd.config import FCTDConfig, GridParams, TCParams

DEPLOY = pathlib.Path(
    "/mnt/mod-server/MOTIVE/Cruises/skq202417s/05_processed_data/24_1120_d11_fctd_to_mooringA"
)
MATLAB_GRID = DEPLOY / "fctd_mat" / "FCTDgrid.mat"

DEPLOY_2025 = pathlib.Path(
    "/mnt/mod-server/MOTIVE/Cruises/skq202521s/05_processed_data/"
    "25_1205_d07_FCTD1_FrontStation"
)
MATLAB_GRID_2025 = DEPLOY_2025 / "fctd_mat" / "FCTDgrid.mat"
#: Raw files for the 2025 deployment are picked out of the full-cruise
#: directory by stem, one per `FCTD25_*.mat` in `fctd_mat/`, so the python
#: inputs are the file set the Matlab chain consumed. The deployment's own
#: `raw/` copy holds one extra file the Matlab chain never converted, which
#: selecting by stem drops.
RAW_2025 = pathlib.Path(
    "/mnt/mod-server/MOTIVE/Cruises/skq202521s/03_raw_mod_data/Raw_full_cruise"
)

pytestmark = pytest.mark.slow
needs_data = pytest.mark.skipif(not MATLAB_GRID.exists(), reason="mod server not mounted")
needs_data_2025 = pytest.mark.skipif(
    not (MATLAB_GRID_2025.exists() and RAW_2025.is_dir()),
    reason="mod server not mounted (needs both the 2025 grid and Raw_full_cruise)",
)

#: Latitude the 2024 Matlab gridder hardcodes into its depth conversion
#: (`FastCTD_GridData.m:286`, `MOD_fish_lib` `0d0d5e4`).
MATLAB_GRID_LAT = 20.0

#: ITS-90 to IPTS-68 factor. PSS-78 is defined on IPTS-68 and the SBE49
#: reports ITS-90, so `gsw.SP_from_C` applies this internally. Dividing the
#: temperature by it before the call undoes that, reproducing a `sw_salt`
#: that treats its input as already IPTS-68.
T68_FACTOR = 1.00024

#: Cast pairing tolerance on profile mean time, seconds.
MATCH_TOL_S = 300.0


def _sw_dpth(p, lat):
    """Depth, m, from pressure, dbar, after the UNESCO 1983 formula.

    A transcription of `sw_dpth.m` (CSIRO seawater toolbox 3.3), needed to
    reproduce the depth axis the Matlab grid is binned onto. Only the
    standard-ocean part of the formula is used there, so nothing else from
    the toolbox is required.
    """
    x = np.sin(np.abs(lat) * np.pi / 180.0) ** 2
    c1, c2, c3, c4 = 9.72659, -2.2512e-5, 2.279e-10, -1.82e-15
    bot = 9.780318 * (1.0 + (5.2788e-3 + 2.36e-5 * x) * x) + 2.184e-6 * 0.5 * p
    top = ((((c4 * p + c3) * p + c2) * p + c1) * p)
    return top / bot


@pytest.fixture(scope="module")
def crossval(tmp_path_factory):
    """Run the pipeline on d11 once and return it beside the Matlab grid.

    Returns `(ours, theirs, pairs)` where `pairs` is the list of
    `(our cast index, their time index)` mutual-nearest matches within
    `MATCH_TOL_S`.

    The L0 files and the L1 product together run to about 3.7 GB, and
    pytest keeps the last three numbered tmp roots. On a machine with `/tmp`
    on tmpfs that is enough to fill it and make unrelated tests fail with
    `PermissionError`, so the grid is loaded into memory and the tmp tree is
    removed before any test runs.
    """
    if not MATLAB_GRID.exists():
        pytest.skip("mod server not mounted")

    tmp_path = tmp_path_factory.mktemp("fctd_crossval")
    raw = sorted(DEPLOY.glob("raw/*.modraw"))
    assert len(raw) == 256, f"expected 256 raw files, found {len(raw)}"

    modfish.modraw.convert(raw, tmp_path / "l0", parallel=True)
    l0 = sorted((tmp_path / "l0").glob("*.nc"))
    assert len(l0) == len(raw)

    cfg = FCTDConfig(
        tc=TCParams(phase_match=False, thermal_mass=False),
        grid=GridParams(dz=0.5),
        latitude_fallback=30.0,
    )
    _, grid_path = process_deployment(l0, tmp_path / "out", "d11", cfg)

    with xr.open_dataset(grid_path) as ds:
        ours = ds.load()
    shutil.rmtree(tmp_path, ignore_errors=True)

    theirs = modfish.io.load_fctd_grid(MATLAB_GRID)
    return ours, theirs, _mutual_pairs(ours, theirs)


def _mutual_pairs(ours, theirs):
    """Mutual-nearest `(our cast index, their time index)` pairs, within `MATCH_TOL_S`."""
    dt = np.abs(ours.time.data[:, None] - theirs.time.data[None, :]) / np.timedelta64(
        1, "s"
    )
    nearest_j = dt.argmin(axis=1)
    nearest_i = dt.argmin(axis=0)
    return [
        (i, int(nearest_j[i]))
        for i in range(dt.shape[0])
        if nearest_i[nearest_j[i]] == i and dt[i, nearest_j[i]] < MATCH_TOL_S
    ]


@pytest.fixture(scope="module")
def crossval2025(tmp_path_factory):
    """Run the pipeline on the 2025 d07 deployment once, beside its Matlab grid.

    Raw files are picked out of `Raw_full_cruise` by stem, one per
    `FCTD25_*.mat` in the deployment's `fctd_mat/`, so the python inputs are
    the file set the Matlab chain consumed rather than whatever the
    deployment's own `raw/` copy happens to hold. Stems with no `.modraw` in
    `Raw_full_cruise` are skipped and counted (0 of 690 on 2026-09-01).

    Runs `concat_l0` -> `make_l1` -> `grid_casts` directly instead of
    `process_deployment`, which would also write a 4.8 GB L1 netCDF that
    nothing here reads. `test_fctd_pipeline.py` and the d11 fixture already
    cover the driver. The 690 L0 files still take 4.8 GB, so the tmp tree is
    removed before any test runs, as in the d11 fixture. Peak resident
    memory during `concat_l0` was about 16 GB.

    Returns `(ours, theirs, pairs, n_missing_raw)`. `theirs` carries two
    extra variables beside what `load_fctd_grid` reads: `sd`, the
    `salinity_despike` field (the unsmoothed one, while `load_fctd_grid`
    maps the Chebyshev-smoothed `salinity` onto `s`), and `sp_its90`, that
    same quantity recomputed from the Matlab binned c/t/p with the ITS-90
    conversion PSS-78 requires.
    """
    if not (MATLAB_GRID_2025.exists() and RAW_2025.is_dir()):
        pytest.skip("mod server not mounted")

    stems = sorted(p.stem for p in DEPLOY_2025.glob("fctd_mat/FCTD25_*.mat"))
    assert len(stems) > 600, f"only {len(stems)} per-file .mat found"
    raw = [RAW_2025 / f"{s}.modraw" for s in stems]
    n_missing = sum(not p.exists() for p in raw)
    raw = [p for p in raw if p.exists()]
    # A partially populated mount would otherwise reach the len(l0) ==
    # len(raw) check below as 0 == 0 and quietly compare an empty product.
    assert len(raw) > 600, f"only {len(raw)} of {len(stems)} stems found in {RAW_2025}"

    tmp_path = tmp_path_factory.mktemp("fctd_crossval_2025")
    modfish.modraw.convert(raw, tmp_path / "l0", parallel=True)
    l0 = sorted((tmp_path / "l0").glob("*.nc"))
    assert len(l0) == len(raw)

    cfg = FCTDConfig(
        tc=TCParams(phase_match=False, thermal_mass=False),
        grid=GridParams(dz=0.5),
        latitude_fallback=30.0,
    )
    ours = grid_casts(make_l1(concat_l0(l0, keep_counts=cfg.keep_counts), cfg), cfg.grid)
    shutil.rmtree(tmp_path, ignore_errors=True)

    theirs = modfish.io.load_fctd_grid(MATLAB_GRID_2025)
    raw_mat = modfish.utils.loadmat(MATLAB_GRID_2025)["FCTDgrid"]
    sd_full = xr.DataArray(
        np.asarray(raw_mat.salinity_despike),
        dims=("depth", "time"),
        coords={
            "depth": np.asarray(raw_mat.depth),
            "time": modfish.utils.mattime_to_datetime64(raw_mat.time),
        },
    )
    theirs = theirs.assign(sd=sd_full.sel(depth=theirs.depth.data, time=theirs.time.data))
    theirs["sp_its90"] = (
        ("depth", "time"),
        gsw.SP_from_C(theirs.c.data * 10.0, theirs.t.data, theirs.p.data),
    )

    return ours, theirs, _mutual_pairs(ours, theirs), n_missing


def _profile_diffs(ours, theirs, pairs, oname, tname, depth=None):
    """Median absolute difference per matched cast, on the Matlab depth axis.

    `depth` gives the depths at which the python profile is sampled; it
    defaults to the Matlab bin centers. Passing the depths that the Matlab
    bin centers correspond to under the pipeline's own depth convention
    takes the depth-axis mismatch out of the comparison.
    """
    if depth is None:
        depth = theirs.depth.data
    out = []
    for i, j in pairs:
        a = np.interp(
            depth, ours.depth.data, ours[oname].isel(cast=i).data,
            left=np.nan, right=np.nan,
        )
        b = theirs[tname].isel(time=j).data
        both = np.isfinite(a) & np.isfinite(b)
        out.append(np.nanmedian(np.abs(a[both] - b[both])) if both.sum() >= 10 else np.nan)
    return np.array(out)


def _roughness_ratio(ours, theirs, pairs, oname, tname):
    """Matlab-to-python ratio of bin-scale structure in the matched profiles.

    Per profile, RMS of the second difference along depth over the depth
    range both products cover, a high-pass measure at the 0.5 m bin scale.
    A ratio near 1 says neither product is smoother than the other.
    """
    rough_o, rough_t = [], []
    for i, j in pairs:
        a = np.interp(
            theirs.depth.data, ours.depth.data, ours[oname].isel(cast=i).data,
            left=np.nan, right=np.nan,
        )
        b = theirs[tname].isel(time=j).data
        both = np.isfinite(a) & np.isfinite(b)
        if both.sum() < 200:
            continue
        k0, k1 = np.where(both)[0][[0, -1]]
        for arr, out in ((a[k0:k1], rough_o), (b[k0:k1], rough_t)):
            d = arr[2:] - 2 * arr[1:-1] + arr[:-2]
            d = d[np.isfinite(d)]
            out.append(np.sqrt(np.mean(d**2)) if d.size > 20 else np.nan)
    return float(np.nanmedian(rough_t) / np.nanmedian(rough_o))


@needs_data
def test_crossval_d11_casts_match(crossval):
    ours, theirs, pairs = crossval
    # 186 of 187 Matlab columns pair up. 15 of our 201 casts have no
    # counterpart: 12 in the 02:08:04 to 03:18:35 window whose raw files the
    # cruise chain never converted, plus three shallow bounces its cast
    # detector rejected.
    assert len(pairs) > 0.8 * ours.sizes["cast"]
    assert len(pairs) > 0.95 * theirs.sizes["time"]
    # Observed 2026-09-01: 201 python casts against 187 Matlab columns. The
    # range pins the cast detector against a future change that starts
    # splitting or merging profiles wholesale.
    assert 190 <= ours.sizes["cast"] <= 215
    gaps = np.array(
        [
            np.abs(ours.time.data[i] - theirs.time.data[j]) / np.timedelta64(1, "s")
            for i, j in pairs
        ]
    )
    # Profile mean times agree to a few seconds, so the pairing is not
    # leaning on the 5 min tolerance.
    assert np.median(gaps) < 5.0


@needs_data
def test_crossval_d11_temperature_grid(crossval):
    ours, theirs, pairs = crossval
    diffs = _profile_diffs(ours, theirs, pairs, "t", "t")
    # Measured 2026-09-01: median 0.00214 K, p90 0.00242 K, max 0.0179 K.
    # The floor is the Matlab low-pass, not a calibration or unit error;
    # the same comparison run with phase_match=True gives 0.00169 K.
    assert np.nanmedian(diffs) < 0.01
    assert np.nanpercentile(diffs, 90) < 0.01


@needs_data
def test_crossval_d11_conductivity_grid(crossval):
    ours, theirs, pairs = crossval
    diffs = _profile_diffs(ours, theirs, pairs, "c", "c")
    # Measured 2026-09-01: median 3.21e-4 S/m, p90 4.61e-4, max 4.58e-3.
    assert np.nanmedian(diffs) < 1e-3


@needs_data
def test_crossval_d11_salinity_grid(crossval):
    ours, theirs, pairs = crossval
    diffs = _profile_diffs(ours, theirs, pairs, "SP", "s")
    # Measured 2026-09-01: median 0.00191, p90 0.00310, max 0.0183.
    #
    # Neither of the two differences the design doc anticipated is
    # responsible. `gsw.SP_from_C` and `sw_salt` are both PSS-78 and agree
    # on the Matlab product's own binned c/t/p to 1.4e-14; per-sample
    # salinity and salinity from binned c/t/p differ by 3e-9 (median) in our
    # own product. What is left is the response matching the Matlab gridder
    # applies to conductivity, which removes salinity spiking that the
    # comparison run keeps.
    assert np.nanmedian(diffs) < 0.005

    # The 2024 product does not carry the temperature-scale bug the 2025 one
    # does. `salinity` reproduces from its own binned c/t/p as PSS-78 with
    # the ITS-90 conversion gsw applies, so a seawater 3.3 copy was first on
    # the 2024 MATLAB path. Measured 2026-09-01: max 1.4e-14 over 350,116
    # finite points, 350,112 of them bit-identical. The mirror of this
    # assertion in `test_crossval_d07_salinity_grid` fails without the
    # `/ T68_FACTOR`, which is what makes the two runs' opposite behavior a
    # tested fact instead of a note.
    recomputed = gsw.SP_from_C(theirs.c.data * 10.0, theirs.t.data, theirs.p.data)
    good = np.isfinite(recomputed) & np.isfinite(theirs.s.data)
    assert good.sum() > 100_000
    assert np.nanmax(np.abs(recomputed[good] - theirs.s.data[good])) < 1e-12


@needs_data
def test_crossval_d11_density_grid(crossval):
    ours, theirs, pairs = crossval
    # `load_fctd_grid` recomputes sgth with gsw from the Matlab binned
    # values, so this compares like with like. The Matlab `density` field
    # itself is EOS-80 `sw_pden` and sits 0.0051 kg/m^3 away from gsw
    # sigma0 on identical inputs.
    diffs = _profile_diffs(ours, theirs, pairs, "sgth0", "sgth")
    # Measured 2026-09-01: median 0.00181 kg/m^3, p90 0.00257, max 0.0141.
    assert np.nanmedian(diffs) < 0.005


@needs_data
def test_crossval_d11_depth_coordinate_convention(crossval):
    """The pressure offset is the Matlab gridder's hardcoded latitude.

    Binned pressure is the cleanest probe of the depth axis: within a bin it
    is fixed by depth alone, so a difference there is a difference in the
    depth conversion. Mapping the Matlab bin centers through
    `sw_dpth(., 20)` and back out through `gsw.z_from_p` at the deployment
    latitude should collapse the offset.
    """
    ours, theirs, pairs = crossval
    lat = float(ours.lat.mean())
    assert lat < 5.0, "deployment is near the equator; the LAT=20 story needs that"

    p_ref = np.arange(0.0, 3000.0, 0.05)
    p_of_zm = np.interp(theirs.depth.data, _sw_dpth(p_ref, MATLAB_GRID_LAT), p_ref)
    depth_equiv = -gsw.z_from_p(p_of_zm, lat)

    as_is = np.nanmedian(_profile_diffs(ours, theirs, pairs, "p", "p"))
    remapped = np.nanmedian(
        _profile_diffs(ours, theirs, pairs, "p", "p", depth=depth_equiv)
    )
    # Measured 2026-09-01: 0.308 dbar as-is, 0.0416 dbar remapped.
    assert as_is > 0.1
    assert remapped < 0.1
    assert remapped < as_is / 4.0

    # Temperature tightens on the same remap, for the same reason.
    t_as_is = np.nanmedian(_profile_diffs(ours, theirs, pairs, "t", "t"))
    t_remapped = np.nanmedian(
        _profile_diffs(ours, theirs, pairs, "t", "t", depth=depth_equiv)
    )
    # Measured 2026-09-01: 0.00214 K as-is, 0.00056 K remapped.
    assert t_remapped < t_as_is


# ---------------------------------------------------------------------------
# 2025 d07: the correction-free baseline
# ---------------------------------------------------------------------------


@needs_data_2025
def test_crossval_d07_casts_match(crossval2025):
    ours, theirs, pairs, n_missing = crossval2025
    # Every stem in fctd_mat/ resolved in Raw_full_cruise on 2026-09-01, so
    # the python and Matlab chains saw the same 690 files.
    assert n_missing == 0, f"{n_missing} fctd_mat stems missing from Raw_full_cruise"
    # Observed: 495 python casts, 494 Matlab columns, all 494 paired. The one
    # python cast without a counterpart is a 36 dbar upcast fragment at the
    # bottom of a turnaround that the Matlab detector keeps joined to the
    # ascent that follows it.
    assert len(pairs) == theirs.sizes["time"]
    assert len(pairs) > 0.98 * ours.sizes["cast"]
    gaps = np.array(
        [
            np.abs(ours.time.data[i] - theirs.time.data[j]) / np.timedelta64(1, "s")
            for i, j in pairs
        ]
    )
    assert np.median(gaps) < 5.0


@needs_data_2025
def test_crossval_d07_no_response_matching_in_baseline(crossval2025):
    """The 2025 baseline really is correction-free, checked in the data.

    `make_FCTDall_L1.m` carries `apply_response_matching_code = 0` and stamps
    a `response_match_applied` flag into the product. Reading the flag is one
    check; the other is that the two products have the same amount of
    structure at the bin scale, which is what fails on d11.
    """
    ours, theirs, pairs, _ = crossval2025
    flag = np.asarray(modfish.utils.loadmat(MATLAB_GRID_2025)["FCTDgrid"].
                      response_match_applied, dtype=float)
    finite = np.isfinite(flag)
    assert finite.any()
    assert np.all(flag[finite] == 0.0)

    ratio = _roughness_ratio(ours, theirs, pairs, "t", "t")
    # Measured 2026-09-01: ours 0.01094, Matlab 0.01096, ratio 1.002. On d11
    # the same statistic is 0.64.
    assert 0.95 < ratio < 1.05


@needs_data_2025
def test_crossval_d07_temperature_grid(crossval2025):
    ours, theirs, pairs, _ = crossval2025
    diffs = _profile_diffs(ours, theirs, pairs, "t", "t")
    # Measured 2026-09-01: median 7.82e-6 K, p90 7.84e-6, max 8.94e-6. Our L0
    # is bit-identical to the Matlab per-file .mat (checked on
    # FCTD25_12_05_080305: t, p, c all zero difference), so what is left is
    # bin membership under the two depth conventions.
    assert np.nanmedian(diffs) < 1e-4
    assert np.nanmax(diffs) < 1e-3


@needs_data_2025
def test_crossval_d07_conductivity_and_pressure_grid(crossval2025):
    ours, theirs, pairs, _ = crossval2025
    # Measured 2026-09-01: c median 1.91e-9 S/m, p median 7.47e-5 dbar.
    assert np.nanmedian(_profile_diffs(ours, theirs, pairs, "c", "c")) < 1e-6
    assert np.nanmedian(_profile_diffs(ours, theirs, pairs, "p", "p")) < 1e-3


@needs_data_2025
def test_crossval_d07_depth_convention_is_already_aligned(crossval2025):
    """No LAT=20 offset here: the 2025 chain uses the deployment mean latitude.

    Depth comes from `make_FCTDall_L0.m:87`, `sw_dpth(pressure, LAT)` with
    `LAT = nanmean(FCTDall.latitude)`, and `FastCTD_GridData` on `master`
    bins on that field instead of recomputing it. The remap that collapses
    the d11 pressure offset therefore has nothing to collapse.
    """
    ours, theirs, _pairs, _ = crossval2025
    lat = float(ours.lat.mean())
    p_ref = np.arange(0.0, 3000.0, 0.05)
    depth_equiv = -gsw.z_from_p(
        np.interp(theirs.depth.data, _sw_dpth(p_ref, lat), p_ref), lat
    )
    # Measured 2026-09-01: max 0.0008 m over the gridded depth range.
    assert np.nanmax(np.abs(depth_equiv - theirs.depth.data)) < 0.01


@needs_data_2025
def test_crossval_d07_salinity_grid(crossval2025):
    """Salinity, against the stored field and against a corrected recompute.

    `salinity_despike` is PSS-78 evaluated with the ITS-90 temperature passed
    through as if it were already IPTS-68, because
    `MOD_fish_lib/FastCTD_MATLAB/seawater/sw_sals.m` uses `del_T = T - 15`
    and was first on the 2025 MATLAB path. Recomputing the same quantity
    from the Matlab binned c/t/p with the conversion PSS-78 requires brings
    agreement down to the temperature comparison's own floor, which is what
    pins the explanation.
    """
    ours, theirs, pairs, _ = crossval2025
    stored = _profile_diffs(ours, theirs, pairs, "SP", "sd")
    corrected = _profile_diffs(ours, theirs, pairs, "SP", "sp_its90")
    # Measured 2026-09-01: 0.00160 against the stored field, 7.89e-6 against
    # the recompute; the temperature comparison's floor is 7.82e-6.
    assert np.nanmedian(stored) < 0.005
    assert np.nanmedian(corrected) < 1e-4
    assert np.nanmedian(corrected) < np.nanmedian(stored) / 100.0

    # The stored field reproduces exactly when the conversion is undone, so
    # the offset is the temperature scale and nothing else.
    undone = gsw.SP_from_C(
        theirs.c.data * 10.0, theirs.t.data / T68_FACTOR, theirs.p.data
    )
    good = np.isfinite(undone) & np.isfinite(theirs.sd.data)
    assert np.nanmax(np.abs(undone[good] - theirs.sd.data[good])) < 1e-12
