"""Cross-validation of the python pipeline against the cruise-era Matlab grid.

Runs the full chain (`modfish.modraw.convert` -> `process_deployment`) on the
2024 deployment `24_1120_d11_fctd_to_mooringA` and compares the gridded
product against `fctd_mat/FCTDgrid.mat`, which was written at sea on
2024-11-21. Skips unless the mod server is mounted.

Runtime: the module fixture converts 256 `.modraw` files (1.2 GB) and runs
the pipeline once for the whole module. Measured on samoan 2026-09-01:
579 s for the conversion, 22 s for `process_deployment`, so about 10 minutes
for the module. `slow` tests are not deselected by default in
`pyproject.toml`, so a plain `uv run pytest tests/` pays that cost.

Comparison contract (task brief): corrections off (`phase_match=False`,
`thermal_mass=False`), `latitude_fallback=30.0` (a no-op here, the
deployment has continuous GPS), `dz=0.5`, casts matched to Matlab grid
columns by profile mean time. Matching is mutual-nearest rather than
one-sided nearest: the python product has 201 casts against the Matlab
product's 187, and a one-sided match assigns four python casts to Matlab
columns that are already claimed.

Findings behind the tolerances are written up in `fctd_validation_notes.md`
at the repo root. The three that shape the assertions here:

- The Matlab gridder hardcodes `sw_dpth(pressure, 20)`
  (`FastCTD_GridData.m:286` at `MOD_fish_lib` `0d0d5e4`, the cruise-era
  commit), while this deployment sits at 0.73 degN and the pipeline uses
  `gsw.z_from_p` with per-sample latitude. The two depth axes drift apart
  by 0.65 m at 1046 m, which is what
  `test_crossval_d11_depth_coordinate_convention` measures and removes.
- The cruise-era gridder applies San's legacy frequency-domain response
  matching to every profile (`use_old_code = 1`, `FastCTD_GridData.m:245`):
  a gain and phase correction on conductivity plus a
  `(cos(pi f / 2 f_Ny))**30` low pass on t, c, and p with `f_Ny = 8` Hz. The
  comparison run has no T-C correction, so the Matlab profiles are the
  smoother of the two and the residuals are set by that, not by the
  salinity or density formulation.
- 17 of the 256 raw files were never converted by the cruise chain (239
  `EPSI*.mat` files in `fctd_mat/`), 16 of them inside 02:08-03:22 on
  2024-11-21, so 13 full-depth python casts have no Matlab counterpart.
"""

import pathlib
import shutil

import gsw
import numpy as np
import pytest
import xarray as xr

import modfish
from modfish.fctd import process_deployment
from modfish.fctd.config import FCTDConfig, TCParams

DEPLOY = pathlib.Path(
    "/mnt/mod-server/MOTIVE/Cruises/skq202417s/05_processed_data/24_1120_d11_fctd_to_mooringA"
)
MATLAB_GRID = DEPLOY / "fctd_mat" / "FCTDgrid.mat"

pytestmark = pytest.mark.slow
needs_data = pytest.mark.skipif(not MATLAB_GRID.exists(), reason="mod server not mounted")

#: Latitude the Matlab gridder hardcodes into its depth conversion
#: (`FastCTD_GridData.m:286`, `MOD_fish_lib` `0d0d5e4`).
MATLAB_GRID_LAT = 20.0

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
        latitude_fallback=30.0,
    )
    _, grid_path = process_deployment(l0, tmp_path / "out", "d11", cfg)

    with xr.open_dataset(grid_path) as ds:
        ours = ds.load()
    shutil.rmtree(tmp_path, ignore_errors=True)

    theirs = modfish.io.load_fctd_grid(MATLAB_GRID)

    dt = np.abs(ours.time.data[:, None] - theirs.time.data[None, :]) / np.timedelta64(
        1, "s"
    )
    nearest_j = dt.argmin(axis=1)
    nearest_i = dt.argmin(axis=0)
    pairs = [
        (i, int(nearest_j[i]))
        for i in range(dt.shape[0])
        if nearest_i[nearest_j[i]] == i and dt[i, nearest_j[i]] < MATCH_TOL_S
    ]
    return ours, theirs, pairs


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


@needs_data
def test_crossval_d11_casts_match(crossval):
    ours, theirs, pairs = crossval
    # 186 of 187 Matlab columns pair up; 15 of our 201 casts have no
    # counterpart (13 in the window the cruise chain never converted, plus
    # three shallow bounces its cast detector rejected).
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
