"""Bias-budget regression of the chi chain against the shipboard 2025 d07
column (spec "Testing", regression). Slow; skips without the mod server.

The port is run as close to the v3 estimator as its one chain allows:
gain 8.8 (the shipboard constant 22 is S/m per normalized ADC fraction;
modfish c1 is volts at 2.5 V per fraction, so 22 / 2.5), AP00 sinc^2, caps
only, closure off. Known residuals the shipboard column carries and the
port does not: the first-difference attenuation (4.4 % at 12.5 cpm at
3 m/s), the 320 Hz resample (1.7 % on the axis, about 3 % of power at
12.5 cpm), the gsw-versus-SBE dTdC linearization (0.5 to 4.2 %, growing
with depth), and the whole-range circular FFT at range edges (excluded
here). Expected offset of the port above the column is therefore a few
percent, well inside the 0.1 log10 tolerance. The coverage threshold below
is 1000, not 2000: the chi window's 0.25 s step at this deployment's 3.6
to 3.8 m/s fall rate advances about 0.9 m between estimates, so a ~347 s
cast populates at most about 1400 of the 0.5 m bins.
"""

import pathlib

import numpy as np
import pytest
import scipy.io
import xarray as xr

import modfish
from modfish.chi import ChiParams, add_chi
from modfish.chi.load import load_c1, range_time
from modfish.fctd import FCTDConfig, concat_l0, make_l1
from modfish.utils import mattime_to_datetime64, parse_filename_datetime

DEPLOY = pathlib.Path(
    "/mnt/mod-server/MOTIVE/Cruises/skq202521s/05_processed_data/25_1205_d07_FCTD1_FrontStation"
)
MATLAB_GRID = DEPLOY / "fctd_mat" / "FCTDgrid.mat"
RAW = DEPLOY / "raw"

pytestmark = pytest.mark.slow
needs_data = pytest.mark.skipif(
    not (MATLAB_GRID.exists() and RAW.is_dir()), reason="mod server not mounted"
)

#: The three review casts (note section 4), by L1 start and end time.
CASTS = {
    50: ("2025-12-05T13:49:32", "2025-12-05T13:55:18"),
    248: ("2025-12-06T11:28:30", "2025-12-06T11:34:28"),
    444: ("2025-12-07T08:41:59", "2025-12-07T08:47:56"),
}
EDGES = np.arange(-0.25, 2000.26, 0.5)
EDGE_S = 2.31  # s, range-edge exclusion
TOL = 0.1  # log10
LOOKBACK = np.timedelta64(20, "m")
PAD = np.timedelta64(60, "s")


@pytest.fixture(scope="module")
def matlab():
    raw = scipy.io.loadmat(MATLAB_GRID, variable_names=["FCTDgrid"],
                           struct_as_record=False, squeeze_me=True)["FCTDgrid"]
    ds = xr.Dataset(dict(chi=(("depth", "time"), np.asarray(raw.chi))),
                    coords=dict(depth=np.asarray(raw.depth), time=mattime_to_datetime64(raw.time)))
    return ds.sel(time=~np.isnat(ds.time))


def _bin_mean(depth, values, edges):
    idx = np.digitize(depth, edges) - 1
    out = np.full(edges.size - 1, np.nan)
    for i in range(edges.size - 1):
        sel = (idx == i) & np.isfinite(values)
        if sel.any():
            out[i] = values[sel].mean()
    return out


def _run_cast(tmp_path, start, end):
    start, end = np.datetime64(start), np.datetime64(end)
    raw = [f for f in sorted(RAW.glob("*.modraw"))
           if start - LOOKBACK <= parse_filename_datetime(f) <= end + PAD]
    assert raw, "no raw files around the cast"
    l0_dir = tmp_path / f"l0_{str(start)[:19].replace(':', '')}"
    modfish.modraw.convert(raw, l0_dir)
    files = sorted(l0_dir.glob("*.nc"))
    l1 = make_l1(concat_l0(files, groups=("ctd", "gps")), FCTDConfig())
    params = ChiParams(enabled=True, gain=8.8, antialias="ap00_sinc2", snr=0.0, closure=False)
    chi = add_chi(l1, files, params)["chi"].to_dataset()
    c1, ranges = load_c1(files)
    edge = np.zeros(chi.sizes["time"], dtype=bool)
    t = chi["time"].values
    for _, r in ranges.iterrows():
        if not np.isfinite(r.fs):
            continue
        tr = range_time(r.start, r.n, r.fs)
        near = (np.abs((t - tr[0]).astype("int64")) < EDGE_S * 1e9) | (
            np.abs((t - tr[-1]).astype("int64")) < EDGE_S * 1e9)
        edge |= near
    inside = (t >= start) & (t <= end) & ~edge
    ours = _bin_mean(chi["depth"].values[inside], chi["chi"].values[inside], EDGES)
    mean_time = start + (end - start) / 2
    return ours, mean_time


@needs_data
@pytest.mark.parametrize("cast", sorted(CASTS))
def test_d07_cast_within_bias_budget(tmp_path, matlab, cast):
    ours, mean_time = _run_cast(tmp_path, *CASTS[cast])
    j = int(np.argmin(np.abs((matlab.time.values - mean_time).astype("int64"))))
    assert abs((matlab.time.values[j] - mean_time).astype("int64")) < 300e9
    theirs = np.interp(EDGES[:-1] + 0.25, matlab.depth.values, matlab.chi.values[:, j],
                       left=np.nan, right=np.nan)
    both = np.isfinite(ours) & np.isfinite(theirs) & (ours > 0) & (theirs > 0)
    assert both.sum() > 1000
    dlog = np.log10(ours[both]) - np.log10(theirs[both])
    median = float(np.median(dlog))
    mad = float(np.median(np.abs(dlog - median)))
    print(f"cast {cast}: n={both.sum()} median dlog10={median:+.4f} MAD={mad:.4f}")
    assert abs(median) < TOL
    assert mad < TOL
