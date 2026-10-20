"""
Readers for the .mat files the MOD Matlab processing writes for FastCTD and
Epsifish: Epsifish gridded profiles, FastCTD per-file raw time series and
FastCTD grids.
"""

import re
from pathlib import Path

import gsw
import numpy as np
import xarray as xr

import modfish


def _add_teos(ds):
    """Add TEOS-10 `SA`, `CT` and `sgth` (sigma_0) from `s`, `t`, `p`, `lon`, `lat`."""
    ds["SA"] = gsw.SA_from_SP(ds.s, ds.p, ds.lon, ds.lat)
    ds.SA.attrs = dict(long_name="absolute salinity", units="g/kg")
    ds["CT"] = gsw.CT_from_t(ds.SA, ds.t, ds.p)
    ds.CT.attrs = dict(long_name="conservative temperature", units="°C")
    ds["sgth"] = gsw.density.sigma0(ds.SA, ds.CT)
    ds.sgth.attrs = dict(long_name=r"$\sigma_0$", units="kg/m$^3$")
    return ds


def _add_distance(ds):
    """Add along-track distance `dist` (km) on `time` from `lon`/`lat`.

    Missing positions are filled by linear interpolation in sample index
    first, so one NaN position does not turn every later distance NaN.
    All NaN when fewer than two positions are finite.
    """
    lon = np.asarray(ds.lon.values, dtype=float)
    lat = np.asarray(ds.lat.values, dtype=float)
    ok = np.isfinite(lon) & np.isfinite(lat)
    if ok.sum() < 2:
        dist = np.full(lon.size, np.nan)
    else:
        i = np.arange(lon.size)
        lon = np.interp(i, i[ok], lon[ok])
        lat = np.interp(i, i[ok], lat[ok])
        dist = np.insert(np.cumsum(gsw.distance(lon, lat) / 1e3), 0, 0.0)
    ds.coords["dist"] = (("time"), dist)
    ds.dist.attrs = dict(long_name="distance", units="km")
    return ds


def load_epsi_grid(file):
    """Load Epsifish gridded data file.

    This is usually the file `<survey>/profiles/griddedProfiles.mat`.

    Parameters
    ----------
    file : str or pathlib.Path
        The Matlab grid file.

    Returns
    -------
    xr.Dataset
        Dims `(depth, time)`. Data variables `t`, `th` (potential
        temperature), `s`, `w`, `chi1`, `chi2`, `eps1`, `eps2`, `eps`,
        `a1`-`a3` from the file, TEOS-10 `SA`, `CT`, `sgth`, and `n2`
        (`add_n2`). Coordinates `depth`, `p`, `time`, `lon`, `lat`,
        `profn` and the along-track `dist` (km).
    """
    grd = modfish.utils.loadmat(file)
    time = modfish.utils.mattime_to_datetime64(grd.dnum)
    ds = xr.Dataset(
        coords=dict(
            depth=(("depth"), grd.z),
            p=(("depth"), grd.pr),
            time=(("time"), time),
            lon=(("time"), grd.longitude),
            lat=(("time"), grd.latitude),
            profn=(("time"), grd.profNum),
        ),
        data_vars=dict(
            t=(("depth", "time"), grd.t),
            th=(("depth", "time"), grd.th),
            # sgth=(("depth", "time"), grd.sgth - sgth_subtract),
            w=(("depth", "time"), grd.w),
            s=(("depth", "time"), grd.s),
            chi1=(("depth", "time"), grd.chi1),
            chi2=(("depth", "time"), grd.chi2),
            eps1=(("depth", "time"), grd.epsilon_co1),
            eps2=(("depth", "time"), grd.epsilon_co2),
            eps=(("depth", "time"), grd.epsilon_final),
            a1=(("depth", "time"), grd.a1),
            a2=(("depth", "time"), grd.a2),
            a3=(("depth", "time"), grd.a3),
        ),
    )

    ds = _add_teos(ds)
    ds.p.data = np.float64(ds.p.data)
    ds = add_n2(ds, dp=10)
    ds = _add_distance(ds)
    ds.t.attrs = dict(long_name="temperature", units="°C")
    ds.th.attrs = dict(long_name=r"$\theta$", units="°C")
    ds.s.attrs = dict(long_name="salinity", units="psu")
    ds.eps1.attrs = dict(long_name=r"$\epsilon$", units="W/kg")
    ds.eps2.attrs = dict(long_name=r"$\epsilon$", units="W/kg")
    ds.eps.attrs = dict(long_name=r"$\epsilon$", units="W/kg")
    ds.chi1.attrs = dict(long_name=r"$\chi$", units=r"K$^2$/s")
    ds.chi2.attrs = dict(long_name=r"$\chi$", units=r"K$^2$/s")
    return ds


def load_fctd_raw_mat(file):
    """Read raw FCTD data at 16 Hz from a single file in the `fctd_mat` processing directory.

    Parameters
    ----------
    file : Path or str
        File path or name of one .mat file in the `fctd_mat` directory.

    Returns
    -------
    ds : xr.Dataset
        Data structure with raw time series data.
    mds : xr.Dataset
        Data structure with raw microconductivity time series data.
    """
    d = modfish.utils.loadmat(file)

    # fctd
    time = modfish.utils.mattime_to_datetime64(d.time)
    ds = xr.Dataset(
        coords=dict(
            time=(("time"), time),
        ),
        data_vars=dict(
            c=(("time"), d.conductivity),
            t=(("time"), d.temperature),
            p=(("time"), d.pressure),
            bb=(("time"), d.bb),
            chla=(("time"), d.chla),
            fDOM=(("time"), d.fDOM),
            lon=(("time"), d.longitude),
            lat=(("time"), d.latitude),
        ),
    )
    # GV 2025-11-30
    # dPdt, chi, chi2 and w are written by the older Matlab routines only.
    for var in ["dPdt", "chi", "chi2", "w"]:
        if var in d:
            ds[var] = (("time"), d[var])

    ds.p.attrs = dict(long_name="pressure", units="dbar")
    ds.c.attrs = dict(long_name="conductivity", units="S/m")
    ds.t.attrs = dict(long_name="temperature", units="°C")
    if "chi" in ds:
        ds.chi.attrs = dict(long_name=r"$\chi$", units="K$^2$/s")
    if "chi2" in ds:
        ds.chi2.attrs = dict(long_name=r"$\chi_2$", units="K$^2$/s")

    # generate depth variable
    mean_lat = np.nanmean(ds.lat.data)
    # we'll do the same as the Matlab routine and set latitude to 30 if nan
    if np.isnan(mean_lat):
        mean_lat = 30
    ds["depth"] = (("time"), -gsw.z_from_p(ds.p.data, mean_lat))

    # microconductivity
    # The newer Matlab routines write uConductivity/time_fast as stacked
    # matrices, the older ones ucon/ucon_corr/microtime as flat vectors.
    if "uConductivity" in d:
        ucon = d.uConductivity.reshape(-1)
    else:
        ucon = d.ucon

    if "time_fast" in d:
        microtime = modfish.utils.mattime_to_datetime64(d.time_fast.reshape(-1))
    elif "microtime" in d:
        microtime = modfish.utils.mattime_to_datetime64(d.microtime)
    else:
        # generate time vector if none exists (this happens when the Matlab
        # routines generate a microconductivity matrix with all NaNs).
        microtime = modfish.utils.datetime_linspace(time[0], time[-1], len(ucon))

    # generate output dataset for microconductivity
    mds = xr.Dataset(
        coords=dict(
            time=(("time"), microtime),
        ),
        data_vars=dict(
            ucon=(("time"), ucon),
        ),
    )
    if "ucon_corr" in d:
        mds["ucon_corr"] = (("time"), d.ucon_corr)

    return ds, mds


def fctd_mat_combine(files):
    """Read a number of files from fctd_mat directory and combine into one time series.

    Parameters
    ----------
    files : list
        List of files.

    Returns
    -------
    ds : xr.Dataset
        Data structure with raw time series data.
    mds : xr.Dataset
        Data structure with raw microconductivity time series data.
    """
    dsa = [load_fctd_raw_mat(file) for file in files]
    # extract ctd (ds) and microconductivity (mds)
    ds = xr.concat([dsi[0] for dsi in dsa], dim="time")
    mds = xr.concat([dsi[1] for dsi in dsa], dim="time")
    # add pressure to microconductivity
    p_interp = ds.p.interp_like(mds)
    mds["p"] = (("time"), p_interp.data)
    return ds, mds


def load_fctd_raw_time_series(fctd_mat_dir, start, end):
    """Combine data from a number of raw FCTD .mat files in the fctd_mat
    directory.

    Parameters
    ----------
    fctd_mat_dir : pathlib.Path
        `fctd_mat` directory.
    start : np.datetime64 or str
        Start time.
    end : np.datetime64 or str
        End time.

    Returns
    -------
    ds : xr.Dataset
        Raw FCTD time series, trimmed to `[start, end]`.
    mds : xr.Dataset
        Raw FCTD microconductivity time series, trimmed to `[start, end]`.

    Raises
    ------
    ValueError
        If no file starts before `end`.
    """
    all_files = _find_fctd_mat_files(fctd_mat_dir)
    file_times = np.array(
        [modfish.utils.parse_filename_datetime(file) for file in all_files]
    )
    start = np.datetime64(start)
    end = np.datetime64(end)
    # files are named by their start time: the file holding `start` is the
    # last one that starts at or before it
    first = max(int(np.searchsorted(file_times, start, side="right")) - 1, 0)
    last = int(np.searchsorted(file_times, end, side="left"))
    files = all_files[first:last]
    if not files:
        raise ValueError(f"no fctd_mat file in {fctd_mat_dir} covers {start} to {end}")
    ds, mds = fctd_mat_combine(files)
    return ds.sel(time=slice(start, end)), mds.sel(time=slice(start, end))


def _find_fctd_mat_files(data_directory):
    """Per-file data files in an fctd_mat directory.

    Some deployments name them `EPSI...`, others `FCTD...`. Products such
    as `FCTDgrid.mat` do not match the name pattern.

    Parameters
    ----------
    data_directory : str or pathlib.Path
        The `fctd_mat` directory.

    Returns
    -------
    list of pathlib.Path
        Files named `(EPSI|FCTD)YY_MM_DD_hhmmss.mat`, sorted by name,
        which is chronological.
    """
    pattern = re.compile(r"^(EPSI|FCTD)\d{2}_\d{2}_\d{2}_\d{6}\.mat$", re.IGNORECASE)
    return sorted(p for p in Path(data_directory).glob("*.mat") if pattern.match(p.name))


def load_fctd_grid(file, what="all"):
    """Load FastCTD gridded data file.

    This is usually the file `<survey>/fctd_mat/FCTDgrid.mat`.

    Parameters
    ----------
    file : str or pathlib.Path
        The Matlab grid file.
    what : {"all", "down", "dn", "up"}, optional
        Which grid to read: all casts (`FCTDgrid`, default), downcasts
        (`FCTDdown`) or upcasts (`FCTDup`).

    Returns
    -------
    xr.Dataset
        Dims `(depth, time)`. Data variables `t`, `c`, `s`, `density`, `p`
        and whichever of `drop`, `altDist`, `w`, `bb`, `chla`, `chi`, `chi2`
        the file has, TEOS-10 `SA`, `CT`, `sgth`, and `n2` (`add_n2`).
        Coordinates `depth`, `time`, per-profile mean `lon`/`lat`, the
        full `longitude_full`/`latitude_full`, and along-track `dist` (km).
        All-NaN depth rows and NaT times are dropped.

    Raises
    ------
    ValueError
        If `what` is not one of the values above.
    """
    keys = {"all": "FCTDgrid", "down": "FCTDdown", "dn": "FCTDdown", "up": "FCTDup"}
    if what not in keys:
        raise ValueError(f"what must be one of {sorted(keys)}, got {what!r}")
    grd = modfish.utils.loadmat(file)[keys[what]]
    time = modfish.utils.mattime_to_datetime64(grd.time)
    ds = xr.Dataset(
        coords=dict(
            depth=(("depth"), grd.depth),
            time=(("time"), time),
            lon=(("time"), np.nanmean(grd.longitude, axis=0)),
            lat=(("time"), np.nanmean(grd.latitude, axis=0)),
            longitude_full=(("depth", "time"), grd.longitude),
            latitude_full=(("depth", "time"), grd.latitude),
        ),
        data_vars=dict(
            t=(("depth", "time"), grd.temperature),
            c=(("depth", "time"), grd.conductivity),
            s=(("depth", "time"), grd.salinity),
            density=(("depth", "time"), grd.density),
            p=(("depth", "time"), grd.pressure),
        ),
    )
    for var in ["drop", "altDist", "w", "bb", "chla", "chi", "chi2"]:
        if var in grd.keys():
            ds[var] = (("depth", "time"), grd[var])

    ds = _add_teos(ds)
    ds = ds.dropna("depth", how="all")
    mask = ~np.isnat(ds.time)
    ds = ds.sel(time=mask)

    ds = add_n2(ds, dp=10)

    if "chi" in ds:
        ds.chi.attrs = dict(long_name=r"$\chi_1$", units="K$^2$/s")
    if "chi2" in ds:
        ds.chi2.attrs = dict(long_name=r"$\chi_2$", units="K$^2$/s")
    ds.t.attrs = dict(long_name="temperature", units="°C")
    ds.s.attrs = dict(long_name="salinity", units="psu")
    ds.depth.attrs = dict(long_name="depth", units="m")
    return _add_distance(ds)


def add_n2(ds, dp=10):
    """Add buoyancy frequency squared to a gridded dataset.

    Parameters
    ----------
    ds : xr.Dataset
        Gridded profiles with `s`, `t`, `p`, `lon` and `lat`. `t` and `s`
        span `depth` and `time` in either order; `p` may span both or
        `depth` only; `lon` and `lat` are per `time`.
    dp : float
        Pressure interval and smoothing length passed to
        `modfish.utils.nsqfcn` [dbar]. Defaults to 10.

    Returns
    -------
    xr.Dataset
        `ds` with `n2` [s$^{-2}$] on the dims of `t`, interpolated from
        the `dp` grid back onto each profile's pressure. NaN where the
        profile pressure lies outside the `dp` grid, and through a whole
        profile that `nsqfcn` returns NaN for (no usable samples, fewer
        samples than the filter needs, or non-monotonic low-passed
        pressure) or whose position is NaN.
    """
    n2 = np.full(ds.t.shape, np.nan)
    time_axis = ds.t.dims.index("time")
    for i in range(ds.sizes["time"]):
        dsi = ds.isel(time=i)
        p = np.broadcast_to(dsi.p.data, dsi.t.shape)
        n2i, midp = modfish.utils.nsqfcn(
            dsi.s.data, dsi.t.data, p, p0=0, dp=dp,
            lon=float(dsi.lon), lat=float(dsi.lat),
        )
        if np.ndim(n2i) == 0:
            continue
        np.moveaxis(n2, time_axis, 0)[i] = np.interp(
            p, midp, n2i, left=np.nan, right=np.nan
        )
    ds["n2"] = (ds.t.dims, n2)
    ds.n2.attrs = dict(
        long_name=r"N$^2$", units=r"s$^{-2}$", comment=f"smoothed over {dp} dbar"
    )
    return ds
