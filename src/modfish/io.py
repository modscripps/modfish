#!/usr/bin/env python
# coding: utf-8
"""
I/O functions mostly for reading .mat files produced from FastCTD and Epsifish observations.
"""

import re
from pathlib import Path
import gsw
import numpy as np
import scipy
import xarray as xr

import modfish


def load_epsi_profile(name):
    """Load one epsi profile.

    Parameters
    ----------
    name : Path or str
        File path or name.

    Returns
    -------
    ds : xr.Dataset
        Data structure with basic variables.

    Notes
    -----
    Note that there are also `load_epsi_raw` and `load_epsi_ctd_raw` for
    reading raw time series.
    """
    epsi = scipy.io.loadmat(name)
    epsi = epsi["Profile"][0, 0]

    depth = epsi["z"].squeeze()
    dnum = epsi["dnum"].squeeze()
    eps_co1 = epsi["epsilon_co"].squeeze()[:, 0]
    eps_co2 = epsi["epsilon_co"].squeeze()[:, 1]
    chi1 = epsi["chi"].squeeze()[:, 0]
    chi2 = epsi["chi"].squeeze()[:, 1]

    prof = xr.Dataset(
        data_vars=dict(
            time=(("depth"), modfish.utils.mattime_to_datetime64(dnum)),
            eps_co1=(("depth"), eps_co1),
            eps_co2=(("depth"), eps_co2),
            chi1=(("depth"), chi1),
            chi2=(("depth"), chi2),
        ),
        coords=dict(depth=(("depth"), depth)),
    )
    for v in [
        "w",
        "t",
        "s",
        "th",
        "pr",
        "sgth",
        "pitch",
        "roll",
        "kvis",
        "epsilon_final",
    ]:
        prof[v] = (("depth"), epsi[v].squeeze())

    prof = prof.rename(dict(roll="rol", epsilon_final="eps_final"))

    start_dnum = dnum[~np.isnan(dnum)]
    start_time = modfish.utils.mattime_to_datetime64(np.nanmean(start_dnum[:10]))
    prof.attrs["start_time"] = modfish.utils.datetime64_to_str(start_time, unit="m")
    prof.attrs["lon"] = epsi["longitude"].squeeze()
    prof.attrs["lat"] = epsi["latitude"].squeeze()
    prof.attrs["profile number"] = epsi["profNum"].squeeze()
    # prof = prof.where(~np.isnan(prof.depth), drop=True)
    # new_depth= np.arange(0, 1000.5, 0.5)
    # profi = prof.interp(depth=new_depth)
    return prof


def load_epsi_raw(name):
    """Load raw time series for one epsi profile.

    Parameters
    ----------
    name : Path or str
        File path or name.

    Returns
    -------
    ds : xr.Dataset
        Data structure with basic variables.
    """
    epsi = scipy.io.loadmat(name)
    epsi = epsi["Profile"][0, 0]
    epsi_raw = epsi["epsi"][0][0]
    dnum = epsi_raw["dnum"].squeeze()
    raw = xr.Dataset(
        data_vars=dict(time_s=epsi_raw["time_s"].squeeze()),
        coords=dict(
            time=(("time"), modfish.utils.mattime_to_datetime64(dnum)),
        ),
    )
    for k in epsi_raw.dtype.names:
        raw[k] = (("time"), epsi_raw[k].squeeze())
    return raw


def load_epsi_raw_mat(file):
    d = scipy.io.loadmat(file)
    epsi = d["epsi"][0, 0]
    dnum = epsi["dnum"].squeeze()
    time = modfish.utils.mattime_to_datetime64(dnum)
    ds = xr.Dataset(
        coords=dict(
            time=(("time"), time),
        ),
        data_vars=dict(
            s1_volt=(("time"), epsi["s1_volt"].squeeze()),
            s2_volt=(("time"), epsi["s2_volt"].squeeze()),
        ),
    )

    ctd = d["ctd"][0, 0]
    dnum = ctd["dnum"].squeeze()
    time = modfish.utils.mattime_to_datetime64(dnum)
    cds = xr.Dataset(
        coords=dict(
            time=(("time"), time),
        ),
        data_vars=dict(
            t=(("time"), ctd["T"].squeeze()),
            s=(("time"), ctd["S"].squeeze()),
            c=(("time"), ctd["C"].squeeze()),
            p=(("time"), ctd["P"].squeeze()),
            dzdt=(("time"), ctd["dzdt"].squeeze()),
        ),
    )
    # time = modfish.utils.mattime_to_datetime64(d.ctd.dnum)
    # ctd = xr.Dataset(
    #     coords=dict(
    #         time=(("time"), time),
    #     ),
    #     data_vars=dict(
    #         p=(("time"), d.ctd.P),
    #         t=(("time"), d.ctd.T),
    #         c=(("time"), d.ctd.C),
    #         s=(("time"), d.ctd.S),
    #     ),
    # )
    ds["p"] = (("time"), cds.p.interp_like(ds).data)
    ds["t"] = (("time"), cds.t.interp_like(ds).data)
    ds["c"] = (("time"), cds.c.interp_like(ds).data)
    ds["s"] = (("time"), cds.s.interp_like(ds).data)
    ds["dzdt"] = (("time"), cds.dzdt.interp_like(ds).data)
    return ds


def load_epsi_ctd_raw(name):
    """Load raw CTD time series for one epsi profile.

    Parameters
    ----------
    name : Path or str
        File path or name.

    Returns
    -------
    ds : xr.Dataset
        Data structure with basic variables.
    """
    epsi = scipy.io.loadmat(name)
    epsi = epsi["Profile"][0, 0]
    epsi_raw = epsi["epsi"][0][0]
    dnum = epsi_raw["dnum"].squeeze()
    raw = xr.Dataset(
        data_vars=dict(time_s=epsi_raw["time_s"].squeeze()),
        coords=dict(
            time=(("time"), modfish.utils.mattime_to_datetime64(dnum)),
        ),
    )
    for k in epsi_raw.dtype.names:
        raw[k] = (("time"), epsi_raw[k].squeeze())
    return raw
    epsi = scipy.io.loadmat(name)
    epsi = epsi["Profile"][0, 0]
    ctd_raw = epsi["ctd"][0][0]
    dnum = ctd_raw["dnum"].squeeze()
    raw = xr.Dataset(
        data_vars=dict(time_s=ctd_raw["time_s"].squeeze()),
        coords=dict(
            time=(("time"), modfish.utils.mattime_to_datetime64(dnum)),
        ),
    )
    for k in ctd_raw.dtype.names:
        raw[k] = (("time"), ctd_raw[k].squeeze())
    return raw


def load_epsi_grid(file):
    """Load Epsifish gridded data file.

    This is usually the file `<survey>/profiles/griddedProfiles.mat`.

    Parameters
    ----------
    file : str or pathlib.Path

    Returns
    -------
    xr.Dataset
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

    ds["SA"] = gsw.SA_from_SP(ds.s, ds.p, ds.lon, ds.lat)
    ds.SA.attrs = dict(long_name="absolute salinity", units="kg/m$^3$")
    ds["CT"] = gsw.CT_from_t(ds.SA, ds.t, ds.p)
    ds.CT.attrs = dict(long_name="conservative temperature", units="°C")
    ds["sgth"] = gsw.density.sigma0(ds.SA, ds.CT)
    ds.sgth.attrs = dict(long_name=r"$\sigma_0$", units="kg/m$^3$")

    ds.p.data = np.float64(ds.p.data)

    ds = add_n2(ds, dp=10)

    dist = gsw.distance(ds.lon, ds.lat)
    cdist = np.cumsum(dist)
    cdist = np.insert(cdist, 0, 0)
    ds.coords["dist"] = (("time"), cdist / 1e3)
    ds.dist.attrs = dict(long_name="distance", units="km")
    ds.t.attrs = dict(long_name="temperature", units="°C")
    ds.th.attrs = dict(long_name=r"$\Theta$", units="°C")
    ds.s.attrs = dict(long_name="salinity", units="psu")
    ds.sgth.attrs = dict(long_name=r"$\sigma_\theta$", units=r"kg/m$^3$")
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
    ds.c.attrs = dict(long_name="conductivity", units="mS/cm")
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
        Raw FCTD time series.
    mds : xr.Dataset
        Raw FCTD microconductivity time series.
    """
    # We need a workaround for some of the files starting with EPSI, others
    # with FCTD.

    # all_files = sorted(fctd_mat_dir.glob("EPSI*.mat"))
    all_files = _find_fctd_mat_files(fctd_mat_dir)
    file_times = np.array(
        [modfish.utils.parse_filename_datetime(file) for file in all_files]
    )
    if type(start) is str:
        start = np.datetime64(start)
    if type(end) is str:
        end = np.datetime64(end)
    ind = np.flatnonzero((file_times > start) & (file_times < end))
    files = [all_files[i] for i in ind]
    return fctd_mat_combine(files)


def _find_fctd_mat_files(data_directory):
    """Locate data files in an fctd_mat directory.

    Parameters
    ----------
    data_directory :

    Returns
    -------

    """
    # Convert directory from str to pathlib.Path if necessary
    data_directory = modfish.utils.process_file_path(data_directory)
    # Define the flexible glob pattern.
    # '????' matches the 4-character prefix (EPSI/FCTD)
    # '*' matches the entire variable date/time part (YY_MM_DD_hhmmss)
    search_pattern = '????*.mat'

    # Define specific filenames to exclude
    excluded_files = {
        'FCTDgrid.mat',
        'FCTDall_L0.mat',
        'FCTDall_L1.mat',
        'FastCTD_MATfile_TimeIndex.mat'
    }

    # Define a regex to validate the *start* of the filename after the glob is applied.
    # This ensures only 'EPSI' or 'FCTD' files are included,
    # preventing other 4-letter prefixes like 'TEST' from sneaking in.
    # ^(EPSI|FCTD): Starts with 'EPSI' OR 'FCTD'
    # \d{2}_\d{2}_\d{2}: Followed by YY_MM_DD
    # _: Followed by an underscore
    # \d{6}: Followed by hhmmss
    # \.mat$: Ends with .mat
    # The full pattern is a good way to be specific after the broad glob.
    validation_regex = re.compile(r"^(EPSI|FCTD)\d{2}_\d{2}_\d{2}_\d{6}\.mat$", re.IGNORECASE)

    # Path.glob() to find and filter all matching files
    matching_files = [
        # Get the string representation of the path object
        p
        for p in data_directory.glob(search_pattern)
        if (p.name not in excluded_files and validation_regex.match(p.name))
    ]

    return sorted(matching_files)


def load_fctd_grid(file, what="all"):
    """Load FastCTD gridded data file.

    This is usually the file `<survey>/fctd_mat/FCTDgrid.mat`.

    Parameters
    ----------
    file : str or pathlib.Path

    Returns
    -------
    xr.Dataset
    """
    tmp = modfish.utils.loadmat(file)
    match what:
        case "down":
            grd = tmp["FCTDdown"]
        case "dn":
            grd = tmp["FCTDdown"]
        case "up":
            grd = tmp["FCTDup"]
        case _:
            grd = tmp["FCTDgrid"]
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

    ds["SA"] = gsw.SA_from_SP(ds.s, ds.p, ds.lon, ds.lat)
    ds.SA.attrs = dict(long_name="absolute salinity", units="kg/m$^3$")
    ds["CT"] = gsw.CT_from_t(ds.SA, ds.t, ds.p)
    ds.CT.attrs = dict(long_name="conservative temperature", units="°C")
    ds["sgth"] = gsw.density.sigma0(ds.SA, ds.CT)
    ds.sgth.attrs = dict(long_name=r"$\sigma_0$", units="kg/m$^3$")

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
    dist = gsw.distance(ds.lon.data, ds.lat.data) / 1e3
    dist = np.insert(np.cumsum(dist), 0, 0)
    ds.coords["dist"] = (("time"), dist)
    ds.dist.attrs = dict(long_name="distance", units="km")
    return ds


def plot_epsi_profile(prof):
    start_str = prof.start_time.replace("T", " ")
    opts = dict(linewidth=1)
    fig, ax = gv.plot.quickfig(c=6, sharey=True, grid=True, fgs=(12, 5))
    ax[0].plot(prof.eps_co1, prof.depth, color="C0", **opts)
    ax[0].plot(prof.eps_co2, prof.depth, color="C6", **opts)
    ax[0].set(
        xscale="log",
        xlim=[1e-11, 1e-6],
        ylabel="depth [m]",
        xlabel="$\\mathrm{log}_{10}(\\epsilon)$ [W/kg]",
        title=f"profile {prof.attrs['profile number']:03d}",
    )
    ax[1].plot(prof.chi1, prof.depth, color="C0", **opts)
    ax[1].plot(prof.chi2, prof.depth, color="C6", **opts)
    ax[1].set(
        xscale="log",
        xlabel="$\\mathrm{log}_{10}(\\chi)$ [K$^2$/s]",
        title=start_str,
    )
    ax[0].invert_yaxis()
    ax[2].plot(prof.w, prof.depth, **opts)
    ax[2].set(xlabel="fall rate [m/s]")
    ax[3].plot(prof.pitch, prof.depth, color="C0", **opts)
    ax[3].plot(prof.rol, prof.depth, color="C6", **opts)
    gv.plot.xsym(ax[3])
    ax[3].set(xlabel="pitch/roll [deg]")
    ax[4].plot(prof.t, prof.depth, **opts)
    ax[4].set(xlabel="temperature [°C]")
    ax[5].plot(prof.s, prof.depth, **opts)
    ax[5].set(xlabel="salinity")
    return ax


def add_n2(ds, dp=10):
    # add N^2 calculation that does not depend on gvpy
    return ds
    # # calculate buoyancy frequency
    # ds["n2"] = ds.t.copy() * np.nan
    # ds.n2.attrs = dict(
    #     long_name=r"N$^2$", units=r"s$^{-2}$", info=f"smoothed over {dp} dbar"
    # )
    # for i in range(len(ds.time)):
    #     dsi = ds.isel(time=i)
    #     # dsi = dsi.dropna("depth", how="any", subset=["t", "s", "p"])
    #     try:
    #         n2, midp = gv.ocean.nsqfcn(
    #             dsi.s.data,
    #             dsi.t.data,
    #             dsi.p.data,
    #             p0=0,
    #             dp=dp,
    #             lon=dsi.lon.data,
    #             lat=dsi.lat.data,
    #         )
    #         n2i = scipy.interpolate.interp1d(midp, n2, bounds_error=False)(dsi.p.data)
    #         shape = ds.t.shape
    #         if len(ds.time) == shape[0]:
    #             ds.n2.data[i, :] = n2i
    #         elif len(ds.time) == shape[1]:
    #             ds.n2.data[:, i] = n2i
    #     except:
    #         pass
    # return ds
