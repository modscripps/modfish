#!/usr/bin/env python
# coding: utf-8
"""Utilities"""

import pandas as pd


def mattime_to_datetime64(dnum):
    """Convert Matlab datenum time format to numpy's datetime64 format.

    Parameters
    ----------
    dtnum : array-like
        Time in Matlab datenum format.

    Returns
    -------
    time : np.datetime64
        Time in numpy datetime64 format

    Notes
    -----
    In Matlab, datevec(719529) = [1970 1 1 0 0 0]
    """
    t = pd.to_datetime(dnum - 719529, unit="D")
    if isinstance(t, pd.Timestamp):
        time = t.to_datetime64()
    elif isinstance(t, pd.DatetimeIndex):
        time = t.values
    return time


def datetime64_to_str(dt64, unit="D"):
    """Convert numpy datetime64 object or array to str or array of str.

    Parameters
    ----------
    dt64 : np.datetime64 or array-like
        Time in numpy datetime64 format
    unit : str, optional
        Date unit. Defaults to "D".

    Returns
    -------
    str or array of str

    Notes
    -----
    Valid date unit formats are listed at
    https://numpy.org/doc/stable/reference/arrays.datetime.html#arrays-dtypes-dateunits

    """

    return np.datetime_as_string(dt64, unit=unit).replace("T", " ")


def loadmat(filename, onevar=False, verbose=False):
    """
    Load Matlab .mat files and return as dictionary.

    Parameters
    ----------
    filename : str
        Path to .mat file

    Returns
    -------
    out : dict
        Data in a dictionary.
    """

    def _check_keys(dict):
        """
        checks if entries in dictionary are mat-objects. If yes
        todict is called to change them to nested dictionaries
        """
        for key in dict:
            ni = np.size(dict[key])
            if ni <= 1:
                if isinstance(dict[key], spio.matlab.mat_struct):
                    dict[key] = _todict(dict[key])
            else:
                for i in range(0, ni):
                    if isinstance(dict[key][i], spio.matlab.mat_struct):
                        dict[key][i] = _todict(dict[key][i])
        return dict

    def _todict(matobj):
        """
        A recursive function which constructs from matobjects nested dictionaries
        """
        dict = {}
        for strg in matobj._fieldnames:
            elem = matobj.__dict__[strg]
            if isinstance(elem, spio.matlab.mat_struct):
                dict[strg] = _todict(elem)
            else:
                dict[strg] = elem
        return dict

    data = spio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    out = _check_keys(data)

    dk = list(out.keys())
    actual_keys = [k for k in dk if k[:2] != "__"]
    if len(actual_keys) == 1:
        if verbose:
            print("found only one variable, returning munchified data structure")
        return out[actual_keys[0]]
    else:
        out2 = {}
        for k in actual_keys:
            out2[k] = out[k]
        return out2


def parse_filename_datetime(file):
    yy, mm, dd, time = file.stem.split("I")[1].split("_")
    dtstr = f"20{yy}-{mm}-{dd} {time[:2]}:{time[2:4]}:{time[4:6]}"
    return np.datetime64(dtstr)


# -----------------------------------------
# add T-C correction functions from ctdproc
# -----------------------------------------


def add_tcfit_default(ds):
    """
    Get default values for tc fit range depending on depth of cast.

    Range for tc fit is 200dbar to maximum pressure if the cast is
    shallower than 1000dbar, 500dbar to max pressure otherwise.

    Parameters
    ----------
    ds : xarray.Dataset
            CTD time series data structure

    Returns
    -------
    tcfit : tuple
            Upper and lower limit for tc fit in phase_correct.
    """
    if ds.p.max() > 1000:
        tcfit = [500, ds.p.max().data]
    elif ds.p.max() > 300:
        tcfit = [200, ds.p.max().data]
    else:
        tcfit = [50, ds.p.max().data]
    ds.attrs["tcfit"] = tcfit
    return ds


def atanfit(x, f, Phi, W):
    f = np.arctan(2 * np.pi * f * x[0]) + 2 * np.pi * f * x[1] + Phi
    f = np.matmul(np.matmul(f.transpose(), W**4), f)
    return f


def phase_correct(ds, N=2**6, plot_spectra=False):
    """
    Bring temperature and conductivity in phase.

    Parameters
    ----------
    ds : dtype
            description
    N : int
        Number of points per fit segment

    Returns
    -------
    ds : dtype
            description
    """

    # remove spikes
    # TODO: bring this back in. however, the function fails later on if there
    # are nan's present. Could interpolate over anything that is just a few data points
    # for field in ["t1", "t2", "c1", "c2"]:
    # 	  ib = np.squeeze(np.where(np.absolute(np.diff(data[field].data)) > 0.5))
    # 	  data[field][ib] = np.nan

    # ---Spectral Analysis of Raw Data---
    # 16Hz data from SBE49 (note the difference to 24Hz on the SBE9/11 system)
    dt = 1 / 16
    # number of points per segment
    # N = 2**9 (setting for 24 Hz)
    # N = 2**6 (now providing as function argument)

    # only data within tcfit range.
    ii = np.squeeze(
        np.argwhere(
            (ds.p.data > ds.attrs["tcfit"][0]) & (ds.p.data < ds.attrs["tcfit"][1])
        )
    )
    i1 = ii[0]
    i2 = ii[-1]
    n = i2 - i1 + 1
    n = (np.floor(n / N) * N).astype("int64")
    # Truncate to be multiple of N elements long
    i2 = (i1 + n).astype("int64")
    # number of segments = dof/2
    m = (n / N).astype("int64")
    print(f"{m} segments")
    # Number of degrees of freedom (power of 2)
    dof = 2 * m
    # Frequency resolution at dof degrees of freedom.
    df = 1 / (N * dt)

    # fft of each segment (row). Data are detrended, then windowed.
    window = signal.windows.triang(N) * np.ones((m, N))
    At1 = fft.fft(
        signal.detrend(np.reshape(ds.t.data[i1:i2], newshape=(m, N))) * window
    )
    Ac1 = fft.fft(
        signal.detrend(np.reshape(ds.c.data[i1:i2], newshape=(m, N))) * window
    )

    # Positive frequencies only
    At1 = At1[:, 0 : int(N / 2)]
    Ac1 = Ac1[:, 0 : int(N / 2)]

    # Frequency
    f = fft.ifftshift(np.linspace(-N / 2, N / 2 - 1, N) / N / dt)
    f = f[: int(N / 2)]
    fold = f

    # Spectral Estimates. Note: In Matlab, At1*conj(At1) is not complex anymore.
    # Here, it is still a complex number but the imaginary part is zero.
    # We keep only the real part to stay consistent.
    Et1 = 2 * np.real(np.nanmean(At1 * np.conj(At1) / df / N**2, axis=0))
    Ec1 = 2 * np.real(np.nanmean(Ac1 * np.conj(Ac1) / df / N**2, axis=0))

    # Cross Spectral Estimates
    Ct1c1 = 2 * np.nanmean(At1 * np.conj(Ac1) / df / N**2, axis=0)

    # Squared Coherence Estimates
    Coht1c1 = np.real(Ct1c1 * np.conj(Ct1c1) / (Et1 * Ec1))

    # Cross-spectral Phase Estimates
    Phit1c1 = np.arctan2(np.imag(Ct1c1), np.real(Ct1c1))

    # ---Determine tau and L---
    # tau is the thermistor time constant (sec)
    # L is the lag of t behind c due to sensor separation (sec)
    # Matrix of weights based on squared coherence.
    W1 = np.diag(Coht1c1)

    x1 = optimize.fmin(func=atanfit, x0=[0, 0], args=(f, Phit1c1, W1), disp=False)

    tau1 = x1[0]
    L1 = x1[1]

    print("tau = {:1.4f}s, lag = {:1.4f}s".format(tau1, L1))

    # ---Apply Phase Correction and LP Filter---
    ii = np.squeeze(np.argwhere(ds.p.data > 1))
    i1 = ii[0]
    i2 = ii[-1]
    n = i2 - i1 + 1
    n = (np.floor(n / N) * N).astype("int64")
    # Truncate to be multiple of N elements long
    i2 = (i1 + n).astype("int64")
    # number of segments = dof/2
    m = (n / N).astype("int64")

    # Number of degrees of freedom (power of 2)
    dof = 2 * m
    # Frequency resolution at dof degrees of freedom.
    df = 1 / (N * dt)

    # Transfer function
    f = fft.ifftshift(np.linspace(-N / 2, N / 2 - 1, N) / N / dt)
    H1 = (1 + 1j * 2 * np.pi * f * tau1) * np.exp(1j * 2 * np.pi * f * L1)

    # Low Pass Filter
    f0 = 9  # Cutoff frequency (set to 6 for 24Hz; increasing leads to less filtering)
    # (not sure what the exponent does - set to 6 for 24Hz; decreasing to 3
    # leads to lots of noise)
    LP = 1 / (1 + (f / f0) ** 6)

    # Restructure data with overlapping segments.
    # Staggered segments
    vars = [
        "t",
        "c",
        "p",
    ]
    vard = {}
    for v in vars:
        if v in ds:
            vard[v] = np.zeros((2 * m - 1, N))
            vard[v][: 2 * m - 1 : 2, :] = np.reshape(ds[v].data[i1:i2], newshape=(m, N))
            vard[v][1::2, :] = np.reshape(
                ds[v].data[i1 + int(N / 2) : i2 - int(N / 2)],
                newshape=(m - 1, N),
            )

    time = ds.time[i1:i2]
    lon = ds.lon[i1:i2]
    lat = ds.lat[i1:i2]

    # FFTs of staggered segments (each row)
    Ad = {}
    for v in vars:
        if v in ds:
            Ad[v] = fft.fft(vard[v])

    # Corrected Fourier transforms of temperature.
    Ad["t"] = Ad["t"] * ((H1 * LP) * np.ones((2 * m - 1, 1)))

    # LP filter remaining variables
    vars2 = [
        "c",
        "p",
    ]
    for v in vars2:
        if v in ds:
            Ad[v] = Ad[v] * (LP * np.ones((2 * m - 1, 1)))

    # Inverse transforms of corrected temperature
    # and low passed other variables
    Adi = {}
    for v in vars:
        if v in ds:
            Adi[v] = np.real(fft.ifft(Ad[v]))
            Adi[v] = np.squeeze(
                np.reshape(Adi[v][:, int(N / 4) : (3 * int(N / 4))], newshape=(1, -1))
            )

    time = time[int(N / 4) : -int(N / 4)]
    lon = lon[int(N / 4) : -int(N / 4)]
    lat = lat[int(N / 4) : -int(N / 4)]

    # Generate output structure. Copy attributes over.
    out = xr.Dataset(coords={"time": time})
    out.attrs = ds.attrs
    out["lon"] = lon
    out["lat"] = lat
    for v in vars:
        if v in ds:
            out[v] = xr.DataArray(Adi[v], coords=(out.time,))
            out[v].attrs = ds[v].attrs
    out.assign_attrs(
        dict(
            tau1=tau1,
            L1=L1,
        )
    )

    # ---Recalculate and replot spectra, coherence and phase---
    t1 = Adi["t"][int(N / 4) : -int(N / 4)]  # Now N elements shorter
    c1 = Adi["c"][int(N / 4) : -int(N / 4)]
    # p = Adi["p"][int(N / 4) : -int(N / 4)]

    m = (i2 - N) / N  # number of segments = dof/2
    m = np.floor(m).astype("int64")
    dof = 2 * m  # Number of degrees of freedom (power of 2)
    df = 1 / (N * dt)  # Frequency resolution at dof degrees of freedom.

    window = signal.windows.triang(N) * np.ones((m, N))
    At1 = fft.fft(signal.detrend(np.reshape(t1, newshape=(m, N))) * window)
    Ac1 = fft.fft(signal.detrend(np.reshape(c1, newshape=(m, N))) * window)

    # Positive frequencies only
    At1 = At1[:, 0 : int(N / 2)]
    Ac1 = Ac1[:, 0 : int(N / 2)]
    fn = f[0 : int(N / 2)]

    # Et1 = 2 * np.real(np.nanmean(At1 * np.conj(At1) / df / N**2, axis=0))
    Et1n = 2 * np.nanmean(np.absolute(At1[:, : int(N / 2)]) ** 2, 0) / df / N**2
    Ec1n = 2 * np.nanmean(np.absolute(Ac1[:, : int(N / 2)]) ** 2, 0) / df / N**2

    # Cross Spectral Estimates
    Ct1c1n = 2 * np.nanmean(At1 * np.conj(Ac1) / df / N**2, axis=0)

    # Squared Coherence Estimates
    Coht1c1n = np.real(Ct1c1n * np.conj(Ct1c1n) / (Et1n * Ec1n))
    # 95% confidence bound
    # epsCoht1c1n = np.sqrt(2) * (1 - Coht1c1n) / np.sqrt(Coht1c1n) / np.sqrt(m)
    # epsCoht2c2n = np.sqrt(2) * (1 - Coht2c2n) / np.sqrt(Coht2c2n) / np.sqrt(m)
    # 95% significance level for coherence from Gille notes
    betan = 1 - 0.05 ** (1 / (m - 1))

    # Cross-spectral Phase Estimates
    Phit1c1n = np.arctan2(np.imag(Ct1c1n), np.real(Ct1c1n))
    # 95% error bound
    # epsPhit1c1n = np.arcsin(
    # 	  stats.t.ppf(0.05, dof) * np.sqrt((1 - Coht1c1n) / (dof * Coht1c1n))
    # )
    # epsPhit1c2n = np.arcsin(
    # 	  stats.t.ppf(0.05, dof) * np.sqrt((1 - Coht2c2n) / (dof * Coht2c2n))
    # )
    if plot_spectra:
        fig, ax = plt.subplots(
            nrows=2, ncols=2, figsize=(9, 7), constrained_layout=True
        )
        ax0, ax1, ax2, ax3 = ax.flatten()

        ax0.plot(fold, Et1, label="1 uncorrected", color="0.5")
        ax0.plot(fn, Et1n, label="sensor 1")
        ax0.set(
            yscale="log",
            xscale="log",
            xlabel="frequency [Hz]",
            ylabel=r"spectral density [$^{\circ}$C$^2$/Hz]",
            title="temperature spectra",
        )
        ax0.plot(
            [fn[16], fn[16]],
            [
                dof * Et1n[16] / stats.distributions.chi2.ppf(0.05 / 2, dof),
                dof * Et1n[16] / stats.distributions.chi2.ppf(1 - 0.05 / 2, dof),
            ],
            "k",
        )
        ax0.legend()

        ax1.plot(fold, Ec1, label="1 uncorrected", color="0.5")
        ax1.plot(fn, Ec1n, label="1")
        ax1.set(
            yscale="log",
            xscale="log",
            xlabel="frequency [Hz]",
            ylabel=r"spectral density [mmho$^2$/cm$^2$/Hz]",
            title="conductivity spectra",
        )
        # ax1.plot(
        #     [fn[50], fn[50]],
        #     [
        #         dof * Ec1n[100] / stats.distributions.chi2.ppf(0.05 / 2, dof),
        #         dof * Ec1n[100] / stats.distributions.chi2.ppf(1 - 0.05 / 2, dof),
        #     ],
        #     "k",
        # )

        # Coherence between Temperature and Conductivity
        ax2.plot(fold, Coht1c1, color="0.5")
        ax2.plot(fn, Coht1c1n)

        # ax.plot(fn, Coht1c1 / (1 + 2 * epsCoht1c1), color="b", linewidth=0.5, alpha=0.2)
        # ax.plot(fn, Coht1c1 / (1 - 2 * epsCoht1c1), color="b", linewidth=0.5, alpha=0.2)
        ax2.plot(fn, betan * np.ones(fn.size), "k--")
        ax2.set(
            xlabel="frequency [Hz]",
            ylabel="squared coherence",
            ylim=(-0.1, 1.1),
            title="t/c coherence",
        )

        # Phase between Temperature and Conductivity
        ax3.plot(fold, Phit1c1, color="0.5", marker=".", linestyle="")
        ax3.plot(fn, Phit1c1n, marker=".", linestyle="")
        ax3.set(
            xlabel="frequency [Hz]",
            ylabel="phase [rad]",
            ylim=[-4, 4],
            title="t/c phase",
            # 	  xscale="log",
        )
        ax3.plot(
            fold,
            -np.arctan(2 * np.pi * fold * x1[0]) - 2 * np.pi * fold * x1[1],
            "k--",
        )
        # ax3.plot(
        #     fold,
        #     -np.arctan(2 * np.pi * fold * x2[0]) - 2 * np.pi * fold * x2[1],
        #     "k--",
        # )

    return out


def calc_sal(data):
    # Salinity
    SA, SP = calc_allsal(data.c, data.t, data.p, data.lon, data.lat)

    # Absolute salinity
    data["SA"] = (
        ("time",),
        SA.data,
        {
            "long_name": "absolute salinity",
            "units": "g kg-1",
            "standard_name": "sea_water_absolute_salinity",
        },
    )

    # Practical salinity
    data["s"] = (
        ("time",),
        SP.data,
        {
            "long_name": "practical salinity",
            "units": "",
            "standard_name": "sea_water_practical_salinity",
        },
    )
    return data


def calc_allsal(c, t, p, lon, lat):
    """
    Calculate absolute and practical salinity.

    Wrapper for gsw functions. Converts conductivity
    from S/m to mS/cm if output salinity is less than 5.

    Parameters
    ----------
    c : array-like
        Conductivity. See notes on units above.
    t : array-like
        In-situ temperature (ITS-90), degrees C
    p : array-like
        Sea pressure, dbar
    lon : array-like
        Longitude, -360 to 360 degrees
    lat : array-like
        Latitude, -90 to 90 degrees
    Returns
    -------
    SA : array-like, g/kg
        Absolute Salinity
    SP : array-like
        Practical Salinity
    """
    SP = gsw.SP_from_C(c, t, p)
    if np.nanmean(SP) < 5:
        SP = gsw.SP_from_C(10 * c, t, p)
    SA = gsw.SA_from_SP(SP, p, lon, lat)
    return SA, SP
