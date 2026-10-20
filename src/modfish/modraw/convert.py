"""
Batch conversion of `.modraw` files to NetCDF.
"""

import logging
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from .reader import read


logger = logging.getLogger(__name__)


def _worker(file, outdir, overwrite):
    """Module-level worker function for ProcessPoolExecutor.

    Must be at module level for pickling to work with multiprocessing.

    Parameters
    ----------
    file : Path or str
        Input `.modraw` file.
    outdir : Path or str
        Output directory for the `.nc` file.
    overwrite : bool
        Whether to overwrite existing output files.

    Returns
    -------
    Path or None
        Path to the written `.nc` file if successful, None if skipped or failed.

    Notes
    -----
    The tree is written to `<stem>.nc.tmp` and renamed into place, so a
    write that dies partway (killed job, full disk) leaves no `<stem>.nc`
    for a later `overwrite=False` run to mistake for a finished file.
    """
    file = Path(file)
    outdir = Path(outdir)
    outpath = outdir / f"{file.stem}.nc"

    # Skip if exists and overwrite is False
    if outpath.exists() and not overwrite:
        return None

    tmppath = outpath.with_name(outpath.name + ".tmp")
    try:
        tree = read(file)
        tree.to_netcdf(tmppath)
        os.replace(tmppath, outpath)
        return outpath
    except Exception as e:
        logger.warning("skipping %s: %s", file, e)
        tmppath.unlink(missing_ok=True)
        return None


def convert(files, outdir, overwrite=False, parallel=False):
    """Convert `.modraw` files to NetCDF in batch.

    Reads each `.modraw` file with `read()` and writes a `.nc` file to
    `outdir`. Output files are named `<input_stem>.nc`. Files that raise
    exceptions during read or write are logged and skipped without halting.

    Parameters
    ----------
    files : list of Path or str
        Input `.modraw` files to convert.
    outdir : Path or str
        Output directory for `.nc` files. Created if missing.
    overwrite : bool, optional
        If True, overwrite existing output files. Default: False.
        If False, skip files whose outputs already exist.
    parallel : bool, optional
        If True, convert files in a `ProcessPoolExecutor` with its default
        number of workers (the CPU count). Default: False. Errors in
        worker setup are not caught and fail the whole batch.

    Returns
    -------
    list of Path
        Paths to `.nc` files that were successfully written this call, in
        input order. Skipped files (existing without overwrite) are not
        included.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    files = [Path(f) for f in files]

    args = (files, [outdir] * len(files), [overwrite] * len(files))
    if parallel:
        with ProcessPoolExecutor() as executor:
            results = list(executor.map(_worker, *args))
    else:
        results = list(map(_worker, *args))
    return [r for r in results if r is not None]
