#!/usr/bin/env python
# coding: utf-8
"""
Batch conversion of `.modraw` files to NetCDF.
"""

import logging
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
    """
    file = Path(file)
    outdir = Path(outdir)
    outpath = outdir / f"{file.stem}.nc"

    # Skip if exists and overwrite is False
    if outpath.exists() and not overwrite:
        return None

    try:
        tree = read(file)
        tree.to_netcdf(outpath)
        return outpath
    except Exception as e:
        logger.warning("skipping %s: %s", file, e)
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
        If True, process files in parallel using ProcessPoolExecutor.
        Default: False (sequential processing).

        When parallel=True, each worker process independently reads the
        input file and writes to NetCDF. The caller must be able to pickle
        the input data (standard for built-in types and numpy/xarray objects).
        File-level exceptions (bad input, I/O errors) are caught and logged;
        errors in worker setup are not caught and will fail the entire batch.

    Returns
    -------
    list of Path
        Paths to `.nc` files that were successfully written this call.
        Skipped files (existing without overwrite) are not included.

    Notes
    -----
    Exception handling: Any exception during read(), write, or intermediate
    processing is caught per file and logged as a warning. The batch continues
    with remaining files.

    Parallel processing: Uses `concurrent.futures.ProcessPoolExecutor` with
    one worker per input file (capped by system CPU count). Each worker is
    independent; there is no shared state or progress reporting. The main
    process collects results and returns the list of written paths.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    files = [Path(f) for f in files]

    if not parallel:
        # Sequential: process files one by one
        written = []
        for file in files:
            result = _worker(file, outdir, overwrite)
            if result is not None:
                written.append(result)
        return written
    else:
        # Parallel: use ProcessPoolExecutor
        written = []
        with ProcessPoolExecutor() as executor:
            futures = [
                executor.submit(_worker, file, outdir, overwrite)
                for file in files
            ]
            for future in futures:
                result = future.result()
                if result is not None:
                    written.append(result)
        return written
