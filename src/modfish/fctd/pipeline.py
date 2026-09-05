"""Deployment driver: chains concat -> l1 -> grid and writes both products.

The only stage in the pipeline that touches the filesystem for its outputs
(`concat_l0` itself reads the per-file inputs); every other stage is a pure
Dataset/DataTree transform. See
`plans/2026-09-01-fctd-pipeline-design.md` for the stage order and
rationale.
"""

import logging
from pathlib import Path

import xarray as xr

from modfish.chi import add_chi
from modfish.fctd.concat import concat_l0
from modfish.fctd.config import FCTDConfig
from modfish.fctd.grid import grid_casts
from modfish.fctd.l1 import make_l1

logger = logging.getLogger(__name__)


def process_deployment(
    files: list,
    outdir,
    name: str,
    config: FCTDConfig | None = None,
    overwrite: bool = False,
) -> tuple[Path, Path]:
    """Process one deployment's L0 files into L1 and gridded products.

    Chains `concat_l0` -> `make_l1` -> `add_chi` (when `config.chi.enabled`)
    -> `grid_casts` and writes both resulting products to `outdir`.

    Parameters
    ----------
    files : list of Path or str
        Per-file L0 netCDF paths, passed to `concat_l0` and, when chi is
        enabled, to `add_chi` as the `efe/c1` source.
    outdir : Path or str
        Output directory; created (with parents) if missing.
    name : str
        Deployment name, used to build the output filenames and, on
        failure, named in the wrapped error message.
    config : FCTDConfig or None, optional
        Pipeline configuration. Defaults to `FCTDConfig()`. `config.groups`
        selects which L0 groups `concat_l0` loads; `config.chi` controls
        whether `add_chi` runs.
    overwrite : bool, optional
        When False (default) and both output files already exist,
        processing is skipped and the existing paths are returned without
        rereading the inputs.

    Returns
    -------
    tuple of pathlib.Path
        `(l1_path, grid_path)`: `<outdir>/fctd_<name>_l1.nc` (the L1
        `xr.DataTree`, written with `to_netcdf`) and
        `<outdir>/fctd_<name>_grid.nc` (the gridded `xr.Dataset`).

    Raises
    ------
    ValueError
        Re-raised, with `name` added to the message, from any `ValueError`
        raised by `make_l1`, `add_chi`, or `grid_casts` (e.g. zero casts
        detected).
    """
    if config is None:
        config = FCTDConfig()

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    l1_path = outdir / f"fctd_{name}_l1.nc"
    grid_path = outdir / f"fctd_{name}_grid.nc"

    if not overwrite and l1_path.exists() and grid_path.exists():
        logger.info(
            "process_deployment(%s): both outputs already exist, skipping "
            "(l1=%s, grid=%s)",
            name,
            l1_path,
            grid_path,
        )
        return l1_path, grid_path

    l0_tree = concat_l0(files, keep_counts=config.keep_counts, groups=config.groups)

    try:
        l1_tree = make_l1(l0_tree, config)
        if config.chi.enabled:
            l1_tree = add_chi(l1_tree, files, config.chi)
        grid = grid_casts(l1_tree, config.grid)
    except ValueError as exc:
        raise ValueError(f"process_deployment({name}): {exc}") from exc

    l1_tree.to_netcdf(l1_path)
    grid.to_netcdf(grid_path)

    return l1_path, grid_path


def add_chi_to_products(l1_path, l0_files, config: FCTDConfig, grid_path=None):
    """Add the `chi` group to an existing L1 file and regrid.

    Reads the L1 tree fully into memory, runs `add_chi` with `config.chi`,
    writes the result to a temporary file beside `l1_path` and replaces the
    original; when `grid_path` is given the grid is recomputed with
    `config.grid` and replaced the same way.

    Parameters
    ----------
    l1_path : Path or str
        Existing L1 product.
    l0_files : sequence of Path
        The deployment's L0 files (the `efe/c1` source).
    config : FCTDConfig
        `config.chi.enabled` must be True.
    grid_path : Path or str or None, optional
        Existing grid product to rewrite.

    Returns
    -------
    tuple
        `(l1_path, grid_path)` as Paths, `grid_path` None when not given.

    Raises
    ------
    ValueError
        From `add_chi` when `config.chi.enabled` is False or
        `config.chi.gain` is None.
    """
    l1_path = Path(l1_path)
    with xr.open_datatree(l1_path) as tree:
        l1_tree = tree.load()
    l1_tree = add_chi(l1_tree, l0_files, config.chi)
    tmp = l1_path.with_suffix(".tmp.nc")
    l1_tree.to_netcdf(tmp)
    tmp.replace(l1_path)
    if grid_path is None:
        return l1_path, None
    grid_path = Path(grid_path)
    grid = grid_casts(l1_tree, config.grid)
    tmp = grid_path.with_suffix(".tmp.nc")
    grid.to_netcdf(tmp)
    tmp.replace(grid_path)
    return l1_path, grid_path
