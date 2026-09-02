"""Deployment driver: chains concat -> l1 -> grid and writes both products.

The only stage in the pipeline that touches the filesystem for its outputs
(`concat_l0` itself reads the per-file inputs); every other stage is a pure
Dataset/DataTree transform. See
`plans/2026-09-01-fctd-pipeline-design.md` for the stage order and
rationale.
"""

import logging
from pathlib import Path

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

    Chains `concat_l0` -> `make_l1` -> `grid_casts` and writes both
    resulting products to `outdir`.

    Parameters
    ----------
    files : list of Path or str
        Per-file L0 netCDF paths, passed to `concat_l0`.
    outdir : Path or str
        Output directory; created (with parents) if missing.
    name : str
        Deployment name, used to build the output filenames and, on
        failure, named in the wrapped error message.
    config : FCTDConfig or None, optional
        Pipeline configuration. Defaults to `FCTDConfig()`.
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
        raised by `make_l1` or `grid_casts` (e.g. zero casts detected).
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

    l0_tree = concat_l0(files, keep_counts=config.keep_counts)

    try:
        l1_tree = make_l1(l0_tree, config)
        grid = grid_casts(l1_tree, config.grid)
    except ValueError as exc:
        raise ValueError(f"process_deployment({name}): {exc}") from exc

    l1_tree.to_netcdf(l1_path)
    grid.to_netcdf(grid_path)

    return l1_path, grid_path
