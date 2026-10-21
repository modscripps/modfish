"""
# Overview

Python package to work with data from FastCTD and Epsifish instruments developed by the
[Multiscale Ocean Dynamics](https://mod.ucsd.edu) group at the [Scripps Institution of Oceanography](https://scripps.ucsd.edu).

`modraw` decodes FCTD `.modraw` files into per-stream NetCDF (L0). `fctd` chains L0
concatenation, cast detection, sensor corrections, and depth gridding into L1 and gridded
products (`fctd.process_deployment`). `chi` computes microconductivity chi from the L0
`efe/c1` stream and adds it to the L1 and gridded products. `tc` holds the T-C sensor
corrections `fctd` applies and the tools that estimate their parameters. `io` reads
legacy Matlab-gridded FCTD and Epsifish output.


# Installation
Clone or download the [repository](https://github.com/modscripps/modfish) and install via `pip install <path-to-modfish>` or, to be able to make changes to the code on the fly, as editable package via `pip install -e <path-to-modfish>`.

With [uv](https://docs.astral.sh/uv/), the package can be added via `uv add <path-to-modfish>` or `uv add --editable <path-to-modfish>`.

Note: The package has **not** been published to pip so you need to install from sources.


# Examples

Process one deployment from raw `.modraw` files to L1 and gridded products:
```python
from pathlib import Path
import modfish

raw = sorted(Path("raw").glob("*.modraw"))
modfish.modraw.convert(raw, "l0")
l0 = sorted(Path("l0").glob("*.nc"))
config = modfish.fctd.FCTDConfig.from_yaml("fctd.yml")
l1_path, grid_path = modfish.fctd.process_deployment(l0, "proc", "d07", config=config)
```

Read a legacy gridded FastCTD dataset into an `xarray.Dataset` structure:
```python
import modfish
ds = modfish.io.load_fctd_grid("FCTDgrid.mat")
```

Read only downcasts from the gridded FastCTD section:
```python
import modfish
ds = modfish.io.load_fctd_grid("FCTDgrid.mat", what="dn")
```

"""
import importlib.metadata

__author__ = """Gunnar Voet"""
__email__ = "gvoet@ucsd.edu"
__version__ = importlib.metadata.version("modfish")

__all__ = ["io", "fctd", "chi", "modraw", "tc", "utils"]
from . import io, fctd, chi, modraw, tc, utils
