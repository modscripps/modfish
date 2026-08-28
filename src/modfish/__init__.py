"""
# Overview

Python package to work with data from FastCTD and Epsifish instruments devloped by the
[Multiscale Ocean Dynamics](https://mod.ucsd.edu) group at the [Scripps Institution of Oceanography](https://scripps.ucsd.edu).


# Installation
Clone or download the [repository](https://github.com/modscripps/modfish) and install via `pip install <path-to-modfish>` or, to be able to make changes to the code on the fly, as editable package via `pip install -e <path-to-modfish>`.

With [uv](https://docs.astral.sh/uv/), the package can be added via `uv add <path-to-modfish>` or `uv add --editable <path-to-modfish>`.

Note: The package has **not** been published to pip so you need to install from sources.


# Examples

Read a gridded FastCTD dataset into an `xarray.Dataset` structure:
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

__all__ = ["io", "modraw", "utils"]
from . import io, modraw, utils
