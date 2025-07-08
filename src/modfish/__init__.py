"""
# Overview

Python package to work with data from FastCTD and Epsifish instruments devloped by the
[Multiscale Ocean Dynamics](https://mod.ucsd.edu) group at the [Scripps Institution of Oceanography](https://scripps.ucsd.edu).

# Installation
Clone or download the [repository](https://github.com/modscripps/modfish) and install via `pip install modfish` or, to be able to make changes to the code on the fly, as editable package via `pip install -e modfish`.

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

__author__ = """Gunnar Voet"""
__email__ = "gvoet@ucsd.edu"
__version__ = "2024.12.0"

__all__ = ["io", "utils"]
from . import io, utils
