"""
# Overview

Python package to work with data from FastCTD and Epsifish instruments devloped by the
[Multiscale Ocean Dynamics](https://mod.ucsd.edu) group at the [Scripps Institution of Oceanography](https://scripps.ucsd.edu).

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
__email__ = 'gvoet@ucsd.edu'
__version__ = '2024.12.0'

__all__ = ["io", "utils"]
from . import io, utils
