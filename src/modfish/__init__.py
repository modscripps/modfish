"""
# Overview

Python package to work with data from FastCTD and Epsifish instruments devloped by the
[Multiscale Ocean Dynamics](https://mod.ucsd.edu) group at the [Scripps Institution of Oceanography](https://scripps.ucsd.edu).

# Example
Read a gridded FastCTD dataset into an `xarray.Dataset` structure:
```python
import modfish
ds = modfish.io.load_fctd_grid("FCTDgrid.mat")
```

# Markdown Syntax
## Figures
Narrow image:
```markdown
![nimage](path/to/figure)
```
Medium sized image:
```markdown
![image](path/to/figure)
```
Wide image:
```markdown
![wimage](path/to/figure)
```


# License

.. include:: ../../LICENSE
"""

__author__ = """Gunnar Voet"""
__email__ = 'gvoet@ucsd.edu'
__version__ = '2024.12.0'

__all__ = ["io", "utils"]
from . import io, utils
