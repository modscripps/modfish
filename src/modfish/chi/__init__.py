"""Microconductivity chi from the FCTD `efe/c1` stream.

See `plans/2026-09-04-chi-design.md`. Modules: `config`, `response`,
`batchelor`, `load`, `spectra`, `closure`, `fit`, `pipeline`. Nothing here
imports `modfish.fctd` at module level (`modfish.fctd.config` imports
`ChiParams` from here).
"""

from modfish.chi.config import ChiParams
from modfish.chi.pipeline import add_chi, chi_dataset

__all__ = ["ChiParams", "add_chi", "chi_dataset"]
