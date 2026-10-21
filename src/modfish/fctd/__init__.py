"""
FCTD post-processing pipeline: deployment-level concatenation of L0 data,
cast detection, sensor corrections (L1), and per-cast depth gridding.

Stages run in the order `concat_l0` -> `make_l1` -> optional `add_chi` ->
`grid_casts`. Each stage is a pure function (Dataset/DataTree in and out).
File I/O happens only in `concat_l0` (reading), `process_deployment`,
`add_chi_to_products`, and the caller.
"""

from modfish.chi.config import ChiParams
from modfish.fctd.casts import casts_to_dataset, find_casts, label_casts
from modfish.fctd.concat import concat_l0
from modfish.fctd.config import CastParams, FCTDConfig, GridParams, TCParams
from modfish.fctd.grid import depth_edges, grid_casts
from modfish.fctd.l1 import make_l1
from modfish.fctd.pipeline import add_chi_to_products, process_deployment

__all__ = [
    "CastParams",
    "ChiParams",
    "FCTDConfig",
    "GridParams",
    "TCParams",
    "add_chi_to_products",
    "casts_to_dataset",
    "concat_l0",
    "depth_edges",
    "find_casts",
    "grid_casts",
    "label_casts",
    "make_l1",
    "process_deployment",
]
