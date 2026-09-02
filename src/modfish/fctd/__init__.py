"""
FCTD post-processing pipeline: deployment-level concatenation of L0 data,
cast detection, sensor corrections (L1), and per-cast depth gridding.

Stages are pure functions (Dataset/DataTree in and out); file I/O happens
only in `process_deployment` and in the caller. See
`plans/2026-09-01-fctd-pipeline-design.md` for the design.
"""

from modfish.fctd.casts import casts_to_dataset, find_casts, label_casts
from modfish.fctd.concat import concat_l0
from modfish.fctd.config import CastParams, FCTDConfig, GridParams, TCParams
from modfish.fctd.grid import grid_casts
from modfish.fctd.l1 import make_l1

__all__ = [
    "CastParams",
    "FCTDConfig",
    "GridParams",
    "TCParams",
    "casts_to_dataset",
    "concat_l0",
    "find_casts",
    "grid_casts",
    "label_casts",
    "make_l1",
]
