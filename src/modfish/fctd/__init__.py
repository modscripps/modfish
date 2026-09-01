"""
FCTD post-processing pipeline: deployment-level concatenation of L0 data,
cast detection, sensor corrections (L1), and per-cast depth gridding.

Stages are pure functions (Dataset/DataTree in and out); file I/O happens
only in `process_deployment` and in the caller. See
`plans/2026-09-01-fctd-pipeline-design.md` for the design.
"""
