"""Create current VDF datasets through eight ordered stages.

Workflow
--------
Stage 0: orchestrate dataset creation.
Stage 1: open timestep VLSV files and select producers.
Stage 2: find VDF-carrying spatial cells.
Stage 3: plan manual, X/O, and current-layer samples.
Stage 4: extract aligned raw, sixteen-value plasma context, and optional
Hermite samples. Magnetic field, electric field, and configured-population
bulk velocity are saved in Cartesian component order; density and canonical
``(Pxx, Pyy, Pzz, Pxy, Pxz, Pyz)`` pressure components complete each float32
context row. The same B and V vectors support optional Hermite rotation.
Serial extraction uses one timestep iterator, while
``creation.extraction_n_jobs`` runs independent timestep workers that stage
temporary arrays for ordered parent-process merging.
Stage 5: write and rename the completed dataset directory.
Stage 6: optionally render colormap and cell-ID-organized VDF frames.
Stage 7: optionally animate colormap and per-cell-ID VDF evolution.
"""

from src.data.step_00_create_dataset import (
    create_dataset,
    run_extraction,
)


__all__ = ["create_dataset", "run_extraction"]
