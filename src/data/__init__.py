"""Create current VDF datasets through eight ordered stages.

Workflow
--------
Stage 0: orchestrate dataset creation.
Stage 1: open timestep VLSV files and select producers.
Stage 2: find VDF-carrying spatial cells.
Stage 3: plan manual, X/O, and current-layer samples.
Stage 4: extract aligned raw and optional Hermite samples.
Stage 5: write and rename the completed dataset directory.
Stage 6: optionally render colormap and cell-ID-organized VDF frames.
Stage 7: optionally animate colormap and per-cell-ID VDF evolution.
"""

from src.data.step_00_create_dataset import (
    create_dataset,
    run_extraction,
)


__all__ = ["create_dataset", "run_extraction"]
