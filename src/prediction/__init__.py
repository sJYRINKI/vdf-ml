"""Predict coordinate and region VDF classes through five ordered stages.

Workflow
--------
Stage 1 loads the current CNN and its training-derived scalers.
Stage 2 opens one timestep's VLSV source and prepares VDF access.
Stage 3 creates a full-resolution raw or complete Hermite input tensor.
Stage 4 runs CNN inference and restores physical topology units.
Stage 5 saves the shared per-timestep coordinate or region CSV.
"""

from src.prediction.coordinate_prediction import (
    run_coordinate_prediction,
)
from src.prediction.region_prediction import (
    run_region_prediction,
)


__all__ = ["run_coordinate_prediction", "run_region_prediction"]
