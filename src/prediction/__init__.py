"""Predict coordinate and region VDF classes through five ordered stages.

Workflow
--------
Stage 1 loads the current CNN and its training-derived scalers.
Stage 2 opens one timestep's VLSV source and resolves VDF/context access.
Stage 3 creates a full raw or Hermite tensor plus 16-value plasma context.
Stage 4 scales context, runs CNN inference, and restores topology units.
Stage 5 saves the shared per-timestep coordinate or region CSV.
"""

from src.prediction.coordinate_prediction import (
    run_coordinate_prediction,
)
from src.prediction.region_prediction import (
    run_region_prediction,
)


__all__ = ["run_coordinate_prediction", "run_region_prediction"]
