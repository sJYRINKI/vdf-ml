"""Train the VDF CNN through nine ordered stages.

Workflow
--------
Stage 0 orchestrates the complete training run.
Stage 1 loads saved raw or Hermite samples and targets.
Stage 2 splits complete timesteps into three partitions.
Stage 3 fits training-only feature and topology scaling.
Stage 4 constructs the representation-specific CNN.
Stage 5 calculates classification and topology loss.
Stage 6 optimizes parameters and selects the best epoch.
Stage 7 calculates final physical-unit evaluation metrics.
Stage 8 saves the checkpoint and consolidated metrics report.
"""

from src.cnn.load_cnn_checkpoint import (
    LoadedCnnCheckpoint,
    load_cnn_checkpoint,
)
from src.cnn.step_00_train_cnn import run_cnn_training


__all__ = [
    "LoadedCnnCheckpoint",
    "load_cnn_checkpoint",
    "run_cnn_training",
]
