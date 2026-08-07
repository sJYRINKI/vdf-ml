"""Train the VDF autoencoder through eight ordered stages.

Workflow
--------
Stage 0 orchestrates the complete reconstruction training run.
Stage 1 loads saved raw or Hermite samples without supervised targets.
Stage 2 splits complete timesteps into three partitions.
Stage 3 fits training-only feature normalization.
Stage 4 constructs the representation-specific autoencoder.
Stage 5 optimizes parameters and selects by validation MSE.
Stage 6 calculates sample, split, and physical-class reconstruction MSE.
Stage 7 saves the checkpoint, text metrics report, and training history.
"""

from src.autoencoder.load_autoencoder_checkpoint import (
    LoadedAutoencoderCheckpoint,
    load_autoencoder_checkpoint,
)
from src.autoencoder.step_00_train_autoencoder import (
    run_autoencoder_training,
)


__all__ = [
    "LoadedAutoencoderCheckpoint",
    "load_autoencoder_checkpoint",
    "run_autoencoder_training",
]
