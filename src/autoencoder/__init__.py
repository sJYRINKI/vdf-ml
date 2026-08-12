"""Train the full-volume topology-aware VDF autoencoder in eight stages.

Workflow
--------
Stage 0 orchestrates the complete reconstruction training run.
Stage 1 reads complete raw or Hermite volumes plus aligned 16-value plasma
context on demand from read-only memory maps.
Stage 2 splits complete timesteps into three partitions.
Stage 3 fits training-only representation normalization.
Stage 4 constructs and places one Conv3d autoencoder with context fusion.
Stage 5 optimizes combined reconstruction and masked topology loss.
Stage 6 reports reconstruction, topology, and total objectives.
Stage 7 saves the checkpoint and four established artifacts.
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
