"""Load and place a full-volume autoencoder checkpoint.

Checkpoint tensors are always read on CPU before the saved Conv3d
architecture and both training-derived scalers are reconstructed. Runtime
CPU, one-GPU, or multi-GPU stage ownership is then assigned from caller
arguments; CUDA identifiers are not persisted architecture.
"""

from copy import deepcopy
from dataclasses import dataclass

import torch

from src.autoencoder.step_04_build_autoencoder import VdfAutoencoder
from src.cnn.step_03_scale_cnn_inputs import InputFeatureScaler
from src.learning.topology_supervision import TopologyTargetScaler


@dataclass(frozen=True)
class LoadedAutoencoderCheckpoint:
    """Hold a reconstructed model and its two preprocessing scalers.

    Parameters
    ----------
    model : VdfAutoencoder
        Evaluation-mode complete-volume model on requested runtime stages.
    input_scaler : InputFeatureScaler
        Training-only raw-voxel or Hermite-coefficient normalization.
    topology_scaler : TopologyTargetScaler
        Training-only six-target scaler used to recover Earth radii.
    checkpoint : dict
        Directly loaded device-independent checkpoint record.

    Notes
    -----
    Raw model inputs have shape ``(batch, 1, vx, vy, vz)`` and Hermite
    inputs have ``(batch, 1, n1, n2, n3)``. Topology remains an auxiliary
    output and is never required as model input during reconstruction.
    """

    model: VdfAutoencoder
    input_scaler: InputFeatureScaler
    topology_scaler: TopologyTargetScaler
    checkpoint: dict


def load_autoencoder_checkpoint(
    checkpoint_path,
    *,
    map_location="cpu",
    model_parallel_gpus=1,
):
    """Load one full-volume checkpoint with fresh runtime stage placement.

    The loader rebuilds representation normalization, topology scaling, and
    the Conv3d architecture on CPU, restores learned values, then applies the
    caller's current one-process stage placement before inference.

    Parameters
    ----------
    checkpoint_path : str or pathlib.Path
        ``autoencoder.pt`` written by the current workflow.
    map_location : str or torch.device, optional
        CPU or first CUDA device for inference stage placement.
    model_parallel_gpus : int, optional
        Requested visible CUDA count for consecutive model stages. CPU uses
        the same stage mechanism with one effective device.

    Returns
    -------
    LoadedAutoencoderCheckpoint
        Placed evaluation model, representation scaler, topology scaler, and
        saved device-independent checkpoint dictionary.

    Notes
    -----
    Retired raw-plane and reconstruction-only checkpoints are not migrated;
    they require retraining under the complete Conv3d architecture.
    """

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    input_scaler = InputFeatureScaler.from_dict(
        checkpoint["input_normalization"]
    )
    topology_scaler = TopologyTargetScaler.from_dict(
        checkpoint["topology_scaler"]
    )
    model = VdfAutoencoder(
        input_mean=input_scaler.mean,
        input_scale=input_scaler.scale,
        **deepcopy(checkpoint["model_architecture"]),
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.place_model_parallel(map_location, model_parallel_gpus)
    model.eval()
    return LoadedAutoencoderCheckpoint(
        model=model,
        input_scaler=input_scaler,
        topology_scaler=topology_scaler,
        checkpoint=checkpoint,
    )


__all__ = [
    "LoadedAutoencoderCheckpoint",
    "load_autoencoder_checkpoint",
]
