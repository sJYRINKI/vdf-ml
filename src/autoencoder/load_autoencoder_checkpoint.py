"""Load the current autoencoder checkpoint for reconstruction.

Checkpoint loading is reusable rather than a numbered training stage. It
directly reconstructs the model and its training-derived feature scaler,
then places the model in evaluation mode on the requested device.
"""

from copy import deepcopy
from dataclasses import dataclass

import torch

from src.autoencoder.step_04_build_autoencoder import VdfAutoencoder
from src.cnn.step_03_scale_cnn_inputs import InputFeatureScaler


@dataclass(frozen=True)
class LoadedAutoencoderCheckpoint:
    """Hold a reconstructed autoencoder and its preprocessing state.

    :func:`load_autoencoder_checkpoint` creates this immutable record before
    reconstruction use. It owns the evaluation-mode model, feature scaler,
    and primitive checkpoint dictionary but no open checkpoint file.

    Parameters
    ----------
    model : VdfAutoencoder
        Evaluation-mode raw or Hermite reconstruction model.
    input_scaler : InputFeatureScaler
        Training-derived feature normalization.
    checkpoint : dict
        Directly loaded version-1 checkpoint record.

    Notes
    -----
    The model is already placed on ``map_location`` and ready for
    normalized-space reconstruction.
    """

    model: VdfAutoencoder
    input_scaler: InputFeatureScaler
    checkpoint: dict


def load_autoencoder_checkpoint(checkpoint_path, *, map_location="cpu"):
    """Load and reconstruct one current autoencoder checkpoint.

    This reusable path follows stage-7 saving. It rebuilds the feature
    scaler and raw or Hermite autoencoder directly from checkpoint fields,
    restores learned tensors, and returns the model in evaluation mode.

    Parameters
    ----------
    checkpoint_path : str or pathlib.Path
        ``autoencoder.pt`` written by the current autoencoder workflow.
    map_location : str or torch.device, optional
        Device used for tensor loading and reconstruction.

    Returns
    -------
    LoadedAutoencoderCheckpoint
        Evaluation model, feature scaler, and saved checkpoint record.
    """

    device = torch.device(map_location)
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )
    input_scaler = InputFeatureScaler.from_dict(
        checkpoint["input_normalization"]
    )
    model = VdfAutoencoder(
        input_mean=input_scaler.mean,
        input_scale=input_scaler.scale,
        **deepcopy(checkpoint["model_architecture"]),
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device)
    model.eval()
    return LoadedAutoencoderCheckpoint(
        model=model,
        input_scaler=input_scaler,
        checkpoint=checkpoint,
    )


__all__ = [
    "LoadedAutoencoderCheckpoint",
    "load_autoencoder_checkpoint",
]
