"""Load the current CNN checkpoint for evaluation and prediction.

Checkpoint loading is reusable rather than a numbered training stage. It
directly reconstructs the fused VDF/context model and training-derived
representation, plasma-context, and topology scalers saved by stage 8, then
places contiguous model stages on the requested runtime devices.
"""

from copy import deepcopy
from dataclasses import dataclass

import torch

from src.cnn.step_03_scale_cnn_inputs import InputFeatureScaler
from src.cnn.step_04_build_cnn import VdfCNN
from src.learning.topology_supervision import TopologyTargetScaler
from src.representations.plasma_context import PlasmaContextScaler


@dataclass(frozen=True)
class LoadedCnnCheckpoint:
    """Hold a reconstructed CNN and its prediction preprocessing.

    :func:`load_cnn_checkpoint` creates this immutable record immediately
    before evaluation or prediction. It owns the in-memory model, scalers,
    and primitive checkpoint dictionary but no open checkpoint file.

    Parameters
    ----------
    model : VdfCNN
        Evaluation-mode raw or Hermite model.
    input_scaler : InputFeatureScaler
        Training-derived feature normalization.
    plasma_context_scaler : PlasmaContextScaler
        Training-derived scaling for the 16 plasma-context inputs.
    topology_scaler : TopologyTargetScaler
        Scaling used to restore auxiliary outputs to Earth radii.
    checkpoint : dict
        Directly loaded checkpoint record.

    Notes
    -----
    ``model`` is already placed on the requested runtime device stages and
    set to evaluation mode.
    """

    model: VdfCNN
    input_scaler: InputFeatureScaler
    plasma_context_scaler: PlasmaContextScaler
    topology_scaler: TopologyTargetScaler
    checkpoint: dict


def load_cnn_checkpoint(
    checkpoint_path,
    *,
    map_location="cpu",
    model_parallel_gpus=1,
):
    """Load and reconstruct one current CNN checkpoint.

    This reusable loading path follows stage-8 checkpoint saving and
    precedes coordinate or region inference. It rebuilds the model and both
    training-derived scalers directly from checkpoint fields, restores the
    learned parameters, and returns an evaluation-ready record.

    Parameters
    ----------
    checkpoint_path : str or pathlib.Path
        ``model.pt`` written by the current CNN training workflow.
    map_location : str or torch.device, optional
        First device used for reconstructed inference.
    model_parallel_gpus : int, optional
        Number of consecutive devices hosting contiguous model stages.

    Returns
    -------
    LoadedCnnCheckpoint
        Evaluation model, representation/context/topology scalers, and saved
        record.
    """

    device = torch.device(map_location)
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    input_scaler = InputFeatureScaler.from_dict(
        checkpoint["input_normalization"]
    )
    plasma_context_scaler = PlasmaContextScaler(
        mean=checkpoint["plasma_context_mean"],
        scale=checkpoint["plasma_context_scale"],
    )
    topology_scaler = TopologyTargetScaler.from_dict(
        checkpoint["topology_scaler"]
    )
    model = VdfCNN(**deepcopy(checkpoint["model_architecture"]))
    model.place_model_parallel(device, model_parallel_gpus)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    return LoadedCnnCheckpoint(
        model=model,
        input_scaler=input_scaler,
        plasma_context_scaler=plasma_context_scaler,
        topology_scaler=topology_scaler,
        checkpoint=checkpoint,
    )


__all__ = ["LoadedCnnCheckpoint", "load_cnn_checkpoint"]
