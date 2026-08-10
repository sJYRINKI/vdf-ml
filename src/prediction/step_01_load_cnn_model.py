"""Stage 1: load the trained CNN on the requested prediction device.

This stage follows coordinate or region orchestration and precedes opening
the source VLSV file. It receives a current CNN checkpoint path and device
selection and returns the model, scalers, and checkpoint values used by all
later prediction stages.
"""

import torch

from src.cnn.load_cnn_checkpoint import (
    LoadedCnnCheckpoint,
    load_cnn_checkpoint,
)


def load_prediction_model(
    checkpoint_path,
    device="auto",
    model_parallel_gpus=1,
):
    """Load the CNN and training-derived scalers for prediction.

    Coordinate and region prediction share this first stage so checkpoint
    reconstruction, device placement, and evaluation mode are identical.
    The returned object retains the checkpoint mapping used by later stages
    for representation preparation, class order, and inverse topology
    scaling.

    Parameters
    ----------
    checkpoint_path : str or pathlib.Path
        Current CNN checkpoint produced by the training workflow.
    device : str, optional
        PyTorch device string. ``"auto"`` selects CUDA when available and
        otherwise selects CPU.
    model_parallel_gpus : int, optional
        Number of consecutive devices hosting contiguous model stages.

    Returns
    -------
    LoadedCnnCheckpoint
        Reconstructed model, input scaler, topology scaler, and checkpoint
        mapping on the selected runtime devices.
    """

    resolved_device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if str(device).lower() == "auto"
        else torch.device(device)
    )
    loaded = load_cnn_checkpoint(
        checkpoint_path,
        map_location=resolved_device,
        model_parallel_gpus=model_parallel_gpus,
    )
    loaded.model.eval()
    return loaded


__all__ = ["LoadedCnnCheckpoint", "load_prediction_model"]
