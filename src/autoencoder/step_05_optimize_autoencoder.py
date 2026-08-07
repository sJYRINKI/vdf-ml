"""Stage 5: optimize the autoencoder and select by validation MSE.

This stage follows model construction and calls stage 6 after each epoch.
It performs AdamW updates with normalized-space MSE, tracks the established
early-stopping state, restores the best parameter state, and returns the
history consumed by the final saving stage.

The stage receives the stage-4 model, AdamW optimizer, training and
validation loaders, device, and stopping settings. It mutates the supplied
model and leaves it in evaluation mode with the lowest-validation-MSE
parameters restored.
"""

from dataclasses import dataclass
import time

import torch
from torch.nn import functional

from src.autoencoder.step_06_evaluate_autoencoder import (
    evaluate_autoencoder,
)


@dataclass(frozen=True)
class AutoencoderFitResult:
    """Record selected-epoch metadata and autoencoder training history.

    Stage 5 returns this immutable summary after restoring the selected
    parameter state directly into the supplied model. It contains no model
    or optimizer tensors; stage 7 serializes the history and selected epoch.

    Parameters
    ----------
    best_epoch : int
        One-based epoch with the lowest selected validation MSE.
    history : tuple of dict
        One serializable row per completed optimizer epoch.
    epochs_completed : int
        Number of complete epochs.
    total_seconds : float
        Wall time for the optimization loop.
    """

    best_epoch: int
    history: tuple
    epochs_completed: int
    total_seconds: float


def optimize_autoencoder(
    model,
    train_loader,
    validation_loader,
    *,
    optimizer,
    device,
    max_epochs,
    patience,
    min_delta,
):
    """Fit an autoencoder and restore its best validation state.

    Each epoch performs one complete normalized-space MSE optimizer pass,
    then stage 6 evaluates the validation partition. The function mutates
    ``model`` and restores a CPU-cloned copy of the lowest-MSE parameters
    before returning.

    Parameters
    ----------
    model : VdfAutoencoder
        Raw or Hermite reconstruction model.
    train_loader : torch.utils.data.DataLoader
        Shuffled training partition.
    validation_loader : torch.utils.data.DataLoader
        Stable validation partition.
    optimizer : torch.optim.AdamW
        Optimizer whose state is local to this fit.
    device : str or torch.device
        Single training device.
    max_epochs : int
        Maximum complete optimizer epochs.
    patience : int
        Epochs without MSE improvement before stopping.
    min_delta : float
        Required validation-MSE improvement.

    Returns
    -------
    AutoencoderFitResult
        Selected epoch and complete training history.

    Notes
    -----
    ``min_delta`` defines the required MSE decrease. ``patience`` counts
    complete epochs without that decrease, and timing values use wall-clock
    seconds.
    """

    history = []
    best_epoch = 0
    best_mse = float("inf")
    best_state_dict = None
    epochs_without_improvement = 0
    training_start = time.perf_counter()
    for epoch in range(1, int(max_epochs) + 1):
        epoch_start = time.perf_counter()
        train_mse = train_autoencoder_epoch(
            model,
            train_loader,
            optimizer,
            device,
        )
        validation = evaluate_autoencoder(
            model,
            validation_loader,
            device,
            "validation",
        )
        improved = validation["mse"] < best_mse - float(min_delta)
        if improved:
            best_epoch = epoch
            best_mse = validation["mse"]
            best_state_dict = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        history.append(
            {
                "epoch": epoch,
                "learning_rate": float(
                    optimizer.param_groups[0]["lr"]
                ),
                "train_reconstruction_mse": train_mse,
                "validation_reconstruction_mse": validation["mse"],
                "is_best": False,
                "epochs_without_improvement": epochs_without_improvement,
                "epoch_seconds": time.perf_counter() - epoch_start,
            }
        )
        if epochs_without_improvement >= int(patience):
            break

    model.load_state_dict(best_state_dict, strict=True)
    model.eval()
    history = tuple(
        {
            **row,
            "is_best": row["epoch"] == best_epoch,
        }
        for row in history
    )
    return AutoencoderFitResult(
        best_epoch=best_epoch,
        history=history,
        epochs_completed=len(history),
        total_seconds=time.perf_counter() - training_start,
    )


def train_autoencoder_epoch(model, loader, optimizer, device):
    """Run one normalized-space MSE optimizer epoch.

    This inner stage-5 operation reads every training sample, normalizes it
    with the model's stored training statistics, backpropagates mean-squared
    reconstruction error, and applies one AdamW update per batch. It mutates
    both model parameters and optimizer state.

    Parameters
    ----------
    model : VdfAutoencoder
        Reconstruction model updated in place.
    loader : torch.utils.data.DataLoader
        Shuffled training partition.
    optimizer : torch.optim.Optimizer
        Optimizer for ``model`` parameters.
    device : str or torch.device
        Device hosting the model and batches.

    Returns
    -------
    float
        Mean squared error over every reconstructed scalar in normalized
        representation space.
    """

    device = torch.device(device)
    model.train()
    squared_error = 0.0
    value_count = 0
    for batch in loader:
        inputs = batch["inputs"].to(device)
        normalized = model.normalize_inputs(inputs)
        output = model._forward_normalized(normalized)
        loss = functional.mse_loss(output["reconstruction"], normalized)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        difference = output["reconstruction"] - normalized
        squared_error += float(
            torch.sum(
                torch.square(difference.detach().to(dtype=torch.float64))
            )
        )
        value_count += normalized.numel()
    return squared_error / value_count


__all__ = [
    "AutoencoderFitResult",
    "optimize_autoencoder",
    "train_autoencoder_epoch",
]
