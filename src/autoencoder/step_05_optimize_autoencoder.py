"""Stage 5: optimize and select the full-volume multitask autoencoder.

One optimizer owns every encoder, plasma-context, bottleneck, decoder,
reconstruction, and topology parameter even when those modules occupy
several CUDA devices. Each batch is normalized once, passed sequentially
through the placed stages, and updated from one backward call.

Validation total loss, rather than reconstruction MSE alone, selects the
best epoch. The selected state is copied to CPU so checkpoint content is
independent from runtime CUDA stage ownership.
"""

from dataclasses import dataclass
import time

import torch

from src.autoencoder.autoencoder_loss import calculate_autoencoder_loss
from src.autoencoder.step_06_evaluate_autoencoder import evaluate_autoencoder


@dataclass(frozen=True)
class AutoencoderFitResult:
    """Record validation-total-loss selection and epoch history.

    Parameters
    ----------
    best_epoch : int
        One-based epoch with the lowest qualifying validation total loss.
    best_validation_total_loss : float
        Selected complete-volume MSE plus weighted masked topology loss.
    history : tuple of dict
        Per-epoch reconstruction, topology, and total objectives plus timing.
    epochs_completed : int
        Number of complete optimization epochs.
    total_seconds : float
        Wall time for optimization in seconds.
    """

    best_epoch: int
    best_validation_total_loss: float
    history: tuple
    epochs_completed: int
    total_seconds: float


def optimize_autoencoder(
    model,
    train_loader,
    validation_loader,
    *,
    optimizer,
    topology_scaler,
    topology_loss_weight,
    max_epochs,
    patience,
    min_delta,
):
    """Fit one placed autoencoder and restore its best total-loss state.

    Every epoch performs one combined-objective optimizer pass and then
    evaluates validation rows. A CPU-cloned state replaces the live mixed-
    device parameters only when validation total loss improves sufficiently.

    Parameters
    ----------
    model : VdfAutoencoder
        Full-volume Conv3d model whose consecutive stages may span devices.
    train_loader : torch.utils.data.DataLoader
        Shuffled complete-timestep training partition.
    validation_loader : torch.utils.data.DataLoader
        Stable validation partition.
    optimizer : torch.optim.AdamW
        Single optimizer owning every trainable model parameter once.
    topology_scaler : TopologyTargetScaler
        Scaling fitted only from valid training topology entries.
    topology_loss_weight : float
        Weight in ``MSE + weight * masked Smooth L1``.
    max_epochs : int
        Maximum complete optimizer epochs.
    patience : int
        Epochs without sufficient validation-total-loss improvement.
    min_delta : float
        Required decrease in validation total loss.

    Returns
    -------
    AutoencoderFitResult
        Best epoch, best validation total objective, and complete history.

    Notes
    -----
    Test data is never passed to this function. One ``backward`` call lets
    autograd traverse activation transfers between model-stage devices.
    """

    history = []
    best_epoch = 0
    best_total_loss = float("inf")
    best_state_dict = None
    epochs_without_improvement = 0
    training_start = time.perf_counter()
    for epoch in range(1, int(max_epochs) + 1):
        epoch_start = time.perf_counter()
        train_metrics = train_autoencoder_epoch(
            model,
            train_loader,
            optimizer,
            topology_loss_weight,
        )
        validation = evaluate_autoencoder(
            model,
            validation_loader,
            topology_scaler,
            "validation",
            topology_loss_weight,
        )
        improved = (
            validation["total_loss"]
            < best_total_loss - float(min_delta)
        )
        if improved:
            best_epoch = epoch
            best_total_loss = validation["total_loss"]
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
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "train_reconstruction_loss": train_metrics[
                    "reconstruction_loss"
                ],
                "train_topology_loss": train_metrics["topology_loss"],
                "train_total_loss": train_metrics["total_loss"],
                "validation_reconstruction_loss": validation[
                    "reconstruction_mse"
                ],
                "validation_topology_loss": validation["topology_loss"],
                "validation_total_loss": validation["total_loss"],
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
        {**row, "is_best": row["epoch"] == best_epoch}
        for row in history
    )
    return AutoencoderFitResult(
        best_epoch=best_epoch,
        best_validation_total_loss=best_total_loss,
        history=history,
        epochs_completed=len(history),
        total_seconds=time.perf_counter() - training_start,
    )


def train_autoencoder_epoch(
    model,
    loader,
    optimizer,
    topology_loss_weight,
):
    """Run one full-volume reconstruction and topology optimizer epoch.

    Each batch crosses the placed stages once and contributes one backward
    call. Scalar sums are accumulated separately so epoch reconstruction and
    topology means remain correct for unequal batch sizes and mask counts.

    Parameters
    ----------
    model : VdfAutoencoder
        Placed raw or Hermite Conv3d model updated in place.
    loader : torch.utils.data.DataLoader
        On-demand memory-mapped complete-volume training partition.
    optimizer : torch.optim.Optimizer
        One optimizer containing each mixed-device parameter exactly once.
    topology_loss_weight : float
        Multiplier applied to globally masked topology Smooth L1.

    Returns
    -------
    dict
        Aggregate reconstruction loss, topology loss, and weighted total
        objective over the epoch.

    Notes
    -----
    Reconstruction MSE covers every normalized voxel or coefficient.
    Missing topology entries contribute no auxiliary loss, while their
    samples continue to update the reconstruction branch.
    """

    model.train()
    reconstruction_error = 0.0
    reconstruction_value_count = 0
    topology_error = 0.0
    valid_topology_count = 0
    for batch in loader:
        normalized = model.normalize_inputs(batch["vdf_input"])
        output = model.forward_from_normalized(
            normalized,
            batch["plasma_context"],
        )
        target_inputs = normalized.to(model.output_device)
        topology_targets = batch["topology_targets"].to(model.output_device)
        topology_mask = batch["topology_mask"].to(model.output_device)
        losses = calculate_autoencoder_loss(
            output["reconstruction"],
            target_inputs,
            output["topology_predictions"],
            topology_targets,
            topology_mask,
            topology_loss_weight,
        )
        optimizer.zero_grad(set_to_none=True)
        losses["total_loss"].backward()
        optimizer.step()

        difference = output["reconstruction"] - target_inputs
        reconstruction_error += float(
            torch.sum(
                torch.square(
                    difference.detach().to(dtype=torch.float64)
                )
            )
        )
        reconstruction_value_count += target_inputs.numel()
        batch_valid_count = losses["valid_topology_count"]
        topology_error += (
            float(losses["topology_loss"].detach()) * batch_valid_count
        )
        valid_topology_count += batch_valid_count

    reconstruction_loss = (
        reconstruction_error / reconstruction_value_count
    )
    topology_loss = (
        topology_error / valid_topology_count
        if valid_topology_count
        else 0.0
    )
    return {
        "reconstruction_loss": reconstruction_loss,
        "topology_loss": topology_loss,
        "total_loss": (
            reconstruction_loss
            + float(topology_loss_weight) * topology_loss
        ),
    }


__all__ = [
    "AutoencoderFitResult",
    "optimize_autoencoder",
    "train_autoencoder_epoch",
]
