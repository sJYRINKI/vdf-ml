"""Stage 6: optimize the CNN and select its best validation epoch.

This stage follows loss definition and calls stage 7 after each epoch for
validation metrics. It performs AdamW updates in stable loader order,
selects by macro F1 with the established loss tie-breaker, restores the best
state, and returns the training history for final saving.

The stage receives the stage-4 model, AdamW optimizer, training and
validation loaders, target scaling, and stopping settings. It mutates the
model during fitting and leaves it in evaluation mode with the selected
parameters restored.
"""

from dataclasses import dataclass
import random
import time

import numpy as np
import torch

from src.cnn.step_01_load_cnn_data import create_cnn_dataloader
from src.cnn.step_05_calculate_cnn_loss import calculate_cnn_loss
from src.cnn.step_07_evaluate_cnn import evaluate_cnn


@dataclass(frozen=True)
class EpochTrainingMetrics:
    """Hold aggregate losses from one optimizer epoch.

    :func:`train_cnn_epoch` creates this immutable summary after updating
    the model over a complete training partition. Stage 6 combines it with
    validation metrics to append one serializable history row.

    Parameters
    ----------
    total_loss : float
        Cross entropy plus weighted topology loss.
    classification_loss : float
        Sample-weighted classification cross entropy.
    topology_loss : float
        Valid-target-weighted Smooth L1 loss.
    valid_topology_target_count : int
        Number of valid topology scalars seen.
    sample_count : int
        Number of representation samples seen.
    """

    total_loss: float
    classification_loss: float
    topology_loss: float
    valid_topology_target_count: int
    sample_count: int


@dataclass(frozen=True)
class CnnTrainingFitResult:
    """Record selected-epoch metadata and the complete training history.

    Stage 6 returns this record after restoring the selected parameter state
    directly into the supplied model. The record therefore owns summary
    values only; it does not duplicate model tensors or optimizer state.

    Parameters
    ----------
    best_epoch : int
        One-based epoch selected by validation macro F1 and loss.
    best_validation_result : EvaluationResult
        Metrics calculated from the selected model state.
    history : tuple of dict
        One serializable record per completed epoch.
    epochs_completed : int
        Number of optimizer epochs completed.
    total_seconds : float
        Wall time for the epoch loop.
    """

    best_epoch: int
    best_validation_result: object
    history: tuple
    epochs_completed: int
    total_seconds: float


def optimize_cnn(
    model,
    train_loader,
    validation_loader,
    *,
    optimizer,
    topology_scaler,
    class_mapping,
    device,
    topology_loss_weight,
    max_epochs,
    patience,
    min_delta,
    gradient_clip_norm=None,
):
    """Fit a CNN and restore the validation-selected parameter state.

    Each epoch performs one complete optimizer pass followed by stage-7
    validation. Macro F1 is the primary selection metric, with total loss
    breaking ties within ``min_delta``. The function mutates ``model`` and
    restores the selected CPU-cloned state before returning.

    Parameters
    ----------
    model : VdfCNN
        Raw or Hermite multitask model.
    train_loader : torch.utils.data.DataLoader
        Shuffled training partition loader.
    validation_loader : torch.utils.data.DataLoader
        Stable validation partition loader.
    optimizer : torch.optim.AdamW
        Optimizer whose state is local to this fit.
    topology_scaler : TopologyTargetScaler
        Training-derived target scaling.
    class_mapping : ClassMapping
        Physical classes in model-output order.
    device : str or torch.device
        First model-stage device receiving input batches.
    topology_loss_weight : float
        Auxiliary topology-loss multiplier.
    max_epochs : int
        Maximum complete optimizer epochs.
    patience : int
        Epochs without selection improvement before stopping.
    min_delta : float
        Macro-F1 comparison tolerance.
    gradient_clip_norm : float, optional
        Maximum global gradient norm across every model stage.

    Returns
    -------
    CnnTrainingFitResult
        Selected epoch, its validation metrics, and history.

    Notes
    -----
    History timing values report wall-clock seconds. Final train,
    validation, and test evaluation occurs later in the stage-0
    orchestrator.
    """

    device = torch.device(device)
    history = []
    best_epoch = 0
    best_macro_f1 = float("-inf")
    best_validation_loss = float("inf")
    best_validation_result = None
    best_state_dict = None
    epochs_without_improvement = 0
    fit_start = time.perf_counter()

    for epoch in range(1, int(max_epochs) + 1):
        epoch_start = time.perf_counter()
        training_metrics = train_cnn_epoch(
            model,
            train_loader,
            optimizer=optimizer,
            device=device,
            topology_loss_weight=topology_loss_weight,
            gradient_clip_norm=gradient_clip_norm,
        )
        validation_result = evaluate_cnn(
            model=model,
            loader=validation_loader,
            topology_scaler=topology_scaler,
            class_mapping=class_mapping,
            device=device,
            topology_loss_weight=topology_loss_weight,
            split_name="validation",
        )
        improved = is_better_validation_result(
            macro_f1=validation_result.macro_f1,
            total_loss=validation_result.total_loss,
            best_macro_f1=best_macro_f1,
            best_total_loss=best_validation_loss,
            min_delta=min_delta,
        )
        if improved:
            best_epoch = epoch
            best_macro_f1 = validation_result.macro_f1
            best_validation_loss = validation_result.total_loss
            best_validation_result = validation_result
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
                "train_total_loss": training_metrics.total_loss,
                "train_classification_loss": (
                    training_metrics.classification_loss
                ),
                "train_topology_loss": training_metrics.topology_loss,
                "train_valid_topology_count": (
                    training_metrics.valid_topology_target_count
                ),
                "validation_total_loss": validation_result.total_loss,
                "validation_classification_loss": (
                    validation_result.classification_loss
                ),
                "validation_topology_loss": (
                    validation_result.topology_loss
                ),
                "validation_accuracy": validation_result.accuracy,
                "validation_macro_f1": validation_result.macro_f1,
                "validation_weighted_f1": validation_result.weighted_f1,
                "validation_topology_mae": (
                    validation_result.topology_mae_re
                ),
                "validation_topology_rmse": (
                    validation_result.topology_rmse_re
                ),
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
            "is_best": int(row["epoch"]) == best_epoch,
        }
        for row in history
    )
    return CnnTrainingFitResult(
        best_epoch=best_epoch,
        best_validation_result=best_validation_result,
        history=history,
        epochs_completed=len(history),
        total_seconds=time.perf_counter() - fit_start,
    )


def train_cnn_epoch(
    model,
    loader,
    *,
    optimizer,
    device,
    topology_loss_weight,
    gradient_clip_norm=None,
):
    """Run one optimizer epoch over every training sample.

    This inner stage-6 operation moves each sample-wise batch to the chosen
    device, performs the multitask forward and backward calculations, and
    applies one AdamW update. It mutates both the model parameters and
    optimizer state.

    Parameters
    ----------
    model : VdfCNN
        CNN updated in place.
    loader : torch.utils.data.DataLoader
        Shuffled training partition loader.
    optimizer : torch.optim.Optimizer
        Optimizer for ``model`` parameters.
    device : str or torch.device
        First model-stage device receiving input batches.
    topology_loss_weight : float
        Auxiliary topology-loss multiplier.
    gradient_clip_norm : float, optional
        Maximum global gradient norm across every model stage.

    Returns
    -------
    EpochTrainingMetrics
        Sample- and valid-target-weighted losses.

    Notes
    -----
    Classification loss is aggregated by sample count. Topology loss is
    aggregated by the number of valid scalar targets so missing X- or
    O-point values do not change its weighting.
    """

    device = torch.device(device)
    model.train()
    sample_count = 0
    classification_loss_sum = 0.0
    topology_loss_sum = 0.0
    valid_topology_count = 0

    for batch in loader:
        inputs = batch["vdf_input"].to(
            device,
            non_blocking=device.type == "cuda",
        )
        output_device = model.output_device
        plasma_context = batch["plasma_context"].to(
            output_device,
            non_blocking=output_device.type == "cuda",
        )
        batch_size = int(inputs.shape[0])
        optimizer.zero_grad(set_to_none=True)
        outputs = model(inputs, plasma_context)
        targets = {
            name: batch[name].to(
                output_device,
                non_blocking=output_device.type == "cuda",
            )
            for name in (
                "class_target",
                "topology_targets",
                "topology_mask",
            )
        }
        loss = calculate_cnn_loss(
            outputs,
            targets["class_target"],
            targets["topology_targets"],
            targets["topology_mask"],
            topology_loss_weight=topology_loss_weight,
        )
        loss.total_loss.backward()
        if gradient_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                [
                    parameter
                    for parameter_group in optimizer.param_groups
                    for parameter in parameter_group["params"]
                ],
                max_norm=float(gradient_clip_norm),
            )
        optimizer.step()

        sample_count += batch_size
        valid_count = loss.valid_topology_target_count
        classification_loss_sum += (
            float(loss.classification_loss.detach().cpu()) * batch_size
        )
        topology_loss_sum += (
            float(loss.topology_loss.detach().cpu()) * valid_count
        )
        valid_topology_count += valid_count

    classification_loss = classification_loss_sum / sample_count
    topology_loss = (
        topology_loss_sum / valid_topology_count
        if valid_topology_count
        else 0.0
    )
    return EpochTrainingMetrics(
        total_loss=(
            classification_loss
            + float(topology_loss_weight) * topology_loss
        ),
        classification_loss=classification_loss,
        topology_loss=topology_loss,
        valid_topology_target_count=valid_topology_count,
        sample_count=sample_count,
    )


def is_better_validation_result(
    *,
    macro_f1,
    total_loss,
    best_macro_f1,
    best_total_loss,
    min_delta,
):
    """Compare macro F1 and the deterministic loss tie-breaker.

    Stage 6 uses this pure comparison after every validation pass. A macro
    F1 increase beyond ``min_delta`` wins; values within that tolerance use
    the lower total multitask loss.

    Parameters
    ----------
    macro_f1 : float
        Candidate validation macro F1.
    total_loss : float
        Candidate validation total loss.
    best_macro_f1 : float
        Current selected macro F1.
    best_total_loss : float
        Current selected total loss.
    min_delta : float
        Macro-F1 equality and improvement tolerance.

    Returns
    -------
    bool
        Whether the candidate replaces the selected epoch.
    """

    if float(macro_f1) > float(best_macro_f1) + float(min_delta):
        return True
    return (
        abs(float(macro_f1) - float(best_macro_f1)) <= float(min_delta)
        and float(total_loss) < float(best_total_loss)
    )


def set_training_seed(random_seed):
    """Set Python, NumPy, and PyTorch random generators.

    The training entry points call this before model construction and
    DataLoader creation. Besides seeding all random generators, it enables
    deterministic PyTorch algorithms where implementations are available,
    warns for unsupported operations, and disables cuDNN benchmarking. These
    process-wide side effects make supported model calculations repeatable
    without stopping CUDA training on nondeterministic kernels.

    Parameters
    ----------
    random_seed : int
        Seed shared by model initialization and loader shuffling.
    """

    random_seed = int(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def resolve_training_device(device):
    """Resolve the configured CPU, CUDA, or automatic device.

    Both training workflows call this before creating loaders or moving
    models. ``"auto"`` selects the current CUDA device when available and
    otherwise uses CPU.

    Parameters
    ----------
    device : str
        ``"auto"`` or a PyTorch device string.

    Returns
    -------
    torch.device
        First device used by the training workflow.
    """

    value = str(device).strip().lower()
    if value == "auto":
        value = (
            f"cuda:{torch.cuda.current_device()}"
            if torch.cuda.is_available()
            else "cpu"
        )
    return torch.device(value)


def create_partition_loader(
    data,
    indices,
    topology_scaler,
    plasma_context_scaler,
    config,
    *,
    shuffle,
    device,
):
    """Create one configured timestep-partition loader.

    Stage 0 uses this adapter for training, validation, and final evaluation
    partitions. It binds workflow configuration to the sample-wise loader
    while deriving pinned-memory behavior from the selected device.

    Parameters
    ----------
    data : CnnTrainingData
        Memory-mapped saved representation description and aligned targets.
    indices : sequence of int
        Saved rows in this partition.
    topology_scaler : TopologyTargetScaler
        Training-derived target scaling.
    plasma_context_scaler : PlasmaContextScaler
        Training-derived scaling for the 16-value plasma context.
    config : mapping
        CNN data-loader and random-seed configuration.
    shuffle : bool
        Whether to shuffle traversal order.
    device : torch.device
        Training device controlling automatic pinned memory.

    Returns
    -------
    torch.utils.data.DataLoader
        Sample-wise representation loader.
    """

    pin_memory = config["loader"]["pin_memory"]
    if pin_memory == "auto":
        pin_memory = device.type == "cuda"
    return create_cnn_dataloader(
        data,
        indices=indices,
        topology_scaler=topology_scaler,
        plasma_context_scaler=plasma_context_scaler,
        batch_size=config["loader"]["batch_size"],
        shuffle=shuffle,
        random_seed=config["random_state"],
        num_workers=config["loader"]["num_workers"],
        pin_memory=pin_memory,
    )


def close_loader(loader):
    """Close a DataLoader dataset's representation memmap.

    The stage-0 orchestrator calls this after a partition is no longer
    needed so the process-local file descriptor does not remain open through
    later evaluation or output writing.

    Parameters
    ----------
    loader : torch.utils.data.DataLoader
        Loader whose dataset owns a process-local memmap.
    """

    loader.dataset.close()


__all__ = [
    "CnnTrainingFitResult",
    "EpochTrainingMetrics",
    "close_loader",
    "create_partition_loader",
    "is_better_validation_result",
    "optimize_cnn",
    "resolve_training_device",
    "set_training_seed",
    "train_cnn_epoch",
]
