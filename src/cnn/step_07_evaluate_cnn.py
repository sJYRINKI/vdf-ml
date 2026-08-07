"""Stage 7: evaluate CNN partitions in physical units.

This stage is called during optimization for validation selection and after
optimization for final train, validation, and test metrics. It follows loss
calculation, restores topology predictions to Earth radii, and returns the
metrics consumed by the final saving stage.

The stage receives an evaluation-mode model, one stable partition loader,
the training-derived topology scaler, and the class mapping. It returns
aggregate class metrics, a fixed-order confusion matrix, and masked
topology errors for stage 6 selection or stage 8 saving.
"""

from dataclasses import dataclass
import math

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from src.cnn.step_05_calculate_cnn_loss import calculate_cnn_loss
from src.data.metadata_columns import TOPOLOGY_TARGET_COLUMNS


@dataclass(frozen=True)
class EvaluationResult:
    """Hold aggregate CNN metrics for one timestep partition.

    Stage 7 creates one immutable result for training, validation, or test
    samples. Stage 6 reads validation macro F1 and total loss for epoch
    selection; stage 8 serializes the final aggregate fields after the
    selected model has been restored.

    Parameters
    ----------
    split_name : str
        Evaluated partition name.
    sample_count : int
        Number of evaluated samples.
    total_loss : float
        Classification loss plus weighted topology loss.
    classification_loss : float
        Sample-weighted cross entropy.
    topology_loss : float
        Valid-target-weighted scaled Smooth L1 loss.
    accuracy : float
        Physical-class accuracy.
    macro_f1 : float
        Macro F1 in explicit physical-class order.
    weighted_f1 : float
        Support-weighted F1.
    classification_report_text : str or None
        Precision, recall, F1, and support table in physical-class order.
        Repeated epoch validation omits the formatted table.
    confusion_matrix : numpy.ndarray
        Integer matrix with shape ``(n_classes, n_classes)``; both axes use
        ``ClassMapping.class_ids`` order, with true classes on rows.
    topology_valid_target_count : int
        Number of valid topology scalars.
    topology_mae_re : float or None
        Aggregate mean absolute topology error in Earth radii.
    topology_rmse_re : float or None
        Aggregate root-mean-square topology error in Earth radii.
    topology_metrics : tuple of dict
        Per-target error records in schema order.
    """

    split_name: str
    sample_count: int
    total_loss: float
    classification_loss: float
    topology_loss: float
    accuracy: float
    macro_f1: float
    weighted_f1: float
    classification_report_text: str | None
    confusion_matrix: np.ndarray
    topology_valid_target_count: int
    topology_mae_re: float | None
    topology_rmse_re: float | None
    topology_metrics: tuple


def evaluate_cnn(
    model,
    loader,
    topology_scaler,
    class_mapping,
    device,
    topology_loss_weight,
    split_name,
    include_classification_report=False,
):
    """Evaluate one CNN partition in bounded batches.

    The function runs inference without gradients, aggregates multitask
    losses using the same weighting as training, decodes logit columns to
    project class IDs, and restores topology values to Earth radii. It
    preserves the model's incoming training/evaluation mode.

    Parameters
    ----------
    model : VdfCNN
        Raw or Hermite multitask model.
    loader : torch.utils.data.DataLoader
        Stable partition loader.
    topology_scaler : TopologyTargetScaler
        Training-derived scaler used for physical-unit errors.
    class_mapping : ClassMapping
        Physical classes in model-output order.
    device : str or torch.device
        First model-stage device receiving input batches.
    topology_loss_weight : float
        Auxiliary topology-loss multiplier.
    split_name : str
        Partition label saved with the result.
    include_classification_report : bool, optional
        Format the physical-class precision, recall, F1, and support table.
        Final partition evaluation enables this; per-epoch validation keeps
        it disabled to avoid repeatedly formatting a report that is not
        used for model selection.

    Returns
    -------
    EvaluationResult
        Aggregate losses, class metrics, and physical topology errors.

    Notes
    -----
    Class probabilities are used only for ``argmax`` selection. Topology
    metrics include only target entries marked available in metadata.
    """

    device = torch.device(device)
    sample_count = 0
    classification_loss_sum = 0.0
    topology_loss_sum = 0.0
    topology_valid_count = 0
    true_project_ids = []
    predicted_project_ids = []
    physical_targets = []
    physical_predictions = []
    topology_masks = []

    was_training = model.training
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            inputs = batch["inputs"].to(
                device,
                non_blocking=device.type == "cuda",
            )
            outputs = model(inputs)
            output_device = outputs["class_logits"].device
            tensors = {
                name: batch[name].to(
                    output_device,
                    non_blocking=output_device.type == "cuda",
                )
                for name in (
                    "class_targets",
                    "topology_targets",
                    "topology_mask",
                )
            }
            loss = calculate_cnn_loss(
                outputs,
                tensors["class_targets"],
                tensors["topology_targets"],
                tensors["topology_mask"],
                topology_loss_weight=topology_loss_weight,
            )
            probabilities = torch.softmax(outputs["class_logits"], dim=1)
            predicted_indices = torch.argmax(probabilities, dim=1)
            true_project_ids.append(
                class_mapping.decode(
                    tensors["class_targets"].detach().cpu().numpy()
                )
            )
            predicted_project_ids.append(
                class_mapping.decode(
                    predicted_indices.detach().cpu().numpy()
                )
            )
            physical_targets.append(
                topology_scaler.inverse_transform(
                    tensors["topology_targets"],
                    tensors["topology_mask"],
                ).detach().cpu().numpy()
            )
            physical_predictions.append(
                topology_scaler.inverse_transform(
                    outputs["topology_predictions"]
                ).detach().cpu().numpy()
            )
            topology_masks.append(
                tensors["topology_mask"].detach().cpu().numpy()
            )

            batch_size = int(inputs.shape[0])
            valid_count = loss.valid_topology_target_count
            sample_count += batch_size
            classification_loss_sum += (
                float(loss.classification_loss.detach().cpu()) * batch_size
            )
            topology_loss_sum += (
                float(loss.topology_loss.detach().cpu()) * valid_count
            )
            topology_valid_count += valid_count
    if was_training:
        model.train()

    true_project_ids = np.concatenate(true_project_ids)
    predicted_project_ids = np.concatenate(predicted_project_ids)
    physical_targets = np.concatenate(physical_targets)
    physical_predictions = np.concatenate(physical_predictions)
    topology_masks = np.concatenate(topology_masks)
    classification_loss = classification_loss_sum / sample_count
    topology_loss = (
        topology_loss_sum / topology_valid_count
        if topology_valid_count
        else 0.0
    )
    class_ids = np.asarray(class_mapping.class_ids, dtype=np.int64)
    topology = calculate_topology_metrics(
        physical_targets,
        physical_predictions,
        topology_masks,
    )
    return EvaluationResult(
        split_name=str(split_name),
        sample_count=sample_count,
        total_loss=(
            classification_loss
            + float(topology_loss_weight) * topology_loss
        ),
        classification_loss=classification_loss,
        topology_loss=topology_loss,
        accuracy=float(
            accuracy_score(true_project_ids, predicted_project_ids)
        ),
        macro_f1=float(
            f1_score(
                true_project_ids,
                predicted_project_ids,
                labels=class_ids,
                average="macro",
                zero_division=0,
            )
        ),
        weighted_f1=float(
            f1_score(
                true_project_ids,
                predicted_project_ids,
                labels=class_ids,
                average="weighted",
                zero_division=0,
            )
        ),
        classification_report_text=(
            classification_report(
                true_project_ids,
                predicted_project_ids,
                labels=class_ids,
                target_names=class_mapping.class_names,
                zero_division=0,
            )
            if include_classification_report
            else None
        ),
        confusion_matrix=confusion_matrix(
            true_project_ids,
            predicted_project_ids,
            labels=class_ids,
        ),
        topology_valid_target_count=topology_valid_count,
        topology_mae_re=topology["mae"],
        topology_rmse_re=topology["rmse"],
        topology_metrics=topology["per_target"],
    )


def calculate_topology_metrics(targets, predictions, mask):
    """Calculate masked topology errors in physical Earth radii.

    Stage 7 calls this after inverse scaling. It calculates one MAE/RMSE
    record per topology column and aggregate errors weighted over all valid
    scalar targets.

    Parameters
    ----------
    targets : numpy.ndarray
        Physical targets with shape ``(n_samples, 6)``.
    predictions : numpy.ndarray
        Physical predictions with the same shape.
    mask : numpy.ndarray
        Missing-target mask with the same shape.

    Returns
    -------
    dict
        ``mae`` and ``rmse`` in Earth radii plus ``per_target`` records in
        ``TOPOLOGY_TARGET_COLUMNS`` order.
    """

    per_target = []
    absolute_error_sum = 0.0
    squared_error_sum = 0.0
    total_valid_count = 0
    for target_index, target_name in enumerate(TOPOLOGY_TARGET_COLUMNS):
        valid = mask[:, target_index]
        valid_count = int(np.count_nonzero(valid))
        if valid_count:
            errors = (
                predictions[valid, target_index]
                - targets[valid, target_index]
            ).astype(np.float64, copy=False)
            target_absolute_sum = float(np.sum(np.abs(errors)))
            target_squared_sum = float(np.sum(np.square(errors)))
            mae = target_absolute_sum / valid_count
            rmse = math.sqrt(target_squared_sum / valid_count)
            absolute_error_sum += target_absolute_sum
            squared_error_sum += target_squared_sum
            total_valid_count += valid_count
        else:
            mae = None
            rmse = None
        per_target.append(
            {
                "target_name": target_name,
                "units": "R_E",
                "valid_count": valid_count,
                "mae": mae,
                "rmse": rmse,
            }
        )
    return {
        "mae": (
            absolute_error_sum / total_valid_count
            if total_valid_count
            else None
        ),
        "rmse": (
            math.sqrt(squared_error_sum / total_valid_count)
            if total_valid_count
            else None
        ),
        "per_target": tuple(per_target),
    }


__all__ = [
    "EvaluationResult",
    "calculate_topology_metrics",
    "evaluate_cnn",
]
