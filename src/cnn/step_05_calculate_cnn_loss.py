"""Stage 5: calculate the CNN multitask training loss.

This stage follows model construction and is called by optimization and
evaluation. It combines unweighted physical-class cross entropy with masked
Smooth L1 topology regression and returns the components needed for epoch
aggregation.

The stage receives one forward-pass output dictionary, contiguous class
targets, scaled topology targets, and their availability mask. It returns
scalar differentiable loss tensors plus the valid-target count used by
stages 6 and 7.
"""

from dataclasses import dataclass

import torch
from torch.nn import functional


@dataclass(frozen=True)
class CnnLossResult:
    """Hold classification and masked topology loss components.

    Each call to :func:`calculate_cnn_loss` creates this immutable batch
    record. Optimization backpropagates ``total_loss``; training and
    evaluation aggregate the separate components with their appropriate
    sample or valid-target weights.

    Parameters
    ----------
    total_loss : torch.Tensor
        Classification loss plus weighted topology loss.
    classification_loss : torch.Tensor
        Unweighted physical-class cross entropy.
    topology_loss : torch.Tensor
        Smooth L1 loss averaged over valid topology scalars.
    valid_topology_target_count : int
        Number of valid scalar topology targets in the batch.
    """

    total_loss: torch.Tensor
    classification_loss: torch.Tensor
    topology_loss: torch.Tensor
    valid_topology_target_count: int


def calculate_cnn_loss(
    outputs,
    class_targets,
    topology_targets=None,
    topology_mask=None,
    *,
    topology_loss_weight,
):
    """Calculate class cross entropy plus masked topology regression.

    Stage 6 calls this for optimizer batches and stage 7 calls it for
    evaluation batches. Cross entropy treats every sample equally. Smooth
    L1 regression averages only the topology scalars selected by the mask,
    then the configured weight combines it with classification.

    Parameters
    ----------
    outputs : mapping
        Model ``class_logits`` with shape ``(batch, n_classes)`` and scaled
        ``topology_predictions`` with shape ``(batch, 6)``.
    class_targets : torch.Tensor
        Contiguous output indices with shape ``(batch,)``.
    topology_targets : torch.Tensor, optional
        Scaled topology values with shape ``(batch, 6)`` in
        ``TOPOLOGY_TARGET_COLUMNS`` order.
    topology_mask : torch.Tensor, optional
        Missing-target mask with shape ``(batch, 6)``.
    topology_loss_weight : float
        Multiplier for the topology loss.

    Returns
    -------
    CnnLossResult
        Scalar total, classification, and topology tensors plus the number
        of valid topology scalars.

    Notes
    -----
    When topology targets are omitted or every mask entry is false, the
    topology component remains a differentiable zero connected to the
    topology head.
    """

    class_logits = outputs["class_logits"]
    topology_predictions = outputs["topology_predictions"]
    classification_loss = functional.cross_entropy(
        class_logits,
        class_targets,
    )
    if topology_targets is None:
        valid_count = 0
        topology_loss = topology_predictions.sum() * 0.0
    else:
        valid_count = int(torch.count_nonzero(topology_mask).item())
        if valid_count:
            topology_loss = functional.smooth_l1_loss(
                topology_predictions[topology_mask],
                topology_targets[topology_mask],
                reduction="sum",
            ) / valid_count
        else:
            topology_loss = topology_predictions.sum() * 0.0
    total_loss = (
        classification_loss
        + float(topology_loss_weight) * topology_loss
    )
    return CnnLossResult(
        total_loss=total_loss,
        classification_loss=classification_loss,
        topology_loss=topology_loss,
        valid_topology_target_count=valid_count,
    )


__all__ = ["CnnLossResult", "calculate_cnn_loss"]
