"""Combine complete-volume reconstruction and auxiliary topology losses.

This focused owner joins the autoencoder's established mean-squared
reconstruction objective with the same globally masked Smooth L1 topology
calculation used by the CNN. It owns no validation, device placement,
optimization, or persistence behavior.
"""

from torch.nn import functional

from src.learning.topology_supervision import calculate_masked_topology_loss


def calculate_autoencoder_loss(
    reconstruction,
    target_inputs,
    topology_predictions,
    topology_targets,
    topology_mask,
    topology_loss_weight,
):
    """Combine full-volume reconstruction and masked topology losses.

    Reconstruction mean-squared error compares complete normalized
    three-dimensional inputs and outputs. The auxiliary Smooth L1 term uses
    only finite topology entries after scaling targets from the training
    partition. A sample with missing topology values therefore remains
    available for reconstruction training while unavailable target entries
    contribute no topology error.

    Parameters
    ----------
    reconstruction : torch.Tensor
        Complete normalized raw or Hermite reconstruction with shape
        ``(batch_size, 1, depth_1, depth_2, depth_3)``.
    target_inputs : torch.Tensor
        Complete normalized reconstruction target with the same shape.
    topology_predictions : torch.Tensor
        Scaled topology predictions with shape ``(batch_size, 6)``.
    topology_targets : torch.Tensor
        Training-scaled topology targets in the fixed six-value project
        order.
    topology_mask : torch.Tensor
        Boolean finite-value mask with shape ``(batch_size, 6)``.
    topology_loss_weight : float
        Multiplier applied to the globally masked topology loss.

    Returns
    -------
    dict
        Scalar ``total_loss``, ``reconstruction_loss``, ``topology_loss``,
        and integer ``valid_topology_count``.

    Notes
    -----
    The exact objective is ``reconstruction_loss + topology_loss_weight *
    topology_loss``. Topology targets are never model inputs. Both objectives
    update the shared encoder through the non-detached latent representation.
    Target order is X distance, O distance, X-point ``dx``, X-point ``dz``,
    O-point ``dx``, and O-point ``dz``; physical metadata use Earth radii and
    the loss receives their training-standardized values.
    """

    reconstruction_loss = functional.mse_loss(
        reconstruction,
        target_inputs,
    )
    topology_loss, valid_count = calculate_masked_topology_loss(
        topology_predictions,
        topology_targets,
        topology_mask,
    )
    total_loss = (
        reconstruction_loss
        + float(topology_loss_weight) * topology_loss
    )
    return {
        "total_loss": total_loss,
        "reconstruction_loss": reconstruction_loss,
        "topology_loss": topology_loss,
        "valid_topology_count": valid_count,
    }


__all__ = ["calculate_autoencoder_loss"]
