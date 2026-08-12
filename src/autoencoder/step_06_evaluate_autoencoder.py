"""Stage 6: evaluate reconstruction and auxiliary topology objectives.

The restored validation-total-loss-selected model is evaluated on train,
validation, and test partitions without changing their chronological row
order. Complete normalized three-dimensional inputs determine reconstruction
MSE. Finite topology entries determine globally masked scaled Smooth L1 and
per-target MAE/RMSE after inverse conversion to Earth radii.

Physical class identity is introduced only after inference to retain the
existing reconstruction report. Plasma context is part of the current model
input; neither class nor topology is a model input.
"""

import numpy as np
import pandas as pd
import torch

from src.autoencoder.autoencoder_loss import calculate_autoencoder_loss


def evaluate_autoencoder(
    model,
    loader,
    topology_scaler,
    split_name,
    topology_loss_weight,
):
    """Evaluate one complete-timestep partition with the combined objective.

    Each requested batch is normalized once and reconstructed in full. The six
    scaled topology predictions are compared only where metadata supplied a
    finite target. Predictions and valid targets are inverse transformed with
    training-only scaling before physical errors are accumulated in Earth
    radii.

    Parameters
    ----------
    model : VdfAutoencoder
        Placed full-volume raw or Hermite multitask autoencoder.
    loader : torch.utils.data.DataLoader
        Stable partition yielding complete inputs and aligned topology data.
    topology_scaler : TopologyTargetScaler
        Six-target scaler fitted from valid training entries only.
    split_name : str
        Partition label retained in aggregate tables.
    topology_loss_weight : float
        Multiplier in reconstruction MSE plus masked topology Smooth L1.

    Returns
    -------
    dict
        Split count, reconstruction MSE, scaled topology Smooth L1, total
        objective, aligned sample reconstruction MSE, and per-target physical
        MAE/RMSE tables. No per-sample topology predictions are retained.

    Notes
    -----
    All returned objective values aggregate scalar errors across the complete
    partition. The model receives the VDF representation and aligned
    training-scaled 16-value plasma context. Topology targets and masks move
    to :attr:`VdfAutoencoder.output_device` solely for loss calculation.
    """

    model.eval()
    reconstruction_error = 0.0
    reconstruction_value_count = 0
    topology_error = 0.0
    valid_topology_count = 0
    sample_count = 0
    sample_indices = []
    sample_mse = []
    target_count = np.zeros(len(topology_scaler.target_names), dtype=np.int64)
    target_absolute_error = np.zeros_like(target_count, dtype=np.float64)
    target_squared_error = np.zeros_like(target_count, dtype=np.float64)
    with torch.no_grad():
        for batch in loader:
            normalized = model.normalize_inputs(batch["vdf_input"])
            output = model.forward_from_normalized(
                normalized,
                batch["plasma_context"],
            )
            target_inputs = normalized.to(model.output_device)
            topology_targets = batch["topology_targets"].to(
                model.output_device
            )
            topology_mask = batch["topology_mask"].to(model.output_device)
            losses = calculate_autoencoder_loss(
                output["reconstruction"],
                target_inputs,
                output["topology_predictions"],
                topology_targets,
                topology_mask,
                topology_loss_weight,
            )

            difference = output["reconstruction"] - target_inputs
            squared_difference = torch.square(
                difference.detach().to(dtype=torch.float64)
            )
            reconstruction_error += float(torch.sum(squared_difference))
            reconstruction_value_count += target_inputs.numel()
            batch_valid_count = losses["valid_topology_count"]
            topology_error += (
                float(losses["topology_loss"].detach()) * batch_valid_count
            )
            valid_topology_count += batch_valid_count
            sample_count += len(target_inputs)
            sample_indices.append(batch["sample_index"].cpu().numpy())
            sample_mse.append(
                torch.mean(
                    squared_difference,
                    dim=tuple(range(1, squared_difference.ndim)),
                ).cpu().numpy()
            )

            physical_predictions = topology_scaler.inverse_transform(
                output["topology_predictions"].detach()
            ).cpu().numpy()
            physical_targets = topology_scaler.inverse_transform(
                topology_targets.detach(),
                topology_mask,
            ).cpu().numpy()
            mask = topology_mask.cpu().numpy()
            physical_difference = physical_predictions - physical_targets
            for target_index in range(len(topology_scaler.target_names)):
                valid = mask[:, target_index]
                values = physical_difference[valid, target_index]
                target_count[target_index] += len(values)
                target_absolute_error[target_index] += np.sum(
                    np.abs(values),
                    dtype=np.float64,
                )
                target_squared_error[target_index] += np.sum(
                    values * values,
                    dtype=np.float64,
                )

    reconstruction_mse = reconstruction_error / reconstruction_value_count
    topology_loss = (
        topology_error / valid_topology_count
        if valid_topology_count
        else 0.0
    )
    total_loss = (
        reconstruction_mse
        + float(topology_loss_weight) * topology_loss
    )
    mean_absolute_error = np.full_like(
        target_absolute_error,
        np.nan,
    )
    root_mean_squared_error = np.full_like(
        target_squared_error,
        np.nan,
    )
    np.divide(
        target_absolute_error,
        target_count,
        out=mean_absolute_error,
        where=target_count > 0,
    )
    np.sqrt(
        np.divide(
            target_squared_error,
            target_count,
            out=root_mean_squared_error,
            where=target_count > 0,
        ),
        out=root_mean_squared_error,
    )
    topology_metrics = pd.DataFrame(
        {
            "target": topology_scaler.target_names,
            "valid_count": target_count,
            "mae_re": mean_absolute_error,
            "rmse_re": root_mean_squared_error,
        }
    )
    return {
        "split": str(split_name),
        "sample_count": int(sample_count),
        "reconstruction_mse": reconstruction_mse,
        "topology_loss": topology_loss,
        "total_loss": total_loss,
        "sample_indices": np.concatenate(sample_indices),
        "sample_mse": np.concatenate(sample_mse),
        "topology_metrics": topology_metrics,
    }


def create_autoencoder_reconstruction_tables(evaluations, sample_identity):
    """Summarize complete-volume reconstruction error by split and class.

    Sample-aligned reconstruction errors are joined to reporting-only class
    identity after inference. Stable grouping retains the existing split and
    physical-class summaries without affecting the trained objectives.

    Parameters
    ----------
    evaluations : mapping
        Final split results with aligned sample indices and complete-volume
        normalized reconstruction MSE.
    sample_identity : pandas.DataFrame
        Saved sample and physical class identity. Classes are reporting-only.

    Returns
    -------
    split_metrics : pandas.DataFrame
        Count, mean, median, minimum, and maximum sample MSE by split.
    class_metrics : pandas.DataFrame
        The same reconstruction-only statistics by physical class.

    Notes
    -----
    Class identity never affects reconstruction loss, topology loss,
    optimization, or best-epoch selection.
    """

    sample_metrics = pd.concat(
        [
            pd.DataFrame(
                {
                    "split": result["split"],
                    "sample_index": result["sample_indices"],
                    "reconstruction_mse": result["sample_mse"],
                }
            )
            for result in evaluations.values()
        ],
        ignore_index=True,
    )
    sample_metrics = sample_metrics.merge(
        sample_identity.loc[
            :,
            ["sample_index", "class_id", "class_name"],
        ],
        on="sample_index",
        how="left",
        sort=False,
    )
    aggregations = {
        "count": "count",
        "mean_mse": "mean",
        "median_mse": "median",
        "min_mse": "min",
        "max_mse": "max",
    }
    split_metrics = (
        sample_metrics.groupby("split", sort=False)["reconstruction_mse"]
        .agg(**aggregations)
        .reset_index()
    )
    class_metrics = (
        sample_metrics.groupby(
            ["class_id", "class_name"],
            sort=False,
        )["reconstruction_mse"]
        .agg(**aggregations)
        .reset_index()
        .sort_values("class_id", kind="stable")
        .drop(columns="class_id")
        .reset_index(drop=True)
    )
    return split_metrics, class_metrics


def create_autoencoder_objective_tables(evaluations):
    """Create split objectives and physical topology-error tables.

    The compact tables flatten final in-memory split aggregates into the two
    human-readable sections written to ``metrics.txt``. No sample prediction
    file or topology sidecar is created.

    Parameters
    ----------
    evaluations : mapping
        Ordered final train, validation, and test evaluation results.

    Returns
    -------
    objective_metrics : pandas.DataFrame
        Split, sample count, complete-volume reconstruction MSE, scaled
        topology Smooth L1, and weighted total objective.
    topology_metrics : pandas.DataFrame
        Split, fixed target name, valid count, MAE, and RMSE in Earth radii.
    """

    objective_metrics = pd.DataFrame(
        [
            {
                "split": result["split"],
                "count": result["sample_count"],
                "reconstruction_mse": result["reconstruction_mse"],
                "topology_smooth_l1": result["topology_loss"],
                "total_loss": result["total_loss"],
            }
            for result in evaluations.values()
        ]
    )
    topology_metrics = pd.concat(
        [
            result["topology_metrics"].assign(split=result["split"])
            for result in evaluations.values()
        ],
        ignore_index=True,
    ).loc[
        :,
        ["split", "target", "valid_count", "mae_re", "rmse_re"],
    ]
    return objective_metrics, topology_metrics


__all__ = [
    "create_autoencoder_objective_tables",
    "create_autoencoder_reconstruction_tables",
    "evaluate_autoencoder",
]
