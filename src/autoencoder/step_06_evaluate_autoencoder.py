"""Stage 6: evaluate and summarize reconstruction error.

This stage is called by optimization for validation selection and after
training for final reporting. It calculates normalized-representation-space
mean squared error for each sample and the complete partition without
changing the autoencoder objective. Final sample losses are associated with
physical classes only after inference, then summarized for the text report.

The stage receives the selected model, one stable partition loader, a
device, and partition name. It performs inference without gradients and
keeps sample-level values only in memory until split and class tables have
been calculated.
"""

import numpy as np
import pandas as pd
import torch


def evaluate_autoencoder(model, loader, device, split_name):
    """Calculate reconstruction MSE for one timestep partition.

    Stage 5 calls this after each optimizer epoch for selection, and stage 0
    calls it for final train, validation, and test metrics. The function
    switches the model to evaluation mode and leaves it there after
    accumulating inference-only metrics.

    Parameters
    ----------
    model : VdfAutoencoder
        Raw or Hermite reconstruction model.
    loader : torch.utils.data.DataLoader
        Stable partition loader yielding unnormalized inputs.
    device : str or torch.device
        Device hosting the model and inference batches.
    split_name : str
        Partition label saved in the result.

    Returns
    -------
    dict
        ``split``, sample count, aggregate mean squared error, aligned saved
        sample indices, and one normalized-space MSE value per sample.

    Notes
    -----
    Physical class labels are not present in the loader and do not enter
    reconstruction. Stage 7 introduces them only after this label-blind
    inference pass has produced sample-aligned errors.
    """

    device = torch.device(device)
    model.eval()
    squared_error = 0.0
    value_count = 0
    sample_count = 0
    sample_indices = []
    sample_mse = []
    with torch.no_grad():
        for batch in loader:
            inputs = batch["inputs"].to(device)
            normalized = model.normalize_inputs(inputs)
            output = model._forward_normalized(normalized)
            difference = output["reconstruction"] - normalized
            squared_difference = torch.square(
                difference.detach().to(dtype=torch.float64)
            )
            squared_error += float(torch.sum(squared_difference))
            value_count += normalized.numel()
            sample_count += len(inputs)
            sample_indices.append(
                batch["sample_index"].detach().cpu().numpy()
            )
            sample_mse.append(
                torch.mean(
                    squared_difference,
                    dim=tuple(range(1, squared_difference.ndim)),
                ).cpu().numpy()
            )
    return {
        "split": str(split_name),
        "sample_count": int(sample_count),
        "mse": squared_error / value_count,
        "sample_indices": np.concatenate(sample_indices),
        "sample_mse": np.concatenate(sample_mse),
    }


def create_autoencoder_reconstruction_tables(
    evaluations,
    sample_identity,
):
    """Summarize final reconstruction error by split and physical class.

    This reporting operation follows final inference with the restored
    validation-selected autoencoder. It joins sample-aligned normalized-space
    MSE values to current physical class identity, then calculates count,
    mean, median, minimum, and maximum error tables for ``metrics.txt``.

    Parameters
    ----------
    evaluations : mapping
        Final train, validation, and test dictionaries returned by
        :func:`evaluate_autoencoder`. Each dictionary contains aligned
        ``sample_indices`` and ``sample_mse`` arrays.
    sample_identity : pandas.DataFrame
        Saved sample identity containing ``sample_index``, ``class_id``, and
        ``class_name``. Class identity is reporting-only and was not yielded
        during model inference.

    Returns
    -------
    split_metrics : pandas.DataFrame
        Reconstruction statistics in stable train, validation, and test
        order with columns ``split``, ``count``, ``mean_mse``,
        ``median_mse``, ``min_mse``, and ``max_mse``.
    class_metrics : pandas.DataFrame
        The same statistics for each current physical class, ordered by its
        saved numeric class ID and containing no sample-level rows.

    Notes
    -----
    Physical classes are introduced only for this aggregation. They do not
    alter reconstruction, loss, optimizer updates, early stopping, or model
    selection.
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


__all__ = [
    "create_autoencoder_reconstruction_tables",
    "evaluate_autoencoder",
]
