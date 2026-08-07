"""Stage 2: fit global PyTorch scaling statistics from sample batches.

Stage 1 supplies a repeatable iterator over a memory-mapped representation.
This stage makes the first PCA pass and combines batch population moments
with the Chan/Welford parallel formula. Only one feature batch plus the
running per-feature mean and second moment remain on the selected CPU or CUDA
device.

The result is mathematically equivalent to population-variance
``StandardScaler`` for the same prepared rows, but it never holds the full
sample-by-feature matrix. Physical labels, topology, coordinates, and moments
are absent. Stage 3 reuses the mean and scale while fitting and scoring the
incremental PCA basis.
"""

import torch

from src.analysis.step_01_load_pca_features import (
    iter_pca_feature_batches,
)


def fit_pca_scaling(
    representation_array,
    representation,
    batch_size,
    device,
    log_eps=None,
):
    """Fit one global mean and population scale per PCA feature column.

    Consecutive batches are accumulated in float64 with the parallel
    Chan/Welford update. A zero-variance feature receives scale one, matching
    the established PCA preprocessing while leaving its standardized values
    at zero. No prepared or standardized rows are retained after their batch
    moments have been merged.

    Parameters
    ----------
    representation_array : numpy.ndarray
        Read-only raw ``(sample, vx, vy, vz)`` or physical-VDF Hermite
        ``(sample, order, order, order)`` memory map in its saved unrotated or
        optionally rotated coefficient-axis order.
    representation : {"raw", "hermite"}
        Preparation branch used by the Stage 1 batch iterator.
    batch_size : int
        Number of consecutive saved samples prepared for each statistics
        update.
    device : str or torch.device
        ``cpu``, ``cuda``, ``cuda:N``, or ``auto``. Automatic selection uses
        CUDA when PyTorch reports it available and otherwise uses CPU.
    log_eps : float, optional
        Positive raw-VDF floor applied before ``log10``. Hermite coefficients
        do not use this value.

    Returns
    -------
    feature_mean : torch.Tensor
        Float64 global feature means on the selected device.
    feature_scale : torch.Tensor
        Float64 population standard deviations on the selected device, with
        exact zero scales replaced by one.

    Notes
    -----
    The persistent tensors have shape ``(n_features,)``. For the BCH raw
    grid this is about 147 MiB per float64 vector, rather than the roughly
    844-GiB complete float32 feature matrix reported in production.
    """

    selected_device = _resolve_pca_device(device)
    running_count = 0
    running_mean = None
    running_m2 = None
    with torch.inference_mode():
        for _, _, feature_batch in iter_pca_feature_batches(
            representation_array,
            representation,
            batch_size,
            log_eps,
        ):
            feature_tensor = torch.as_tensor(
                feature_batch,
                dtype=torch.float64,
                device=selected_device,
            )
            batch_count = int(feature_tensor.shape[0])
            batch_variance, batch_mean = torch.var_mean(
                feature_tensor,
                dim=0,
                correction=0,
            )
            batch_m2 = batch_variance * batch_count
            if running_mean is None:
                running_count = batch_count
                running_mean = batch_mean
                running_m2 = batch_m2
                del (
                    feature_batch,
                    feature_tensor,
                    batch_variance,
                    batch_mean,
                    batch_m2,
                )
                continue

            combined_count = running_count + batch_count
            delta = batch_mean - running_mean
            running_mean.add_(delta * (batch_count / combined_count))
            running_m2.add_(
                batch_m2
                + delta.square()
                * (running_count * batch_count / combined_count)
            )
            running_count = combined_count
            del (
                feature_batch,
                feature_tensor,
                batch_variance,
                batch_mean,
                batch_m2,
                delta,
            )

        feature_scale = torch.sqrt(running_m2 / running_count)
        feature_scale = torch.where(
            feature_scale == 0,
            torch.ones_like(feature_scale),
            feature_scale,
        )
    return running_mean, feature_scale


def _resolve_pca_device(device):
    """Resolve the configured PyTorch device without adding fallback logic."""

    if str(device) == "auto":
        return torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    return torch.device(device)
