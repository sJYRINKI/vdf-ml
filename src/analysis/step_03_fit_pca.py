"""Stage 3: fit and score incremental PyTorch PCA from sample batches.

The reported production raw matrix has shape ``(11767, 19248832)`` and
would require about 844 GiB as float32 before standardization or
decomposition. This stage therefore makes two repeatable passes over Stage
1 batches after the statistics pass in Stage 2. The first pass fits a
retained component basis; the second projects every sample into the small
``(n_samples, n_components)`` score array.

``full`` uses exact reduced ``torch.linalg.svd`` at each incremental update,
while ``lowrank`` uses seeded ``torch.pca_lowrank`` at each update. Both are
incremental retained-rank algorithms when more than one update is required.
They hold only the current standardized batch, retained component rows, and
one update matrix. Component vectors still span every raw VDF voxel or every
Hermite coefficient, and labels never enter fitting.
"""

import numpy as np
import torch

from src.analysis.step_01_load_pca_features import (
    get_pca_feature_dtype,
    get_pca_feature_shape,
    iter_pca_feature_batches,
)


def fit_pca(
    representation_array,
    representation,
    feature_mean,
    feature_scale,
    batch_size,
    config,
    random_state,
    log_eps=None,
):
    """Fit global PCA components from memory-mapped sample batches.

    Prepared batches are standardized with Stage 2's global feature mean and
    population scale. Full mode performs an exact reduced SVD of each update
    matrix and retains ``n_components`` rows. Low-rank mode seeds PyTorch once,
    retains ``n_components + lowrank_oversampling`` rows between randomized
    updates, and exposes only the requested score columns.

    Initial batches are accumulated until they contain the working rank. Each
    later update stacks the previous ``S * Vh`` basis rows above one new
    standardized sample batch. With one update, full mode is equivalent to
    the prior monolithic reduced SVD. With multiple updates, both modes are
    incremental truncated decompositions whose results can depend on batch
    size and row order.

    Parameters
    ----------
    representation_array : numpy.ndarray
        Read-only raw ``(sample, vx, vy, vz)`` or physical-VDF Hermite
        ``(sample, order, order, order)`` memory map in its saved unrotated or
        optionally rotated coefficient-axis order.
    representation : {"raw", "hermite"}
        Feature preparation branch used for both fitting and scoring passes.
    feature_mean : torch.Tensor
        Float64 global per-feature means from Stage 2 on the selected device.
    feature_scale : torch.Tensor
        Float64 global population scales from Stage 2 on the selected device.
    batch_size : int
        Number of consecutive saved samples prepared for every PCA update and
        score projection.
    config : dict
        Settings containing ``method``, ``n_components``,
        ``lowrank_oversampling``, and ``lowrank_niter``.
    random_state : int
        PyTorch seed set once before the low-rank update loop.
    log_eps : float, optional
        Positive raw-VDF floor applied before ``log10``. Hermite coefficients
        do not use this value.

    Returns
    -------
    scores : numpy.ndarray
        Deterministically oriented CPU score array with shape
        ``(n_samples, n_components)`` in saved sample order.
    explained_variance_ratio : numpy.ndarray
        Retained sketch singular-value variance divided by total standardized
        feature energy accumulated over every fitting batch.

    Notes
    -----
    No feature, standardized-feature, transposed, or component cache is
    written. The score output is the only allocation whose first dimension is
    the complete sample count. KMeans and t-SNE consume that small CPU array
    after this stage.
    """

    sample_count, feature_count = get_pca_feature_shape(
        representation_array
    )
    feature_dtype = get_pca_feature_dtype(
        representation_array,
        representation,
    )
    torch_dtype = torch.as_tensor(feature_dtype.type(0)).dtype
    fit_mean = feature_mean.to(dtype=torch_dtype)
    fit_scale = feature_scale.to(dtype=torch_dtype)
    n_components = int(config["n_components"])
    working_rank = n_components
    if config["method"] == "lowrank":
        working_rank += int(config["lowrank_oversampling"])
    working_rank = min(working_rank, sample_count, feature_count)

    with torch.inference_mode():
        if config["method"] == "lowrank":
            torch.manual_seed(random_state)
        components, singular_values, total_squared_sum = (
            _fit_incremental_pca_basis(
                representation_array=representation_array,
                representation=representation,
                feature_mean=fit_mean,
                feature_scale=fit_scale,
                batch_size=batch_size,
                config=config,
                working_rank=working_rank,
                feature_count=feature_count,
                log_eps=log_eps,
            )
        )
        _orient_component_rows(components, n_components)
        scores = _calculate_pca_scores(
            representation_array=representation_array,
            representation=representation,
            feature_mean=fit_mean,
            feature_scale=fit_scale,
            components=components[:n_components],
            batch_size=batch_size,
            feature_dtype=feature_dtype,
            log_eps=log_eps,
        )
        explained_variance_ratio = _calculate_explained_variance_ratio(
            singular_values,
            total_squared_sum,
            sample_count,
            n_components,
        )
    return scores, explained_variance_ratio.detach().cpu().numpy()


def _fit_incremental_pca_basis(
    *,
    representation_array,
    representation,
    feature_mean,
    feature_scale,
    batch_size,
    config,
    working_rank,
    feature_count,
    log_eps,
):
    """Retain a bounded component sketch while visiting every sample once."""

    sample_count = int(representation_array.shape[0])
    batch_size = int(batch_size)
    initial_rows = min(
        sample_count,
        ((working_rank + batch_size - 1) // batch_size) * batch_size,
    )
    first_standardized_rows = None
    first_rows_written = 0
    components = None
    singular_values = None
    total_squared_sum = torch.zeros(
        (),
        dtype=feature_mean.dtype,
        device=feature_mean.device,
    )

    for _, _, feature_batch in iter_pca_feature_batches(
        representation_array,
        representation,
        batch_size,
        log_eps,
    ):
        standardized_batch = _standardize_feature_batch(
            feature_batch,
            feature_mean,
            feature_scale,
        )
        total_squared_sum.add_(
            torch.linalg.vector_norm(standardized_batch).square()
        )
        if components is None:
            if first_standardized_rows is None:
                first_standardized_rows = torch.empty(
                    (initial_rows, feature_count),
                    dtype=standardized_batch.dtype,
                    device=standardized_batch.device,
                )
            next_row = first_rows_written + standardized_batch.shape[0]
            first_standardized_rows[
                first_rows_written:next_row
            ].copy_(standardized_batch)
            first_rows_written = next_row
            if first_rows_written < initial_rows:
                del feature_batch, standardized_batch
                continue
            update_matrix = first_standardized_rows
            first_standardized_rows = None
        else:
            previous_basis_rows = (
                singular_values[:, None] * components
            )
            update_matrix = torch.cat(
                (previous_basis_rows, standardized_batch),
                dim=0,
            )
            del previous_basis_rows

        if config["method"] == "full":
            singular_values, components = _update_full_basis(
                update_matrix,
                int(config["n_components"]),
            )
        else:
            singular_values, components = _update_lowrank_basis(
                update_matrix,
                working_rank,
                int(config["lowrank_niter"]),
            )
        del feature_batch, update_matrix, standardized_batch

    return components, singular_values, total_squared_sum


def _standardize_feature_batch(feature_batch, feature_mean, feature_scale):
    """Move one owned feature batch to the device and standardize in place."""

    feature_tensor = torch.as_tensor(
        feature_batch,
        device=feature_mean.device,
    )
    feature_tensor.sub_(feature_mean)
    feature_tensor.div_(feature_scale)
    return feature_tensor


def _update_full_basis(update_matrix, n_components):
    """Apply exact reduced SVD to one incremental full-method update."""

    _, singular_values, right_vectors_h = torch.linalg.svd(
        update_matrix,
        full_matrices=False,
    )
    retained_values = singular_values[:n_components].clone()
    retained_components = right_vectors_h[:n_components].clone()
    return retained_values, retained_components


def _update_lowrank_basis(update_matrix, working_rank, niter):
    """Apply randomized low-rank decomposition to one incremental update."""

    q = min(working_rank, update_matrix.shape[0], update_matrix.shape[1])
    _, singular_values, right_vectors = torch.pca_lowrank(
        update_matrix,
        q=q,
        center=False,
        niter=niter,
    )
    retained_values = singular_values[:working_rank].clone()
    retained_components = right_vectors[
        :, :working_rank
    ].transpose(0, 1)
    return retained_values, retained_components


def _orient_component_rows(components, n_components):
    """Make each retained row's largest absolute loading nonnegative."""

    for component_index in range(n_components):
        component = components[component_index]
        loading_index = torch.argmax(torch.abs(component))
        sign = torch.sign(component[loading_index])
        sign = torch.where(sign == 0, torch.ones_like(sign), sign)
        component.mul_(sign)


def _calculate_pca_scores(
    *,
    representation_array,
    representation,
    feature_mean,
    feature_scale,
    components,
    batch_size,
    feature_dtype,
    log_eps,
):
    """Project standardized batches and copy only their small CPU scores."""

    scores = np.empty(
        (int(representation_array.shape[0]), int(components.shape[0])),
        dtype=feature_dtype,
    )
    component_columns = components.transpose(0, 1)
    for start, stop, feature_batch in iter_pca_feature_batches(
        representation_array,
        representation,
        batch_size,
        log_eps,
    ):
        standardized_batch = _standardize_feature_batch(
            feature_batch,
            feature_mean,
            feature_scale,
        )
        batch_scores = standardized_batch @ component_columns
        scores[start:stop] = batch_scores.detach().cpu().numpy()
        del feature_batch, standardized_batch, batch_scores
    return scores


def _calculate_explained_variance_ratio(
    singular_values,
    total_squared_sum,
    sample_count,
    n_components,
):
    """Divide retained sketch variance by complete standardized energy."""

    denominator = sample_count - 1
    explained_variance = (
        singular_values[:n_components].square() / denominator
    )
    total_variance = total_squared_sum / denominator
    return explained_variance / total_variance
