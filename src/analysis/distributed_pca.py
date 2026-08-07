"""Feature-sharded distributed scaling and incremental PCA.

Every rank visits all saved samples while owning one contiguous interval of
flattened feature columns. Incremental PCA updates form only a local row Gram
matrix. Their sum is shared across ranks, rank zero solves the small symmetric
eigenproblem, and its eigenpairs are broadcast for consistent local component
reconstruction. No complete feature batch or component matrix is
communicated.
"""

import os

import numpy as np
import torch
import torch.distributed as dist

from src.analysis.step_01_load_pca_features import (
    get_pca_feature_dtype,
    get_pca_feature_shape,
    iter_pca_feature_shard_batches,
)


def initialize_multi_gpu_pca():
    """Initialize the production NCCL group from ``torchrun`` environment.

    One process selects the CUDA device identified by ``LOCAL_RANK`` and
    joins the single-node group described by torchrun environment variables.
    The returned values are reused directly by every feature-sharded PCA
    pass; no separate runtime or result object is introduced.

    Returns
    -------
    process_group : torch.distributed.ProcessGroup
        Default single-node NCCL process group.
    rank : int
        Global torchrun rank used for feature ownership and output control.
    world_size : int
        Number of feature shards and participating GPUs.
    device : torch.device
        CUDA device selected from the rank's ``LOCAL_RANK``.
    """

    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    dist.init_process_group(backend="nccl", init_method="env://")
    process_group = dist.group.WORLD
    return (
        process_group,
        dist.get_rank(process_group),
        dist.get_world_size(process_group),
        device,
    )


def destroy_multi_gpu_pca(process_group):
    """Synchronize successful ranks and destroy the production process group.

    Parameters
    ----------
    process_group : torch.distributed.ProcessGroup
        Default group returned by :func:`initialize_multi_gpu_pca`.
    """

    dist.barrier(group=process_group)
    dist.destroy_process_group(process_group)


def get_feature_shard_bounds(feature_count, rank, world_size):
    """Return one balanced contiguous flattened-feature interval.

    Parameters
    ----------
    feature_count : int
        Complete flattened representation width.
    rank : int
        Global process rank.
    world_size : int
        Number of feature-sharding ranks.

    Returns
    -------
    tuple of int
        Inclusive start and exclusive stop column offsets.
    """

    feature_start = int(feature_count) * int(rank) // int(world_size)
    feature_stop = (
        int(feature_count) * (int(rank) + 1) // int(world_size)
    )
    return feature_start, feature_stop


def fit_feature_sharded_scaling(
    representation_array,
    representation,
    batch_size,
    device,
    *,
    feature_start,
    feature_stop,
    log_eps=None,
):
    """Fit rank-local float64 population scaling statistics.

    Every rank visits all sample rows but combines Chan/Welford moments only
    for its owned feature interval. No cross-rank reduction is needed because
    each feature sees the complete sample sequence on exactly one rank.

    Parameters
    ----------
    representation_array : numpy.ndarray
        Read-only raw or Hermite saved representation.
    representation : {"raw", "hermite"}
        Feature preparation branch.
    batch_size : int
        Maximum saved samples processed together.
    device : str or torch.device
        Rank-local CPU or CUDA device.
    feature_start : int
        Inclusive global feature offset owned by this rank.
    feature_stop : int
        Exclusive global feature offset owned by this rank.
    log_eps : float, optional
        Positive raw-VDF floor applied before ``log10``.

    Returns
    -------
    feature_mean : torch.Tensor
        Local float64 feature means with shape ``(local_features,)``.
    feature_scale : torch.Tensor
        Local float64 population scales with zero scales replaced by one.
    """

    selected_device = torch.device(device)
    running_count = 0
    running_mean = None
    running_m2 = None
    with torch.inference_mode():
        for _, _, feature_batch in iter_pca_feature_shard_batches(
            representation_array,
            representation,
            batch_size,
            feature_start,
            feature_stop,
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


def fit_feature_sharded_pca(
    representation_array,
    representation,
    feature_mean,
    feature_scale,
    batch_size,
    config,
    *,
    feature_start,
    feature_stop,
    process_group=None,
    log_eps=None,
):
    """Fit and score one feature-sharded incremental PCA basis.

    Full mode retains ``n_components`` rows after each exact row-Gram solve.
    Low-rank mode uses the same exact small solve but retains
    ``n_components + lowrank_oversampling`` rows between updates. Rank zero
    alone returns the complete CPU scores and explained-variance ratios.

    Parameters
    ----------
    representation_array : numpy.ndarray
        Read-only raw or Hermite saved representation.
    representation : {"raw", "hermite"}
        Feature preparation branch.
    feature_mean : torch.Tensor
        Rank-local float64 feature means.
    feature_scale : torch.Tensor
        Rank-local float64 population scales.
    batch_size : int
        Maximum saved samples processed together.
    config : dict
        PCA settings containing method, retained rank, and oversampling.
    feature_start : int
        Inclusive global feature offset owned by this rank.
    feature_stop : int
        Exclusive global feature offset owned by this rank.
    process_group : torch.distributed.ProcessGroup, optional
        Initialized NCCL or externally supplied test process group.
    log_eps : float, optional
        Positive raw-VDF floor applied before ``log10``.

    Returns
    -------
    scores : numpy.ndarray or None
        Complete ``(n_samples, n_components)`` scores on rank zero only.
    explained_variance_ratio : numpy.ndarray or None
        Retained explained-variance ratios on rank zero only.
    """

    if process_group is None:
        process_group = dist.group.WORLD
    rank = dist.get_rank(process_group)
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
        components, singular_values, total_squared_sum = (
            _fit_feature_sharded_basis(
                representation_array=representation_array,
                representation=representation,
                feature_mean=fit_mean,
                feature_scale=fit_scale,
                batch_size=batch_size,
                working_rank=working_rank,
                feature_start=feature_start,
                feature_stop=feature_stop,
                process_group=process_group,
                log_eps=log_eps,
            )
        )
        _orient_feature_sharded_components(
            components=components,
            n_components=n_components,
            feature_start=feature_start,
            process_group=process_group,
        )
        scores = _calculate_feature_sharded_scores(
            representation_array=representation_array,
            representation=representation,
            feature_mean=fit_mean,
            feature_scale=fit_scale,
            components=components[:n_components],
            batch_size=batch_size,
            feature_dtype=feature_dtype,
            feature_start=feature_start,
            feature_stop=feature_stop,
            process_group=process_group,
            log_eps=log_eps,
        )
        dist.all_reduce(
            total_squared_sum,
            op=dist.ReduceOp.SUM,
            group=process_group,
        )
        if rank == 0:
            variance_denominator = sample_count - 1
            explained_variance = (
                singular_values[:n_components].square()
                / variance_denominator
            )
            total_variance = total_squared_sum / variance_denominator
            explained_variance_ratio = (
                explained_variance / total_variance
            ).detach().cpu().numpy()
        else:
            explained_variance_ratio = None
    return scores, explained_variance_ratio


def _fit_feature_sharded_basis(
    *,
    representation_array,
    representation,
    feature_mean,
    feature_scale,
    batch_size,
    working_rank,
    feature_start,
    feature_stop,
    process_group,
    log_eps,
):
    """Visit every local shard batch and retain distributed component rows.

    Consecutive batches are standardized on their owning rank. Initial rows
    accumulate until the retained working rank can be formed; later updates
    stack the rank-local weighted previous basis above only the current local
    sample columns. The function also accumulates local standardized energy
    for the final cross-rank explained-variance calculation.

    Parameters
    ----------
    representation_array : numpy.ndarray
        Read-only memory-mapped raw or Hermite samples.
    representation : {"raw", "hermite"}
        Rank-local feature preparation branch.
    feature_mean, feature_scale : torch.Tensor
        Rank-local scaling vectors on the owning device.
    batch_size : int
        Consecutive sample rows prepared in each update.
    working_rank : int
        Component rows retained between incremental updates.
    feature_start, feature_stop : int
        Global half-open feature interval owned by this rank.
    process_group : torch.distributed.ProcessGroup
        Group that sums local row-Gram contributions.
    log_eps : float or None
        Positive raw-VDF floor; Hermite coefficients do not use it.

    Returns
    -------
    components : torch.Tensor
        Retained rank-local component columns.
    singular_values : torch.Tensor
        Replicated singular values from the last global update.
    total_squared_sum : torch.Tensor
        Scalar standardized energy contributed by this feature shard.
    """

    sample_count = int(representation_array.shape[0])
    local_feature_count = int(feature_stop) - int(feature_start)
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

    for _, _, feature_batch in iter_pca_feature_shard_batches(
        representation_array,
        representation,
        batch_size,
        feature_start,
        feature_stop,
        log_eps,
    ):
        standardized_batch = _standardize_local_batch(
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
                    (initial_rows, local_feature_count),
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
            previous_basis_rows = singular_values[:, None] * components
            update_matrix = torch.cat(
                (previous_basis_rows, standardized_batch),
                dim=0,
            )
            del previous_basis_rows, components, singular_values

        del feature_batch, standardized_batch
        singular_values, components = _update_row_gram_basis(
            update_matrix,
            working_rank,
            process_group,
        )
        del update_matrix

    return components, singular_values, total_squared_sum


def _update_row_gram_basis(
    update_matrix,
    working_rank,
    process_group,
):
    """Update sharded component columns from one global row-Gram matrix.

    Each rank multiplies its local update columns by their transpose. An
    all-reduce sums these contributions into the exact row Gram of the
    conceptual complete update without assembling its feature columns.
    Rank zero solves that small replicated matrix and broadcasts its
    eigenpairs so every feature shard uses one identical left basis. Each
    rank then reconstructs only the right-singular-vector columns it owns.

    Parameters
    ----------
    update_matrix : torch.Tensor
        Local update columns with shape
        ``(update_rows, local_feature_count)``.
    working_rank : int
        Maximum component rows retained after this update.
    process_group : torch.distributed.ProcessGroup
        Group that all-reduces the row-Gram contribution.

    Returns
    -------
    singular_values : torch.Tensor
        Leading global singular values replicated on every rank.
    components : torch.Tensor
        Local component columns with shape
        ``(retained_rank, local_feature_count)``.
    """

    row_count = int(update_matrix.shape[0])
    retained_rank = min(int(working_rank), row_count)
    row_gram = update_matrix @ update_matrix.transpose(0, 1)
    dist.all_reduce(
        row_gram,
        op=dist.ReduceOp.SUM,
        group=process_group,
    )
    rank = dist.get_rank(process_group)
    if rank == 0:
        eigenvalues, eigenvectors = torch.linalg.eigh(row_gram)
        eigenvectors = eigenvectors.contiguous()
    else:
        eigenvalues = torch.empty(
            row_count,
            dtype=row_gram.dtype,
            device=row_gram.device,
        )
        eigenvectors = torch.empty_like(row_gram)
    dist.broadcast(eigenvalues, src=0, group=process_group)
    dist.broadcast(eigenvectors, src=0, group=process_group)
    retained_indices = torch.arange(
        row_count - 1,
        row_count - retained_rank - 1,
        -1,
        device=row_gram.device,
    )
    retained_eigenvalues = eigenvalues.index_select(
        0,
        retained_indices,
    )
    left_vectors = eigenvectors.index_select(
        1,
        retained_indices,
    ).contiguous()
    singular_values = torch.sqrt(
        torch.clamp(retained_eigenvalues, min=0)
    )
    components = left_vectors.transpose(0, 1) @ update_matrix
    positive = singular_values > 0
    divisors = torch.where(
        positive,
        singular_values,
        torch.ones_like(singular_values),
    )
    components.div_(divisors[:, None])
    components.masked_fill_(~positive[:, None], 0)
    return singular_values, components


def _orient_feature_sharded_components(
    *,
    components,
    n_components,
    feature_start,
    process_group,
):
    """Apply one deterministic sign to every distributed component row.

    Each rank contributes only the magnitude, signed value, and global
    feature index of its largest local loading. All ranks exchange these
    small per-component tensors, select the globally largest loading with
    the lowest feature index breaking exact ties, and multiply their local
    component columns by the same sign. Complete component rows are never
    gathered.

    Parameters
    ----------
    components : torch.Tensor
        Rank-local component columns with shape
        ``(working_rank, local_feature_count)``.
    n_components : int
        Final component rows receiving deterministic orientation.
    feature_start : int
        Global index of the first local component column.
    process_group : torch.distributed.ProcessGroup
        Single-node group used to exchange orientation candidates.
    """

    world_size = dist.get_world_size(process_group)
    retained = components[:n_components]
    local_max_values, local_max_indices = torch.max(retained, dim=1)
    local_min_values, local_min_indices = torch.min(retained, dim=1)
    max_abs = torch.abs(local_max_values)
    min_abs = torch.abs(local_min_values)
    select_min = (min_abs > max_abs) | (
        (min_abs == max_abs) & (local_min_indices < local_max_indices)
    )
    local_abs = torch.where(select_min, min_abs, max_abs)
    local_indices = torch.where(
        select_min,
        local_min_indices,
        local_max_indices,
    )
    local_values = torch.where(
        select_min,
        local_min_values,
        local_max_values,
    )
    component_indices = torch.arange(
        n_components,
        device=components.device,
    )
    global_indices = local_indices.to(torch.int64) + int(feature_start)

    gathered_abs = [torch.empty_like(local_abs) for _ in range(world_size)]
    gathered_values = [
        torch.empty_like(local_values) for _ in range(world_size)
    ]
    gathered_indices = [
        torch.empty_like(global_indices) for _ in range(world_size)
    ]
    dist.all_gather(
        gathered_abs,
        local_abs,
        group=process_group,
    )
    dist.all_gather(
        gathered_values,
        local_values,
        group=process_group,
    )
    dist.all_gather(
        gathered_indices,
        global_indices,
        group=process_group,
    )
    candidate_abs = torch.stack(gathered_abs)
    candidate_values = torch.stack(gathered_values)
    candidate_indices = torch.stack(gathered_indices)
    largest_abs = candidate_abs.max(dim=0).values
    tied_indices = torch.where(
        candidate_abs == largest_abs[None, :],
        candidate_indices,
        torch.full_like(candidate_indices, torch.iinfo(torch.int64).max),
    )
    owner_ranks = tied_indices.argmin(dim=0)
    selected_values = candidate_values[
        owner_ranks,
        component_indices,
    ]
    signs = torch.sign(selected_values)
    signs = torch.where(
        signs == 0,
        torch.ones_like(signs),
        signs,
    )
    retained.mul_(signs[:, None])


def _calculate_feature_sharded_scores(
    *,
    representation_array,
    representation,
    feature_mean,
    feature_scale,
    components,
    batch_size,
    feature_dtype,
    feature_start,
    feature_stop,
    process_group,
    log_eps,
):
    """Reduce local score contributions into rank-zero sample rows.

    The final memory-map pass prepares only the current feature interval.
    Each rank projects those standardized columns onto its oriented local
    component shard, and a sum reduction produces the complete score batch
    on rank zero. Nonzero ranks never allocate the complete score matrix.

    Parameters
    ----------
    representation_array : numpy.ndarray
        Read-only memory-mapped raw or Hermite samples.
    representation : {"raw", "hermite"}
        Rank-local feature preparation branch.
    feature_mean, feature_scale : torch.Tensor
        Rank-local scaling vectors.
    components : torch.Tensor
        Final oriented local component columns.
    batch_size : int
        Consecutive score rows projected together.
    feature_dtype : numpy.dtype
        Dtype of the rank-zero NumPy score matrix.
    feature_start, feature_stop : int
        Global half-open feature interval owned by this rank.
    process_group : torch.distributed.ProcessGroup
        Group used to reduce score contributions.
    log_eps : float or None
        Positive raw-VDF floor; Hermite coefficients do not use it.

    Returns
    -------
    numpy.ndarray or None
        Complete ``(n_samples, n_components)`` scores on rank zero only.
    """

    rank = dist.get_rank(process_group)
    if rank == 0:
        scores = np.empty(
            (int(representation_array.shape[0]), int(components.shape[0])),
            dtype=feature_dtype,
        )
    else:
        scores = None
    component_columns = components.transpose(0, 1)
    for start, stop, feature_batch in iter_pca_feature_shard_batches(
        representation_array,
        representation,
        batch_size,
        feature_start,
        feature_stop,
        log_eps,
    ):
        standardized_batch = _standardize_local_batch(
            feature_batch,
            feature_mean,
            feature_scale,
        )
        batch_scores = standardized_batch @ component_columns
        dist.reduce(
            batch_scores,
            dst=0,
            op=dist.ReduceOp.SUM,
            group=process_group,
        )
        if rank == 0:
            scores[start:stop] = batch_scores.detach().cpu().numpy()
        del feature_batch, standardized_batch, batch_scores
    return scores


def _standardize_local_batch(feature_batch, feature_mean, feature_scale):
    """Move and standardize one rank-local feature batch in place.

    Parameters
    ----------
    feature_batch : numpy.ndarray
        Prepared local columns with shape ``(batch_rows, local_features)``.
    feature_mean, feature_scale : torch.Tensor
        Local global-sample scaling vectors on the rank device.

    Returns
    -------
    torch.Tensor
        Standardized batch on the rank-local PCA device.
    """

    feature_tensor = torch.as_tensor(
        feature_batch,
        device=feature_mean.device,
    )
    feature_tensor.sub_(feature_mean)
    feature_tensor.div_(feature_scale)
    return feature_tensor
