"""Stage 0: orchestrate one memory-bounded label-blind PCA analysis.

This public stage receives the configured dataset and output paths, then
calls stages 1 through 7 in filename order. The representation memory map
remains open only while three sample-batched PCA passes calculate global
scaling, fit the retained component basis, and project final scores. CPU
diagnostics, visualization, and output stages then consume only the small
score arrays and aligned metadata. Their public output contains two
physical-class figures and one consolidated metrics report.

The stage receives a PCA configuration, a saved dataset directory, an
output directory, a representation name, and optional method/device
overrides. It returns a concise summary after the final stage saves the
analysis outputs.
"""

from copy import deepcopy
from pathlib import Path

from src.analysis.distributed_pca import (
    destroy_multi_gpu_pca,
    fit_feature_sharded_pca,
    fit_feature_sharded_scaling,
    get_feature_shard_bounds,
    initialize_multi_gpu_pca,
)
from src.analysis.step_01_load_pca_features import (
    get_pca_feature_shape,
    load_pca_feature_source,
)
from src.analysis.step_02_scale_pca_features import (
    fit_pca_scaling,
)
from src.analysis.step_03_fit_pca import fit_pca
from src.analysis.step_04_fit_kmeans import fit_kmeans
from src.analysis.step_05_calculate_physical_metrics import (
    calculate_kmeans_physical_metrics,
    calculate_nearest_neighbor_purity,
    create_sample_count_table,
    load_physical_classes,
)
from src.analysis.step_06_fit_tsne import fit_tsne
from src.analysis.step_07_save_pca_outputs import save_pca_outputs


def run_pca_analysis(
    config,
    dataset_dir,
    output_dir,
    representation,
    *,
    pca_method=None,
    device=None,
    batch_size=None,
    analysis_components=None,
    nearest_neighbors=None,
    multi_gpu=None,
):
    """
    Run and save one raw or Hermite PCA analysis.

    This public entry point keeps labels out of streamed PyTorch feature
    scaling and incremental PCA as well as CPU KMeans, nearest-neighbor, and
    t-SNE fits. Physical classes determine only the requested KMeans
    partition count before fitting, then describe internal diagnostics and
    color the two figures. The complete raw or Hermite feature matrix is
    never materialized.

    Parameters
    ----------
    config : dict
        PCA workflow settings for raw preprocessing, PyTorch PCA, internal
        KMeans diagnostics, nearest neighbors, t-SNE, plotting, and the
        shared random seed.
    dataset_dir : str or pathlib.Path
        Directory containing the current saved dataset files.
    output_dir : str or pathlib.Path
        Directory in which stage 7 saves the PCA outputs.
    representation : {"raw", "hermite"}
        Saved representation used to construct label-blind features.
    pca_method : {"full", "lowrank"}, optional
        Override for incremental exact-SVD or randomized low-rank updates.
    device : str, optional
        Override for the scaling and PCA device. Accepted current values are
        ``auto``, ``cpu``, ``cuda``, and indexed CUDA devices.
    batch_size : int, optional
        Override for the number of consecutive saved samples prepared during
        each scaling, basis-fitting, and score pass.
    analysis_components : int, optional
        Override for the leading PCA width shared by KMeans, silhouette,
        nearest-neighbor purity, and t-SNE.
    nearest_neighbors : int, optional
        Override for the number of PCA-space neighbors used by local
        physical-label purity.
    multi_gpu : bool, optional
        Override for feature-sharded multi-GPU PCA launched by ``torchrun``.

    Returns
    -------
    dict or None
        ``output_dir``, representation, PCA method, resolved device, sample
        and feature counts, sample batch size, retained and downstream
        component counts, weighted KMeans purity, and paths for the two
        physical-class figures and consolidated metrics report. Nonzero
        multi-GPU ranks return ``None`` after distributed PCA completes.

    Notes
    -----
    Labels are never passed to PCA, KMeans, nearest-neighbor fitting, or
    t-SNE. The physical class count determines only KMeans ``n_clusters``;
    label values enter after fitting for aggregate metrics and plotting.
    """

    resolved = deepcopy(config)
    if pca_method is not None:
        resolved["pca"]["method"] = pca_method
    if device is not None:
        resolved["pca"]["device"] = device
    if batch_size is not None:
        resolved["pca"]["batch_size"] = batch_size
    if analysis_components is not None:
        resolved["analysis_components"] = analysis_components
    if nearest_neighbors is not None:
        resolved["nearest_neighbors"] = nearest_neighbors
    if multi_gpu is not None:
        resolved["pca"]["multi_gpu"] = multi_gpu

    dataset_dir = Path(dataset_dir).resolve()
    output_dir = Path(output_dir).resolve()
    random_state = resolved["random_state"]

    pca_config = resolved["pca"]
    multi_gpu_enabled = pca_config["multi_gpu"]
    if multi_gpu_enabled:
        (
            process_group,
            rank,
            world_size,
            distributed_device,
        ) = initialize_multi_gpu_pca()
    else:
        process_group = None
        rank = 0
        world_size = 1
    log_eps = (
        resolved["raw"]["log_eps"]
        if representation == "raw"
        else None
    )
    with load_pca_feature_source(
        dataset_dir,
        representation,
    ) as feature_source:
        metadata = (
            feature_source.metadata.copy()
            if rank == 0
            else None
        )
        source_shape = tuple(
            int(value) for value in feature_source.values.shape
        )
        sample_count, feature_count = get_pca_feature_shape(
            feature_source.values
        )
        if multi_gpu_enabled:
            feature_start, feature_stop = get_feature_shard_bounds(
                feature_count,
                rank,
                world_size,
            )
            feature_mean, feature_scale = fit_feature_sharded_scaling(
                representation_array=feature_source.values,
                representation=representation,
                batch_size=pca_config["batch_size"],
                device=distributed_device,
                feature_start=feature_start,
                feature_stop=feature_stop,
                log_eps=log_eps,
            )
            selected_device = str(distributed_device)
            scores, explained_variance_ratio = fit_feature_sharded_pca(
                representation_array=feature_source.values,
                representation=representation,
                feature_mean=feature_mean,
                feature_scale=feature_scale,
                batch_size=pca_config["batch_size"],
                config=pca_config,
                feature_start=feature_start,
                feature_stop=feature_stop,
                process_group=process_group,
                log_eps=log_eps,
            )
        else:
            feature_mean, feature_scale = fit_pca_scaling(
                representation_array=feature_source.values,
                representation=representation,
                batch_size=pca_config["batch_size"],
                device=pca_config["device"],
                log_eps=log_eps,
            )
            selected_device = str(feature_mean.device)
            scores, explained_variance_ratio = fit_pca(
                representation_array=feature_source.values,
                representation=representation,
                feature_mean=feature_mean,
                feature_scale=feature_scale,
                batch_size=pca_config["batch_size"],
                config=pca_config,
                random_state=random_state,
                log_eps=log_eps,
            )
        del feature_mean, feature_scale
    if multi_gpu_enabled:
        destroy_multi_gpu_pca(process_group)
        if rank != 0:
            return None
    analysis_component_count = min(
        int(resolved["analysis_components"]),
        int(scores.shape[1]),
    )
    analysis_scores = scores[:, :analysis_component_count]
    physical_class_count = int(metadata["class_name"].nunique())
    kmeans_model, kmeans_labels, kmeans_silhouette = fit_kmeans(
        analysis_scores,
        resolved["kmeans"],
        random_state,
        physical_class_count,
    )
    class_ids, class_names = load_physical_classes(metadata)
    (
        cluster_purity,
        cluster_composition,
        class_concentration,
        weighted_purity,
    ) = calculate_kmeans_physical_metrics(
        kmeans_labels,
        class_ids,
        class_names,
    )
    sample_counts = create_sample_count_table(metadata)
    sample_neighbor_purity, neighbor_purity = (
        calculate_nearest_neighbor_purity(
            analysis_scores,
            metadata,
            resolved["nearest_neighbors"],
        )
    )
    del sample_neighbor_purity
    _, tsne_embedding = fit_tsne(
        analysis_scores,
        resolved["tsne"],
        random_state,
    )
    paths = save_pca_outputs(
        output_dir=output_dir,
        dataset_dir=dataset_dir,
        representation=representation,
        source_shape=source_shape,
        metadata=metadata,
        pca_scores=scores,
        explained_variance_ratio=explained_variance_ratio,
        tsne_embedding=tsne_embedding,
        sample_counts=sample_counts,
        cluster_purity=cluster_purity,
        cluster_composition=cluster_composition,
        class_concentration=class_concentration,
        neighbor_purity=neighbor_purity,
        kmeans_silhouette=kmeans_silhouette,
        weighted_purity=weighted_purity,
        kmeans_cluster_count=int(kmeans_model.n_clusters),
        analysis_component_count=analysis_component_count,
        selected_device=selected_device,
        config=resolved,
    )

    return {
        "output_dir": output_dir,
        "representation": representation,
        "pca_method": resolved["pca"]["method"],
        "device": selected_device,
        "multi_gpu": multi_gpu_enabled,
        "world_size": world_size,
        "sample_count": sample_count,
        "feature_count": feature_count,
        "batch_size": int(pca_config["batch_size"]),
        "n_components": int(pca_config["n_components"]),
        "analysis_components": analysis_component_count,
        "nearest_neighbors": int(resolved["nearest_neighbors"]),
        "weighted_purity": float(weighted_purity),
        "paths": paths,
    }
