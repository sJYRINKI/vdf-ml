import os
from pathlib import Path
import time

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.decomposition import IncrementalPCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import torch

from src.batches import create_contiguous_slice, iter_index_batches
from src.dataset_io import load_dataset
from src.model_split import split_by_timestep
from src.timesteps import create_timestep_path


def print_pca_progress(value, stage, **fields):
    """
    Print a machine-readable PCA progress line.

    Parameters
    ----------
    value : float
        Progress value between 0.0 and 1.0.
    stage : str
        Current processing stage.
    **fields : dict
        Extra key-value fields printed on the same line.
    """

    value = min(1.0, max(0.0, float(value)))
    parts = [f"pca_progress={value:.3f}", f"stage={stage}"]
    for key, field_value in fields.items():
        parts.append(f"{key}={field_value}")

    print(" ".join(parts), flush=True)


def print_batch_progress(stage, batch_number, n_batches, start, end):
    """
    Print progress for a batched PCA subtask.

    Parameters
    ----------
    stage : str
        Current batch stage.
    batch_number : int
        Completed batch number.
    n_batches : int
        Total number of batches.
    start : float
        Progress value at the start of the subtask.
    end : float
        Progress value at the end of the subtask.
    """

    if start is None or end is None:
        return

    if n_batches <= 0:
        progress = end
    else:
        fraction = int(batch_number) / int(n_batches)
        progress = float(start) + (float(end) - float(start)) * fraction

    print_pca_progress(
        progress,
        stage,
        batch=f"{int(batch_number)}/{int(n_batches)}",
    )


def plot_dataset_pca(config, timestep, pca_id=None):
    """
    Create PCA plots and metrics for a saved VDF dataset.

    Parameters
    ----------
    config : dict
        PCA plotting config.
    timestep : str
        Dataset timestep identifier.
    pca_id : str, optional
        PCA output version identifier. Overrides ``output_version`` in config.
    """

    start = time.perf_counter()
    print_pca_progress(0.0, "start")
    dataset_dir = create_timestep_path(
        path_template=config["dataset_dir"],
        timestep=timestep,
    )
    output_dir = create_timestep_path(
        path_template=config["output_dir"],
        timestep=timestep,
    )
    output_version = pca_id
    if output_version is None:
        output_version = config.get("output_version")
    output_dir = resolve_versioned_output_dir(
        output_dir=output_dir,
        output_version=output_version,
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    feature_config = resolve_feature_config(config.get("features", {}))
    pca_config = resolve_pca_config(config.get("pca", {}))
    pca_fit_config = resolve_pca_fit_config(config.get("pca_fit", {}))
    neighbor_config = resolve_neighbor_config(config.get("neighbor_metrics", {}))
    filter_preview_config = resolve_filter_preview_config(
        config.get("filter_preview", {})
    )
    plot_config = resolve_plot_config(config.get("plot", {}))
    embedding_plot_config = resolve_embedding_plot_config(
        config.get("embedding_plot", {})
    )
    split_config = config.get("split", {})

    train_fraction = float(split_config.get("train_fraction", 0.6))
    validation_fraction = float(split_config.get("validation_fraction", 0.2))
    gap_timesteps = int(split_config.get("gap_timesteps", 10))

    X, y, metadata = load_dataset(dataset_dir, mmap=True)
    print_pca_progress(0.03, "dataset_loaded")
    (
        train_indices,
        validation_indices,
        test_indices,
        train_timesteps,
        validation_timesteps,
        test_timesteps,
        train_validation_gap,
        validation_test_gap,
    ) = split_by_timestep(
        metadata=metadata,
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
        gap_timesteps=gap_timesteps,
    )
    print_pca_progress(0.05, "split_done")

    pca_fit_indices = select_pca_fit_indices(
        train_indices=train_indices,
        y=y,
        metadata=metadata,
        pca_fit_config=pca_fit_config,
    )
    n_components = resolve_n_components(
        configured_n_components=pca_config["n_components"],
        X=X,
        train_indices=pca_fit_indices,
        downsample_factor=feature_config["downsample_factor"],
    )
    print_pca_progress(0.08, "configuration_done")

    print(f"Dataset directory: {dataset_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Dataset shape: {X.shape}")
    print(f"Training samples: {len(train_indices)}")
    print(f"PCA fit samples: {len(pca_fit_indices)}")
    print(f"PCA components: {n_components}")
    print(f"PCA backend: {pca_config['backend']}")

    split_indices = {
        "train": train_indices,
        "validation": validation_indices,
        "test": test_indices,
    }
    if pca_config["backend"] == "torch":
        pca_result = fit_transform_torch_pca(
            X=X,
            split_indices=split_indices,
            pca_fit_indices=pca_fit_indices,
            n_components=n_components,
            feature_config=feature_config,
            pca_config=pca_config,
        )
        scores_by_split = pca_result["scores_by_split"]
        explained_variance_ratio = pca_result["explained_variance_ratio"]
        pca_summary = pca_result["summary"]
    else:
        scaler = fit_feature_scaler(
            X=X,
            indices=pca_fit_indices,
            feature_config=feature_config,
            progress_start=0.10,
            progress_end=0.30,
            progress_stage="sklearn_scaler_fit",
        )
        pca = fit_incremental_pca(
            X=X,
            indices=pca_fit_indices,
            scaler=scaler,
            n_components=n_components,
            feature_config=feature_config,
            progress_start=0.30,
            progress_end=0.60,
            progress_stage="sklearn_pca_fit",
        )
        scores_by_split = transform_splits(
            X=X,
            split_indices=split_indices,
            scaler=scaler,
            pca=pca,
            feature_config=feature_config,
            progress_start=0.60,
            progress_end=0.84,
        )
        explained_variance_ratio = np.asarray(
            pca.explained_variance_ratio_,
            dtype=np.float64,
        )
        pca_summary = {
            "backend": "sklearn",
            "algorithm": "incremental",
            "device": "cpu",
            "fit_sample_count": len(pca_fit_indices),
            "fit_balanced": pca_fit_config["balanced"],
        }

    sample_metrics = create_sample_metrics(
        y=y,
        metadata=metadata,
        split_indices=split_indices,
        scores_by_split=scores_by_split,
    )
    print_pca_progress(0.86, "sample_metrics_done")

    if neighbor_config["enabled"]:
        print_pca_progress(0.87, "neighbor_metrics_start")
        add_neighbor_metrics(
            sample_metrics=sample_metrics,
            scores_by_split=scores_by_split,
            y=y,
            train_indices=train_indices,
            neighbor_config=neighbor_config,
        )
        print_pca_progress(0.92, "neighbor_metrics_done")

    if filter_preview_config["enabled"]:
        print_pca_progress(0.925, "filter_preview_start")
        add_filter_preview_metrics(
            sample_metrics=sample_metrics,
            scores_by_split=scores_by_split,
            y=y,
            train_indices=train_indices,
            neighbor_config=neighbor_config,
            filter_preview_config=filter_preview_config,
        )
        print_pca_progress(0.935, "filter_preview_done")

    metrics_text = create_pca_metrics_text(
        dataset_dir=dataset_dir,
        output_dir=output_dir,
        X=X,
        sample_metrics=sample_metrics,
        explained_variance_ratio=explained_variance_ratio,
        n_components=n_components,
        feature_config=feature_config,
        pca_summary=pca_summary,
        pca_fit_config=pca_fit_config,
        split_config={
            "train_fraction": train_fraction,
            "validation_fraction": validation_fraction,
            "gap_timesteps": gap_timesteps,
            "train_timesteps": train_timesteps,
            "validation_timesteps": validation_timesteps,
            "test_timesteps": test_timesteps,
            "train_validation_gap": train_validation_gap,
            "validation_test_gap": validation_test_gap,
        },
        neighbor_config=neighbor_config,
        filter_preview_config=filter_preview_config,
        embedding_plot_config=embedding_plot_config,
    )
    print_pca_progress(0.94, "metrics_text_done")

    saved_plot_paths = []
    if plot_config["enabled"]:
        print_pca_progress(0.95, "plots_start")
        saved_plot_paths = save_pca_plots(
            sample_metrics=sample_metrics,
            output_dir=output_dir,
            plot_config=plot_config,
            filter_preview_config=filter_preview_config,
            embedding_plot_config=embedding_plot_config,
            scores_by_split=scores_by_split,
            split_indices=split_indices,
        )
        print_pca_progress(0.97, "plots_done")

    save_pca_outputs(
        output_dir=output_dir,
        sample_metrics=sample_metrics,
        scores_by_split=scores_by_split,
        split_indices=split_indices,
        explained_variance_ratio=explained_variance_ratio,
        pca_summary=pca_summary,
        metrics_text=metrics_text,
    )
    print_pca_progress(1.0, "outputs_saved")

    elapsed = time.perf_counter() - start
    print(output_dir / "pca_metrics.txt")
    print(output_dir / "pca_sample_metrics.csv")
    print(output_dir / "pca_scores.npz")
    for plot_path in saved_plot_paths:
        print(plot_path)
    print(f"PCA plotting complete in {elapsed:.2f} s")


def resolve_feature_config(config):
    """
    Resolve xz-slice feature settings.

    Parameters
    ----------
    config : dict
        Feature config values.

    Returns
    -------
    dict
        Validated feature settings.
    """

    downsample_factor = int(config.get("downsample_factor", 1))
    batch_size = int(config.get("batch_size", 512))
    n_jobs = int(config.get("n_jobs", 1))
    log_eps = float(config.get("log_eps", 1e-30))
    sample_normalization = str(config.get("sample_normalization", "none")).lower()
    sample_norm_eps = float(config.get("sample_norm_eps", 1e-6))

    if downsample_factor <= 0:
        raise ValueError("features.downsample_factor must be positive")
    if batch_size <= 0:
        raise ValueError("features.batch_size must be positive")
    if n_jobs == 0:
        raise ValueError("features.n_jobs must be non-zero")
    if log_eps <= 0.0:
        raise ValueError("features.log_eps must be positive")
    if sample_normalization not in {"none", "standard"}:
        raise ValueError("features.sample_normalization must be 'none' or 'standard'")
    if sample_norm_eps <= 0.0:
        raise ValueError("features.sample_norm_eps must be positive")

    return {
        "downsample_factor": downsample_factor,
        "batch_size": batch_size,
        "n_jobs": n_jobs,
        "log_eps": log_eps,
        "sample_normalization": sample_normalization,
        "sample_norm_eps": sample_norm_eps,
    }


def resolve_versioned_output_dir(output_dir, output_version=None):
    """
    Add a PCA output version suffix when configured.

    Parameters
    ----------
    output_dir : str or pathlib.Path
        Base PCA output directory.
    output_version : object, optional
        Version value. For example, ``0.01`` becomes ``pca_v0.01`` when the
        base output directory is ``pca``.

    Returns
    -------
    pathlib.Path
        Versioned or unmodified output directory.
    """

    output_dir = Path(output_dir)
    if output_version is None:
        return output_dir

    version = str(output_version).strip()
    if version == "":
        return output_dir
    if "/" in version or "\\" in version or any(char.isspace() for char in version):
        raise ValueError(
            "output_version must not contain path separators or whitespace"
        )
    if not version.startswith("v"):
        version = f"v{version}"

    return output_dir.with_name(f"{output_dir.name}_{version}")


def resolve_pca_config(config):
    """
    Resolve PCA settings.

    Parameters
    ----------
    config : dict
        PCA config values.

    Returns
    -------
    dict
        Validated PCA settings.
    """

    backend = str(config.get("backend", "sklearn")).lower()
    if backend in {"incremental", "incremental_pca", "sklearn_incremental"}:
        backend = "sklearn"
    if backend not in {"sklearn", "torch"}:
        raise ValueError("pca.backend must be 'sklearn' or 'torch'")

    algorithm = str(config.get("algorithm", "lowrank")).lower()
    if backend == "sklearn":
        algorithm = "incremental"
    elif algorithm not in {"lowrank", "exact"}:
        raise ValueError("pca.algorithm must be 'lowrank' or 'exact'")

    n_components = int(config.get("n_components", 32))
    oversamples = int(config.get("oversamples", 16))
    niter = int(config.get("niter", 2))
    random_state = int(config.get("random_state", 1234))

    if n_components <= 0:
        raise ValueError("pca.n_components must be positive")
    if oversamples < 0:
        raise ValueError("pca.oversamples must be non-negative")
    if niter < 0:
        raise ValueError("pca.niter must be non-negative")

    return {
        "backend": backend,
        "algorithm": algorithm,
        "n_components": n_components,
        "device": str(config.get("device", "auto")),
        "oversamples": oversamples,
        "niter": niter,
        "random_state": random_state,
    }


def resolve_pca_fit_config(config):
    """
    Resolve PCA fitting-sample settings.

    Parameters
    ----------
    config : dict
        PCA fit config values.

    Returns
    -------
    dict
        Validated PCA fit settings.
    """

    balanced = bool(config.get("balanced", False))
    samples_per_class = config.get("samples_per_class", "min")
    if isinstance(samples_per_class, str):
        samples_per_class = samples_per_class.lower()
        if samples_per_class != "min":
            samples_per_class = int(samples_per_class)
    else:
        samples_per_class = int(samples_per_class)

    class_names = config.get("class_names", "all")
    if isinstance(class_names, str):
        if class_names.lower() == "all":
            class_names = []
        else:
            class_names = [class_names]
    class_names = [str(class_name) for class_name in class_names]

    if samples_per_class != "min" and int(samples_per_class) <= 0:
        raise ValueError("pca_fit.samples_per_class must be 'min' or positive")

    return {
        "balanced": balanced,
        "samples_per_class": samples_per_class,
        "class_names": class_names,
        "replace": bool(config.get("replace", False)),
        "random_state": int(config.get("random_state", 1234)),
    }


def resolve_neighbor_config(config):
    """
    Resolve nearest-neighbor metric settings.

    Parameters
    ----------
    config : dict
        Neighbor metric config values.

    Returns
    -------
    dict
        Validated neighbor metric settings.
    """

    enabled = bool(config.get("enabled", True))
    k_neighbors = int(config.get("k_neighbors", 25))
    batch_size = int(config.get("batch_size", 4096))
    thresholds = config.get("low_purity_thresholds", [0.5, 0.65, 0.8])
    thresholds = [float(threshold) for threshold in thresholds]

    if k_neighbors <= 0:
        raise ValueError("neighbor_metrics.k_neighbors must be positive")
    if batch_size <= 0:
        raise ValueError("neighbor_metrics.batch_size must be positive")
    if any(threshold < 0.0 or threshold > 1.0 for threshold in thresholds):
        raise ValueError(
            "neighbor_metrics.low_purity_thresholds must be between zero and one"
        )

    return {
        "enabled": enabled,
        "k_neighbors": k_neighbors,
        "batch_size": batch_size,
        "low_purity_thresholds": thresholds,
    }


def resolve_filter_preview_config(config):
    """
    Resolve preview settings for future PCA-space filtering.

    Parameters
    ----------
    config : dict
        Filter-preview config values.

    Returns
    -------
    dict
        Validated filter-preview settings.
    """

    enabled = bool(config.get("enabled", False))
    candidate_classes = [
        str(class_name)
        for class_name in config.get("candidate_classes", ["exhaust", "dayside"])
    ]
    point_neighbor_classes = [
        str(class_name)
        for class_name in config.get(
            "point_neighbor_classes",
            ["reconnection", "o_point"],
        )
    ]
    protected_classes = [
        str(class_name)
        for class_name in config.get("protected_classes", point_neighbor_classes)
    ]
    apply_splits = [
        str(split_name)
        for split_name in config.get("apply_splits", ["train"])
    ]
    min_point_neighbor_fraction = float(
        config.get("min_point_neighbor_fraction", 0.5)
    )
    min_point_neighbor_fraction_by_class = parse_class_thresholds(
        config.get("min_point_neighbor_fraction_by_class", {})
    )
    max_same_class_fraction = config.get("max_same_class_fraction")
    if max_same_class_fraction is not None:
        max_same_class_fraction = float(max_same_class_fraction)
    max_same_class_fraction_by_class = parse_class_thresholds(
        config.get("max_same_class_fraction_by_class", {})
    )

    if enabled and not candidate_classes:
        raise ValueError("filter_preview.candidate_classes must not be empty")
    if enabled and not point_neighbor_classes:
        raise ValueError("filter_preview.point_neighbor_classes must not be empty")
    if enabled and not apply_splits:
        raise ValueError("filter_preview.apply_splits must not be empty")
    if not 0.0 <= min_point_neighbor_fraction <= 1.0:
        raise ValueError(
            "filter_preview.min_point_neighbor_fraction must be between zero and one"
        )
    if any(
        threshold < 0.0 or threshold > 1.0
        for threshold in min_point_neighbor_fraction_by_class.values()
    ):
        raise ValueError(
            "filter_preview.min_point_neighbor_fraction_by_class values must be "
            "between zero and one"
        )
    if max_same_class_fraction is not None and not (
        0.0 <= max_same_class_fraction <= 1.0
    ):
        raise ValueError(
            "filter_preview.max_same_class_fraction must be between zero and one"
        )
    if any(
        threshold < 0.0 or threshold > 1.0
        for threshold in max_same_class_fraction_by_class.values()
    ):
        raise ValueError(
            "filter_preview.max_same_class_fraction_by_class values must be "
            "between zero and one"
        )

    return {
        "enabled": enabled,
        "candidate_classes": candidate_classes,
        "point_neighbor_classes": point_neighbor_classes,
        "protected_classes": protected_classes,
        "apply_splits": apply_splits,
        "min_point_neighbor_fraction": min_point_neighbor_fraction,
        "min_point_neighbor_fraction_by_class": min_point_neighbor_fraction_by_class,
        "max_same_class_fraction": max_same_class_fraction,
        "max_same_class_fraction_by_class": max_same_class_fraction_by_class,
    }


def parse_class_thresholds(thresholds):
    """
    Parse a class-name threshold mapping.

    Parameters
    ----------
    thresholds : dict
        Mapping from class name to threshold.

    Returns
    -------
    dict
        Mapping from class name to float threshold.
    """

    if thresholds is None:
        return {}
    if not isinstance(thresholds, dict):
        raise ValueError("class-specific thresholds must be a mapping")

    return {
        str(class_name): float(threshold)
        for class_name, threshold in thresholds.items()
    }


def resolve_plot_config(config):
    """
    Resolve PCA plot settings.

    Parameters
    ----------
    config : dict
        Plot config values.

    Returns
    -------
    dict
        Validated plot settings.
    """

    max_points_per_class = int(config.get("max_points_per_class", 5000))
    random_state = int(config.get("random_state", 1234))
    alpha = float(config.get("alpha", 0.6))
    point_size = float(config.get("point_size", 8))

    if max_points_per_class <= 0:
        raise ValueError("plot.max_points_per_class must be positive")
    if not 0.0 < alpha <= 1.0:
        raise ValueError("plot.alpha must be between zero and one")
    if point_size <= 0.0:
        raise ValueError("plot.point_size must be positive")

    return {
        "enabled": bool(config.get("enabled", True)),
        "max_points_per_class": max_points_per_class,
        "random_state": random_state,
        "alpha": alpha,
        "point_size": point_size,
    }


def resolve_embedding_plot_config(config):
    """
    Resolve optional nonlinear embedding plot settings.

    Parameters
    ----------
    config : dict
        Embedding plot config values.

    Returns
    -------
    dict
        Validated embedding plot settings.
    """

    enabled = bool(config.get("enabled", False))
    method = str(config.get("method", "tsne")).lower()
    splits = [str(split_name) for split_name in config.get("splits", ["train"])]
    max_points_per_class = int(config.get("max_points_per_class", 1000))
    random_state = int(config.get("random_state", 1234))
    perplexity = float(config.get("perplexity", 30.0))
    n_neighbors = int(config.get("n_neighbors", 15))
    min_dist = float(config.get("min_dist", 0.1))

    if method not in {"tsne", "umap"}:
        raise ValueError("embedding_plot.method must be 'tsne' or 'umap'")
    if not splits:
        raise ValueError("embedding_plot.splits must not be empty")
    if max_points_per_class <= 0:
        raise ValueError("embedding_plot.max_points_per_class must be positive")
    if perplexity <= 0.0:
        raise ValueError("embedding_plot.perplexity must be positive")
    if n_neighbors <= 1:
        raise ValueError("embedding_plot.n_neighbors must be greater than one")
    if min_dist < 0.0:
        raise ValueError("embedding_plot.min_dist must be non-negative")

    return {
        "enabled": enabled,
        "method": method,
        "splits": splits,
        "max_points_per_class": max_points_per_class,
        "random_state": random_state,
        "perplexity": perplexity,
        "n_neighbors": n_neighbors,
        "min_dist": min_dist,
    }


def resolve_n_components(configured_n_components, X, train_indices, downsample_factor):
    """
    Return a PCA component count valid for the dataset.

    Parameters
    ----------
    configured_n_components : int
        Requested PCA component count.
    X : numpy.ndarray
        VDF samples with shape ``(n_samples, vx, vy, vz)``.
    train_indices : numpy.ndarray
        Training sample indices.
    downsample_factor : int
        Xz-slice downsampling factor.

    Returns
    -------
    int
        Valid PCA component count.
    """

    if len(train_indices) < 2:
        raise ValueError("At least two training samples are required for PCA")

    feature_count = get_feature_count(
        slice_shape=(int(X.shape[1]), int(X.shape[3])),
        downsample_factor=downsample_factor,
    )
    if feature_count < 1:
        raise ValueError(
            "features.downsample_factor leaves no xz-slice pixels after trimming"
        )

    return min(int(configured_n_components), int(len(train_indices)), feature_count)


def select_pca_fit_indices(train_indices, y, metadata, pca_fit_config):
    """
    Select training indices used to fit the PCA basis.

    Parameters
    ----------
    train_indices : numpy.ndarray
        Training sample indices.
    y : numpy.ndarray
        Dataset labels.
    metadata : pandas.DataFrame
        Dataset metadata.
    pca_fit_config : dict
        PCA fit-sample settings.

    Returns
    -------
    numpy.ndarray
        Dataset indices used to fit PCA.
    """

    train_indices = np.asarray(train_indices, dtype=int)
    if not pca_fit_config["balanced"]:
        return train_indices

    candidate_indices = train_indices
    class_names = pca_fit_config["class_names"]
    if class_names:
        if "class_name" not in metadata.columns:
            raise ValueError("pca_fit.class_names requires metadata class_name column")
        class_mask = metadata.iloc[candidate_indices]["class_name"].isin(class_names)
        candidate_indices = candidate_indices[class_mask.to_numpy()]

    labels = np.asarray(y[candidate_indices], dtype=int)
    label_values, label_counts = np.unique(labels, return_counts=True)
    if len(label_values) == 0:
        raise ValueError("pca_fit selected no training samples")

    samples_per_class = pca_fit_config["samples_per_class"]
    if samples_per_class == "min":
        target_count = int(np.min(label_counts))
    else:
        target_count = int(samples_per_class)

    rng = np.random.default_rng(int(pca_fit_config["random_state"]))
    selected = []
    for label in label_values:
        label_indices = candidate_indices[labels == label]
        replace = bool(pca_fit_config["replace"]) and len(label_indices) < target_count
        sample_count = target_count if replace else min(target_count, len(label_indices))
        selected.append(
            rng.choice(
                label_indices,
                size=sample_count,
                replace=replace,
            )
        )

    pca_fit_indices = np.concatenate(selected).astype(int, copy=False)
    rng.shuffle(pca_fit_indices)

    return pca_fit_indices


def fit_feature_scaler(
    X,
    indices,
    feature_config,
    progress_start=None,
    progress_end=None,
    progress_stage=None,
):
    """
    Fit a standard scaler on training xz-slice features.

    Parameters
    ----------
    X : numpy.ndarray
        Memory-mapped VDF samples.
    indices : numpy.ndarray
        Training sample indices.
    feature_config : dict
        Feature extraction settings.
    progress_start : float, optional
        Progress value at the start of scaler fitting.
    progress_end : float, optional
        Progress value at the end of scaler fitting.
    progress_stage : str, optional
        Stage name for progress logging.

    Returns
    -------
    sklearn.preprocessing.StandardScaler
        Scaler fitted on training features.
    """

    scaler = StandardScaler()
    n_batches = count_index_batches(
        indices=indices,
        batch_size=feature_config["batch_size"],
    )
    for batch_number, features in enumerate(iter_feature_batches(
        X=X,
        indices=indices,
        batch_size=feature_config["batch_size"],
        downsample_factor=feature_config["downsample_factor"],
        n_jobs=feature_config["n_jobs"],
        log_eps=feature_config["log_eps"],
        sample_normalization=feature_config["sample_normalization"],
        sample_norm_eps=feature_config["sample_norm_eps"],
    ), start=1):
        scaler.partial_fit(features)
        if progress_stage is not None:
            print_batch_progress(
                stage=progress_stage,
                batch_number=batch_number,
                n_batches=n_batches,
                start=progress_start,
                end=progress_end,
            )

    return scaler


def fit_incremental_pca(
    X,
    indices,
    scaler,
    n_components,
    feature_config,
    progress_start=None,
    progress_end=None,
    progress_stage=None,
):
    """
    Fit incremental PCA on scaled training xz-slice features.

    Parameters
    ----------
    X : numpy.ndarray
        Memory-mapped VDF samples.
    indices : numpy.ndarray
        Training sample indices.
    scaler : sklearn.preprocessing.StandardScaler
        Scaler fitted on training features.
    n_components : int
        Number of PCA components.
    feature_config : dict
        Feature extraction settings.
    progress_start : float, optional
        Progress value at the start of PCA fitting.
    progress_end : float, optional
        Progress value at the end of PCA fitting.
    progress_stage : str, optional
        Stage name for progress logging.

    Returns
    -------
    sklearn.decomposition.IncrementalPCA
        PCA model fitted on training features.
    """

    pca = IncrementalPCA(n_components=n_components)
    batch_size = max(feature_config["batch_size"], n_components)
    n_batches = count_index_batches(
        indices=indices,
        batch_size=batch_size,
        min_batch_size=n_components,
    )

    for batch_number, features in enumerate(iter_feature_batches(
        X=X,
        indices=indices,
        batch_size=batch_size,
        downsample_factor=feature_config["downsample_factor"],
        n_jobs=feature_config["n_jobs"],
        log_eps=feature_config["log_eps"],
        sample_normalization=feature_config["sample_normalization"],
        sample_norm_eps=feature_config["sample_norm_eps"],
        min_batch_size=n_components,
    ), start=1):
        scaled_features = scaler.transform(features)
        pca.partial_fit(scaled_features)
        if progress_stage is not None:
            print_batch_progress(
                stage=progress_stage,
                batch_number=batch_number,
                n_batches=n_batches,
                start=progress_start,
                end=progress_end,
            )

    return pca


def transform_splits(
    X,
    split_indices,
    scaler,
    pca,
    feature_config,
    progress_start=None,
    progress_end=None,
):
    """
    Transform train, validation, and test samples into PCA space.

    Parameters
    ----------
    X : numpy.ndarray
        Memory-mapped VDF samples.
    split_indices : dict
        Mapping from split name to sample indices.
    scaler : sklearn.preprocessing.StandardScaler
        Scaler fitted on training features.
    pca : sklearn.decomposition.IncrementalPCA
        PCA model fitted on training features.
    feature_config : dict
        Feature extraction settings.
    progress_start : float, optional
        Progress value at the start of split transforms.
    progress_end : float, optional
        Progress value at the end of split transforms.

    Returns
    -------
    dict
        Mapping from split name to PCA scores.
    """

    scores_by_split = {}
    split_names = list(split_indices)
    for split_number, split_name in enumerate(split_names):
        indices = split_indices[split_name]
        if progress_start is None or progress_end is None:
            split_start = None
            split_end = None
        else:
            split_fraction_start = split_number / len(split_names)
            split_fraction_end = (split_number + 1) / len(split_names)
            split_start = progress_start + (
                progress_end - progress_start
            ) * split_fraction_start
            split_end = progress_start + (
                progress_end - progress_start
            ) * split_fraction_end
        scores_by_split[split_name] = transform_indices(
            X=X,
            indices=indices,
            scaler=scaler,
            pca=pca,
            feature_config=feature_config,
            progress_start=split_start,
            progress_end=split_end,
            progress_stage=(
                f"sklearn_transform_{split_name}"
                if split_start is not None and split_end is not None
                else None
            ),
        )

    return scores_by_split


def fit_transform_torch_pca(
    X,
    split_indices,
    pca_fit_indices,
    n_components,
    feature_config,
    pca_config,
):
    """
    Fit PCA with PyTorch on training features and transform all splits.

    Parameters
    ----------
    X : numpy.ndarray
        Memory-mapped VDF samples.
    split_indices : dict
        Mapping from split name to sample indices.
    pca_fit_indices : numpy.ndarray
        Training sample indices used to fit PCA.
    n_components : int
        Number of PCA components.
    feature_config : dict
        Feature extraction settings.
    pca_config : dict
        PCA backend settings.

    Returns
    -------
    dict
        PCA scores, explained variance ratios, and backend summary.
    """

    devices = resolve_torch_devices(pca_config["device"])
    torch.manual_seed(int(pca_config["random_state"]))
    if any(device.type == "cuda" for device in devices):
        torch.cuda.manual_seed_all(int(pca_config["random_state"]))

    print(f"PyTorch PCA devices: {format_torch_devices(devices)}")
    try:
        if len(devices) > 1:
            if pca_config["algorithm"] != "lowrank":
                raise ValueError(
                    "Multi-GPU PCA currently supports only pca.algorithm: lowrank"
                )
            return fit_transform_torch_pca_multi_device(
                X=X,
                split_indices=split_indices,
                pca_fit_indices=pca_fit_indices,
                n_components=n_components,
                feature_config=feature_config,
                pca_config=pca_config,
                devices=devices,
            )

        return fit_transform_torch_pca_single_device(
            X=X,
            split_indices=split_indices,
            pca_fit_indices=pca_fit_indices,
            n_components=n_components,
            feature_config=feature_config,
            pca_config=pca_config,
            device=devices[0],
        )
    except torch.OutOfMemoryError:
        if str(pca_config["device"]).lower() != "auto":
            raise
        clear_cuda_memory(devices)
        print_pca_progress(0.10, "torch_cuda_oom_cpu_fallback")
        return fit_transform_torch_pca_single_device(
            X=X,
            split_indices=split_indices,
            pca_fit_indices=pca_fit_indices,
            n_components=n_components,
            feature_config=feature_config,
            pca_config=pca_config,
            device=torch.device("cpu"),
        )


def fit_transform_torch_pca_single_device(
    X,
    split_indices,
    pca_fit_indices,
    n_components,
    feature_config,
    pca_config,
    device,
):
    """
    Fit PCA with PyTorch on a single device.

    Parameters
    ----------
    X : numpy.ndarray
        Memory-mapped VDF samples.
    split_indices : dict
        Mapping from split name to sample indices.
    pca_fit_indices : numpy.ndarray
        Training sample indices used to fit PCA.
    n_components : int
        Number of PCA components.
    feature_config : dict
        Feature extraction settings.
    pca_config : dict
        PCA backend settings.
    device : torch.device
        Device used for PCA.

    Returns
    -------
    dict
        PCA scores, explained variance ratios, and backend summary.
    """

    fit_features = create_feature_matrix(
        X=X,
        indices=pca_fit_indices,
        feature_config=feature_config,
        progress_start=0.10,
        progress_end=0.35,
        progress_stage="torch_train_feature_matrix",
    )
    fit_tensor = torch.as_tensor(
        fit_features,
        dtype=torch.float32,
        device=device,
    )

    with torch.no_grad():
        mean = fit_tensor.mean(dim=0, keepdim=True)
        std = fit_tensor.std(dim=0, unbiased=False, keepdim=True)
        std = torch.where(std > 0.0, std, torch.ones_like(std))
        fit_scaled = (fit_tensor - mean) / std
        print_pca_progress(0.38, "torch_train_standardized")

        print_pca_progress(0.40, "torch_pca_fit_start")
        components, singular_values, actual_algorithm = fit_torch_pca_components(
            train_scaled=fit_scaled,
            n_components=n_components,
            pca_config=pca_config,
        )
        print_pca_progress(0.58, "torch_pca_fit_done")
        explained_variance_ratio = compute_torch_explained_variance_ratio(
            train_scaled=fit_scaled,
            singular_values=singular_values,
        )

        scores_by_split = {}
        if np.array_equal(
            np.asarray(pca_fit_indices, dtype=int),
            np.asarray(split_indices["train"], dtype=int),
        ):
            scores_by_split["train"] = (
                torch.matmul(fit_scaled, components)
                .cpu()
                .numpy()
                .astype(np.float32, copy=False)
            )
        else:
            scores_by_split["train"] = transform_torch_pca_split(
                X=X,
                indices=split_indices["train"],
                mean=mean,
                std=std,
                components=components,
                feature_config=feature_config,
                device=device,
                progress_start=0.58,
                progress_end=0.62,
                progress_stage="torch_train_transform_feature_matrix",
            )
        print_pca_progress(0.62, "torch_train_scores_done")
        transform_ranges = {
            "validation": (0.62, 0.72),
            "test": (0.72, 0.82),
        }
        for split_name, indices in split_indices.items():
            if split_name == "train":
                continue
            split_start, split_end = transform_ranges.get(split_name, (0.62, 0.82))
            scores_by_split[split_name] = transform_torch_pca_split(
                X=X,
                indices=indices,
                mean=mean,
                std=std,
                components=components,
                feature_config=feature_config,
                device=device,
                progress_start=split_start,
                progress_end=split_end,
                progress_stage=f"torch_{split_name}_feature_matrix",
            )
        print_pca_progress(0.84, "torch_transform_done")

    return {
        "scores_by_split": scores_by_split,
        "explained_variance_ratio": explained_variance_ratio,
        "summary": {
            "backend": "torch",
            "algorithm": actual_algorithm,
            "device": str(device),
            "torch_version": torch.__version__,
            "oversamples": int(pca_config["oversamples"]),
            "niter": int(pca_config["niter"]),
            "fit_sample_count": len(pca_fit_indices),
        },
    }


def fit_transform_torch_pca_multi_device(
    X,
    split_indices,
    pca_fit_indices,
    n_components,
    feature_config,
    pca_config,
    devices,
):
    """
    Fit randomized PCA with PyTorch split across multiple CUDA devices.

    Parameters
    ----------
    X : numpy.ndarray
        Memory-mapped VDF samples.
    split_indices : dict
        Mapping from split name to sample indices.
    pca_fit_indices : numpy.ndarray
        Training sample indices used to fit PCA.
    n_components : int
        Number of PCA components.
    feature_config : dict
        Feature extraction settings.
    pca_config : dict
        PCA backend settings.
    devices : list of torch.device
        CUDA devices used for row-wise matrix operations.

    Returns
    -------
    dict
        PCA scores, explained variance ratios, and backend summary.
    """

    fit_features = create_feature_matrix(
        X=X,
        indices=pca_fit_indices,
        feature_config=feature_config,
        progress_start=0.10,
        progress_end=0.35,
        progress_stage="torch_train_feature_matrix",
    )
    mean, std, nonconstant_feature_count = compute_feature_mean_std(fit_features)
    print_pca_progress(0.38, "torch_train_standardized")

    print_pca_progress(0.40, "torch_multi_gpu_pca_fit_start")
    components, singular_values, actual_algorithm = fit_multi_device_lowrank_pca(
        features=fit_features,
        mean=mean,
        std=std,
        n_components=n_components,
        pca_config=pca_config,
        devices=devices,
    )
    print_pca_progress(0.58, "torch_multi_gpu_pca_fit_done")
    explained_variance_ratio = compute_standardized_explained_variance_ratio(
        singular_values=singular_values,
        n_samples=len(fit_features),
        nonconstant_feature_count=nonconstant_feature_count,
    )

    scores_by_split = {}
    if np.array_equal(
        np.asarray(pca_fit_indices, dtype=int),
        np.asarray(split_indices["train"], dtype=int),
    ):
        scores_by_split["train"] = transform_feature_matrix_multi_device(
            features=fit_features,
            mean=mean,
            std=std,
            components=components,
            devices=devices,
        )
    else:
        train_features = create_feature_matrix(
            X=X,
            indices=split_indices["train"],
            feature_config=feature_config,
            progress_start=0.58,
            progress_end=0.62,
            progress_stage="torch_train_transform_feature_matrix",
        )
        scores_by_split["train"] = transform_feature_matrix_multi_device(
            features=train_features,
            mean=mean,
            std=std,
            components=components,
            devices=devices,
        )
        del train_features
    print_pca_progress(0.62, "torch_train_scores_done")
    del fit_features

    transform_ranges = {
        "validation": (0.62, 0.72),
        "test": (0.72, 0.82),
    }
    for split_name, indices in split_indices.items():
        if split_name == "train":
            continue
        split_start, split_end = transform_ranges.get(split_name, (0.62, 0.82))
        features = create_feature_matrix(
            X=X,
            indices=indices,
            feature_config=feature_config,
            progress_start=split_start,
            progress_end=split_end,
            progress_stage=f"torch_{split_name}_feature_matrix",
        )
        scores_by_split[split_name] = transform_feature_matrix_multi_device(
            features=features,
            mean=mean,
            std=std,
            components=components,
            devices=devices,
        )
        del features
    print_pca_progress(0.84, "torch_transform_done")

    clear_cuda_memory(devices)

    return {
        "scores_by_split": scores_by_split,
        "explained_variance_ratio": explained_variance_ratio,
        "summary": {
            "backend": "torch",
            "algorithm": actual_algorithm,
            "device": format_torch_devices(devices),
            "torch_version": torch.__version__,
            "oversamples": int(pca_config["oversamples"]),
            "niter": int(pca_config["niter"]),
            "fit_sample_count": len(pca_fit_indices),
        },
    }


def fit_torch_pca_components(train_scaled, n_components, pca_config):
    """
    Fit PCA components from a scaled training tensor.

    Parameters
    ----------
    train_scaled : torch.Tensor
        Standardized training features.
    n_components : int
        Number of PCA components.
    pca_config : dict
        PCA backend settings.

    Returns
    -------
    components : torch.Tensor
        PCA component matrix with shape ``(n_features, n_components)``.
    singular_values : torch.Tensor
        Singular values for the selected components.
    algorithm : str
        Algorithm used for the fit.
    """

    algorithm = pca_config["algorithm"]
    if algorithm == "lowrank":
        q = min(
            int(n_components) + int(pca_config["oversamples"]),
            min(train_scaled.shape),
        )
        q = max(q, int(n_components))
        _, singular_values, components = torch.pca_lowrank(
            train_scaled,
            q=q,
            center=False,
            niter=int(pca_config["niter"]),
        )
        return (
            components[:, :n_components],
            singular_values[:n_components],
            "lowrank",
        )

    _, singular_values, vh = torch.linalg.svd(
        train_scaled,
        full_matrices=False,
    )
    return vh[:n_components].T, singular_values[:n_components], "exact"


def compute_torch_explained_variance_ratio(train_scaled, singular_values):
    """
    Compute explained variance ratio from selected PyTorch PCA singular values.

    Parameters
    ----------
    train_scaled : torch.Tensor
        Standardized training features.
    singular_values : torch.Tensor
        PCA singular values.

    Returns
    -------
    numpy.ndarray
        Explained variance ratio for selected components.
    """

    if train_scaled.shape[0] < 2:
        return np.zeros(len(singular_values), dtype=np.float64)

    explained_variance = singular_values.pow(2) / (train_scaled.shape[0] - 1)
    total_variance = train_scaled.var(dim=0, unbiased=True).sum()
    if float(total_variance.cpu()) <= 0.0:
        return np.zeros(len(singular_values), dtype=np.float64)

    return (
        (explained_variance / total_variance)
        .detach()
        .cpu()
        .numpy()
        .astype(np.float64, copy=False)
    )


def compute_feature_mean_std(features):
    """
    Compute feature mean and standard deviation for standardized PCA.

    Parameters
    ----------
    features : numpy.ndarray
        Training feature matrix.

    Returns
    -------
    mean : numpy.ndarray
        Per-feature mean.
    std : numpy.ndarray
        Per-feature standard deviation with zeros replaced by one.
    nonconstant_feature_count : int
        Number of features with non-zero standard deviation.
    """

    mean = features.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = features.std(axis=0, dtype=np.float64).astype(np.float32)
    nonconstant_feature_count = int(np.count_nonzero(std > 0.0))
    std[std <= 0.0] = 1.0

    return mean, std, nonconstant_feature_count


def fit_multi_device_lowrank_pca(
    features,
    mean,
    std,
    n_components,
    pca_config,
    devices,
):
    """
    Fit randomized PCA components with row chunks distributed across GPUs.

    Parameters
    ----------
    features : numpy.ndarray
        Training feature matrix.
    mean : numpy.ndarray
        Training feature mean.
    std : numpy.ndarray
        Training feature standard deviation.
    n_components : int
        Number of PCA components.
    pca_config : dict
        PCA backend settings.
    devices : list of torch.device
        CUDA devices used for row-wise matrix operations.

    Returns
    -------
    components : numpy.ndarray
        PCA component matrix with shape ``(n_features, n_components)``.
    singular_values : numpy.ndarray
        Singular values for the selected components.
    algorithm : str
        Algorithm used for the fit.
    """

    q = min(
        int(n_components) + int(pca_config["oversamples"]),
        min(features.shape),
    )
    q = max(q, int(n_components))
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(pca_config["random_state"]))
    omega = torch.randn(
        (features.shape[1], q),
        generator=generator,
        dtype=torch.float32,
    ).numpy()

    sample_basis = orthonormalize_cpu(
        matrix=distributed_feature_matmul(
            features=features,
            mean=mean,
            std=std,
            right_matrix=omega,
            devices=devices,
        )
    )
    for _ in range(int(pca_config["niter"])):
        feature_basis = distributed_feature_transpose_matmul(
            features=features,
            mean=mean,
            std=std,
            left_matrix=sample_basis,
            devices=devices,
        )
        sample_basis = orthonormalize_cpu(
            matrix=distributed_feature_matmul(
                features=features,
                mean=mean,
                std=std,
                right_matrix=feature_basis,
                devices=devices,
            )
        )

    reduced_matrix = distributed_left_feature_matmul(
        features=features,
        mean=mean,
        std=std,
        left_matrix=sample_basis,
        devices=devices,
    )
    _, singular_values, vh = torch.linalg.svd(
        torch.as_tensor(reduced_matrix, dtype=torch.float32),
        full_matrices=False,
    )
    components = vh[:n_components].T.contiguous().numpy().astype(
        np.float32,
        copy=False,
    )

    return (
        components,
        singular_values[:n_components].numpy().astype(np.float64, copy=False),
        "multi_gpu_lowrank",
    )


def compute_standardized_explained_variance_ratio(
    singular_values,
    n_samples,
    nonconstant_feature_count,
):
    """
    Compute explained variance ratio for standardized features.

    Parameters
    ----------
    singular_values : numpy.ndarray
        PCA singular values.
    n_samples : int
        Number of training samples.
    nonconstant_feature_count : int
        Number of nonconstant standardized features.

    Returns
    -------
    numpy.ndarray
        Explained variance ratio for selected components.
    """

    singular_values = np.asarray(singular_values, dtype=np.float64)
    if n_samples < 2 or nonconstant_feature_count <= 0:
        return np.zeros(len(singular_values), dtype=np.float64)

    explained_variance = singular_values ** 2 / (int(n_samples) - 1)
    total_variance = (
        float(nonconstant_feature_count) * float(n_samples) / float(n_samples - 1)
    )
    if total_variance <= 0.0:
        return np.zeros(len(singular_values), dtype=np.float64)

    return (explained_variance / total_variance).astype(np.float64, copy=False)


def transform_feature_matrix_multi_device(features, mean, std, components, devices):
    """
    Transform features into PCA scores with row chunks distributed across GPUs.

    Parameters
    ----------
    features : numpy.ndarray
        Feature matrix to transform.
    mean : numpy.ndarray
        Training feature mean.
    std : numpy.ndarray
        Training feature standard deviation.
    components : numpy.ndarray
        PCA component matrix.
    devices : list of torch.device
        CUDA devices used for row-wise matrix operations.

    Returns
    -------
    numpy.ndarray
        PCA scores.
    """

    if len(features) == 0:
        return np.empty((0, components.shape[1]), dtype=np.float32)

    return distributed_feature_matmul(
        features=features,
        mean=mean,
        std=std,
        right_matrix=components,
        devices=devices,
    ).astype(np.float32, copy=False)


def distributed_feature_matmul(features, mean, std, right_matrix, devices):
    """
    Compute ``standardize(features) @ right_matrix`` across row chunks.

    Parameters
    ----------
    features : numpy.ndarray
        Feature matrix.
    mean : numpy.ndarray
        Feature mean.
    std : numpy.ndarray
        Feature standard deviation.
    right_matrix : numpy.ndarray
        Right-hand matrix.
    devices : list of torch.device
        Devices used for row chunks.

    Returns
    -------
    numpy.ndarray
        Matrix product.
    """

    chunks = create_device_row_chunks(n_rows=len(features), devices=devices)
    results = Parallel(n_jobs=len(chunks), prefer="threads")(
        delayed(compute_feature_matmul_chunk)(
            features=features,
            start=start,
            stop=stop,
            mean=mean,
            std=std,
            right_matrix=right_matrix,
            device_name=str(device),
        )
        for start, stop, device in chunks
    )
    output = np.empty((len(features), right_matrix.shape[1]), dtype=np.float32)
    for start, result in results:
        output[start:start + len(result)] = result

    return output


def distributed_feature_transpose_matmul(features, mean, std, left_matrix, devices):
    """
    Compute ``standardize(features).T @ left_matrix`` across row chunks.

    Parameters
    ----------
    features : numpy.ndarray
        Feature matrix.
    mean : numpy.ndarray
        Feature mean.
    std : numpy.ndarray
        Feature standard deviation.
    left_matrix : numpy.ndarray
        Left-hand matrix with one row per feature row.
    devices : list of torch.device
        Devices used for row chunks.

    Returns
    -------
    numpy.ndarray
        Matrix product.
    """

    chunks = create_device_row_chunks(n_rows=len(features), devices=devices)
    partials = Parallel(n_jobs=len(chunks), prefer="threads")(
        delayed(compute_feature_transpose_matmul_chunk)(
            features=features,
            start=start,
            stop=stop,
            mean=mean,
            std=std,
            left_matrix=left_matrix,
            device_name=str(device),
        )
        for start, stop, device in chunks
    )
    output = np.zeros((features.shape[1], left_matrix.shape[1]), dtype=np.float32)
    for partial in partials:
        output += partial

    return output


def distributed_left_feature_matmul(features, mean, std, left_matrix, devices):
    """
    Compute ``left_matrix.T @ standardize(features)`` across row chunks.

    Parameters
    ----------
    features : numpy.ndarray
        Feature matrix.
    mean : numpy.ndarray
        Feature mean.
    std : numpy.ndarray
        Feature standard deviation.
    left_matrix : numpy.ndarray
        Left-hand matrix with one row per feature row.
    devices : list of torch.device
        Devices used for row chunks.

    Returns
    -------
    numpy.ndarray
        Matrix product.
    """

    chunks = create_device_row_chunks(n_rows=len(features), devices=devices)
    partials = Parallel(n_jobs=len(chunks), prefer="threads")(
        delayed(compute_left_feature_matmul_chunk)(
            features=features,
            start=start,
            stop=stop,
            mean=mean,
            std=std,
            left_matrix=left_matrix,
            device_name=str(device),
        )
        for start, stop, device in chunks
    )
    output = np.zeros((left_matrix.shape[1], features.shape[1]), dtype=np.float32)
    for partial in partials:
        output += partial

    return output


def compute_feature_matmul_chunk(
    features,
    start,
    stop,
    mean,
    std,
    right_matrix,
    device_name,
):
    """
    Compute one row chunk of ``standardize(features) @ right_matrix``.

    Parameters
    ----------
    features : numpy.ndarray
        Feature matrix.
    start : int
        First row in the chunk.
    stop : int
        Row after the final row in the chunk.
    mean : numpy.ndarray
        Feature mean.
    std : numpy.ndarray
        Feature standard deviation.
    right_matrix : numpy.ndarray
        Right-hand matrix.
    device_name : str
        Torch device name.

    Returns
    -------
    tuple
        Chunk start and product array.
    """

    device = torch.device(device_name)
    with torch.no_grad():
        set_cuda_device(device)
        feature_tensor = torch.as_tensor(
            features[start:stop],
            dtype=torch.float32,
            device=device,
        )
        mean_tensor = torch.as_tensor(mean, dtype=torch.float32, device=device)
        std_tensor = torch.as_tensor(std, dtype=torch.float32, device=device)
        right_tensor = torch.as_tensor(
            right_matrix,
            dtype=torch.float32,
            device=device,
        )
        product = torch.matmul(
            (feature_tensor - mean_tensor) / std_tensor,
            right_tensor,
        )
        result = product.detach().cpu().numpy().astype(np.float32, copy=False)
        del feature_tensor, mean_tensor, std_tensor, right_tensor, product
        clear_cuda_memory([device])

    return int(start), result


def compute_feature_transpose_matmul_chunk(
    features,
    start,
    stop,
    mean,
    std,
    left_matrix,
    device_name,
):
    """
    Compute one chunk of ``standardize(features).T @ left_matrix``.

    Parameters
    ----------
    features : numpy.ndarray
        Feature matrix.
    start : int
        First row in the chunk.
    stop : int
        Row after the final row in the chunk.
    mean : numpy.ndarray
        Feature mean.
    std : numpy.ndarray
        Feature standard deviation.
    left_matrix : numpy.ndarray
        Left-hand matrix.
    device_name : str
        Torch device name.

    Returns
    -------
    numpy.ndarray
        Partial product.
    """

    device = torch.device(device_name)
    with torch.no_grad():
        set_cuda_device(device)
        feature_tensor = torch.as_tensor(
            features[start:stop],
            dtype=torch.float32,
            device=device,
        )
        mean_tensor = torch.as_tensor(mean, dtype=torch.float32, device=device)
        std_tensor = torch.as_tensor(std, dtype=torch.float32, device=device)
        left_tensor = torch.as_tensor(
            left_matrix[start:stop],
            dtype=torch.float32,
            device=device,
        )
        product = torch.matmul(
            ((feature_tensor - mean_tensor) / std_tensor).T,
            left_tensor,
        )
        result = product.detach().cpu().numpy().astype(np.float32, copy=False)
        del feature_tensor, mean_tensor, std_tensor, left_tensor, product
        clear_cuda_memory([device])

    return result


def compute_left_feature_matmul_chunk(
    features,
    start,
    stop,
    mean,
    std,
    left_matrix,
    device_name,
):
    """
    Compute one chunk of ``left_matrix.T @ standardize(features)``.

    Parameters
    ----------
    features : numpy.ndarray
        Feature matrix.
    start : int
        First row in the chunk.
    stop : int
        Row after the final row in the chunk.
    mean : numpy.ndarray
        Feature mean.
    std : numpy.ndarray
        Feature standard deviation.
    left_matrix : numpy.ndarray
        Left-hand matrix.
    device_name : str
        Torch device name.

    Returns
    -------
    numpy.ndarray
        Partial product.
    """

    device = torch.device(device_name)
    with torch.no_grad():
        set_cuda_device(device)
        feature_tensor = torch.as_tensor(
            features[start:stop],
            dtype=torch.float32,
            device=device,
        )
        mean_tensor = torch.as_tensor(mean, dtype=torch.float32, device=device)
        std_tensor = torch.as_tensor(std, dtype=torch.float32, device=device)
        left_tensor = torch.as_tensor(
            left_matrix[start:stop],
            dtype=torch.float32,
            device=device,
        )
        product = torch.matmul(
            left_tensor.T,
            (feature_tensor - mean_tensor) / std_tensor,
        )
        result = product.detach().cpu().numpy().astype(np.float32, copy=False)
        del feature_tensor, mean_tensor, std_tensor, left_tensor, product
        clear_cuda_memory([device])

    return result


def orthonormalize_cpu(matrix):
    """
    Orthonormalize columns of a small dense matrix on CPU.

    Parameters
    ----------
    matrix : numpy.ndarray
        Matrix to orthonormalize.

    Returns
    -------
    numpy.ndarray
        Matrix with orthonormal columns.
    """

    q_matrix, _ = torch.linalg.qr(
        torch.as_tensor(matrix, dtype=torch.float32),
        mode="reduced",
    )

    return q_matrix.numpy().astype(np.float32, copy=False)


def create_device_row_chunks(n_rows, devices):
    """
    Split rows as evenly as possible across devices.

    Parameters
    ----------
    n_rows : int
        Number of matrix rows.
    devices : list of torch.device
        Devices used for row chunks.

    Returns
    -------
    list of tuple
        ``(start, stop, device)`` chunks.
    """

    n_rows = int(n_rows)
    if n_rows == 0:
        return []

    boundaries = np.linspace(
        0,
        n_rows,
        min(len(devices), n_rows) + 1,
        dtype=int,
    )

    return [
        (int(boundaries[index]), int(boundaries[index + 1]), devices[index])
        for index in range(len(boundaries) - 1)
        if boundaries[index] < boundaries[index + 1]
    ]


def transform_torch_pca_split(
    X,
    indices,
    mean,
    std,
    components,
    feature_config,
    device,
    progress_start=None,
    progress_end=None,
    progress_stage=None,
):
    """
    Transform one split with a fitted PyTorch PCA basis.

    Parameters
    ----------
    X : numpy.ndarray
        Memory-mapped VDF samples.
    indices : numpy.ndarray
        Sample indices.
    mean : torch.Tensor
        Training feature mean.
    std : torch.Tensor
        Training feature standard deviation.
    components : torch.Tensor
        PCA components.
    feature_config : dict
        Feature extraction settings.
    device : torch.device
        Device used for PCA projection.
    progress_start : float, optional
        Progress value at the start of feature materialization.
    progress_end : float, optional
        Progress value at the end of feature materialization.
    progress_stage : str, optional
        Stage name for progress logging.

    Returns
    -------
    numpy.ndarray
        PCA scores.
    """

    features = create_feature_matrix(
        X=X,
        indices=indices,
        feature_config=feature_config,
        progress_start=progress_start,
        progress_end=progress_end,
        progress_stage=progress_stage,
    )
    if len(features) == 0:
        return np.empty((0, components.shape[1]), dtype=np.float32)

    tensor = torch.as_tensor(features, dtype=torch.float32, device=device)
    scores = torch.matmul((tensor - mean) / std, components)

    return scores.cpu().numpy().astype(np.float32, copy=False)


def resolve_torch_devices(device_name):
    """
    Resolve PyTorch devices used for PCA.

    Parameters
    ----------
    device_name : str
        Configured device name, ``"auto"``, ``"cuda:all"``, or comma-separated
        device names.

    Returns
    -------
    list of torch.device
        Resolved PyTorch devices.
    """

    device_name = str(device_name).strip()
    normalized = device_name.lower()
    if normalized == "auto":
        devices = get_available_cuda_devices()
        return devices if devices else [torch.device("cpu")]

    if normalized in {"cuda:all", "all_cuda", "all"}:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA devices were requested but CUDA is not available")
        return [
            torch.device(f"cuda:{device_index}")
            for device_index in range(torch.cuda.device_count())
        ]

    if "," in device_name:
        devices = [
            torch.device(part.strip())
            for part in device_name.split(",")
            if part.strip()
        ]
    else:
        devices = [torch.device(device_name)]

    if any(device.type == "cuda" for device in devices) and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for PCA but is not available")

    return devices


def get_available_cuda_devices(min_free_memory_gib=1.0):
    """
    Return visible CUDA devices with at least a small amount of free memory.

    Parameters
    ----------
    min_free_memory_gib : float, optional
        Minimum free memory in GiB required to use a GPU.

    Returns
    -------
    list of torch.device
        Available CUDA devices.
    """

    if not torch.cuda.is_available():
        return []

    devices = []
    min_free_bytes = int(float(min_free_memory_gib) * 1024 ** 3)
    for device_index in range(torch.cuda.device_count()):
        try:
            free_bytes, _ = torch.cuda.mem_get_info(device_index)
        except TypeError:
            try:
                torch.cuda.set_device(device_index)
                free_bytes, _ = torch.cuda.mem_get_info()
            except RuntimeError:
                continue
        except RuntimeError:
            continue
        if int(free_bytes) >= min_free_bytes:
            devices.append(torch.device(f"cuda:{device_index}"))

    return devices


def format_torch_devices(devices):
    """
    Format a list of torch devices for logs and metrics.

    Parameters
    ----------
    devices : list of torch.device
        Devices used by PCA.

    Returns
    -------
    str
        Comma-separated device names.
    """

    return ", ".join(str(device) for device in devices)


def set_cuda_device(device):
    """
    Set the current CUDA device for a worker thread.

    Parameters
    ----------
    device : torch.device
        Device used by the worker.
    """

    if device.type == "cuda":
        torch.cuda.set_device(device)


def clear_cuda_memory(devices):
    """
    Release cached CUDA memory for the given devices.

    Parameters
    ----------
    devices : list of torch.device
        Devices whose CUDA caches are cleared.
    """

    for device in devices:
        if device.type != "cuda":
            continue
        try:
            torch.cuda.set_device(device)
            torch.cuda.empty_cache()
        except RuntimeError:
            continue


def create_feature_matrix(
    X,
    indices,
    feature_config,
    progress_start=None,
    progress_end=None,
    progress_stage=None,
):
    """
    Create a feature matrix for sample indices.

    Parameters
    ----------
    X : numpy.ndarray
        Memory-mapped VDF samples.
    indices : array-like of int
        Sample indices.
    feature_config : dict
        Feature extraction settings.
    progress_start : float, optional
        Progress value at the start of feature creation.
    progress_end : float, optional
        Progress value at the end of feature creation.
    progress_stage : str, optional
        Stage name for progress logging.

    Returns
    -------
    numpy.ndarray
        Feature matrix.
    """

    indices = np.asarray(indices, dtype=int)
    feature_count = get_feature_count(
        slice_shape=(int(X.shape[1]), int(X.shape[3])),
        downsample_factor=feature_config["downsample_factor"],
    )
    if len(indices) == 0:
        return np.empty((0, feature_count), dtype=np.float32)

    features = np.empty((len(indices), feature_count), dtype=np.float32)
    write_start = 0
    n_batches = count_index_batches(
        indices=indices,
        batch_size=feature_config["batch_size"],
    )
    for batch_number, feature_batch in enumerate(iter_feature_batches(
        X=X,
        indices=indices,
        batch_size=feature_config["batch_size"],
        downsample_factor=feature_config["downsample_factor"],
        n_jobs=feature_config["n_jobs"],
        log_eps=feature_config["log_eps"],
        sample_normalization=feature_config["sample_normalization"],
        sample_norm_eps=feature_config["sample_norm_eps"],
    ), start=1):
        write_end = write_start + len(feature_batch)
        features[write_start:write_end] = feature_batch
        write_start = write_end
        if progress_stage is not None:
            print_batch_progress(
                stage=progress_stage,
                batch_number=batch_number,
                n_batches=n_batches,
                start=progress_start,
                end=progress_end,
            )

    return features


def create_sample_metrics(y, metadata, split_indices, scores_by_split):
    """
    Create per-sample PCA metric rows.

    Parameters
    ----------
    y : numpy.ndarray
        Dataset labels.
    metadata : pandas.DataFrame
        Dataset metadata.
    split_indices : dict
        Mapping from split name to sample indices.
    scores_by_split : dict
        Mapping from split name to PCA scores.

    Returns
    -------
    pandas.DataFrame
        Per-sample PCA metrics.
    """

    rows = []
    for split_name, indices in split_indices.items():
        scores = scores_by_split[split_name]
        split_rows = metadata.iloc[indices].copy()
        split_rows["split"] = split_name
        split_rows["label"] = np.asarray(y[indices], dtype=int)
        split_rows["pca_0"] = scores[:, 0] if scores.shape[1] >= 1 else 0.0
        split_rows["pca_1"] = scores[:, 1] if scores.shape[1] >= 2 else 0.0
        rows.append(split_rows)

    return pd.concat(rows, ignore_index=True)


def add_neighbor_metrics(
    sample_metrics,
    scores_by_split,
    y,
    train_indices,
    neighbor_config,
):
    """
    Add nearest-neighbor label-purity metrics to sample rows.

    Parameters
    ----------
    sample_metrics : pandas.DataFrame
        Per-sample PCA metric rows.
    scores_by_split : dict
        Mapping from split name to PCA scores.
    y : numpy.ndarray
        Dataset labels.
    train_indices : numpy.ndarray
        Training sample indices.
    neighbor_config : dict
        Neighbor metric settings.
    """

    train_scores = scores_by_split["train"]
    train_labels = np.asarray(y[train_indices], dtype=int)
    if len(train_scores) < 2:
        sample_metrics["same_class_fraction"] = np.nan
        sample_metrics["nearest_other_class"] = ""
        return

    label_names = create_label_names(sample_metrics)
    k_neighbors = min(int(neighbor_config["k_neighbors"]), len(train_scores) - 1)
    neighbor_model = NearestNeighbors(n_neighbors=k_neighbors + 1)
    neighbor_model.fit(train_scores)

    sample_metrics["same_class_fraction"] = np.nan
    sample_metrics["nearest_other_class"] = ""

    split_offsets = create_split_offsets(sample_metrics)
    for split_name, scores in scores_by_split.items():
        split_mask = sample_metrics["split"].to_numpy() == split_name
        row_indices = np.flatnonzero(split_mask)
        if len(row_indices) == 0:
            continue

        if split_name == "train":
            query_train_positions = np.arange(len(scores), dtype=int)
        else:
            query_train_positions = None

        same_fractions, nearest_other = compute_neighbor_purity(
            neighbor_model=neighbor_model,
            query_scores=scores,
            query_labels=sample_metrics.loc[row_indices, "label"].to_numpy(dtype=int),
            train_labels=train_labels,
            label_names=label_names,
            k_neighbors=k_neighbors,
            batch_size=neighbor_config["batch_size"],
            query_train_positions=query_train_positions,
        )

        start, stop = split_offsets[split_name]
        sample_metrics.loc[start:stop - 1, "same_class_fraction"] = same_fractions
        sample_metrics.loc[start:stop - 1, "nearest_other_class"] = nearest_other


def add_filter_preview_metrics(
    sample_metrics,
    scores_by_split,
    y,
    train_indices,
    neighbor_config,
    filter_preview_config,
):
    """
    Add PCA-space preview markers for possible future dataset filtering.

    Parameters
    ----------
    sample_metrics : pandas.DataFrame
        Per-sample PCA metric rows.
    scores_by_split : dict
        Mapping from split name to PCA scores.
    y : numpy.ndarray
        Dataset labels.
    train_indices : numpy.ndarray
        Training sample indices.
    neighbor_config : dict
        Neighbor metric settings used for PCA-space searches.
    filter_preview_config : dict
        Filter-preview settings.
    """

    if "class_name" not in sample_metrics.columns:
        raise ValueError("filter_preview requires a class_name column in metadata")

    sample_metrics["point_neighbor_fraction"] = np.nan
    sample_metrics["nearest_point_class"] = ""
    sample_metrics["filter_preview_candidate"] = False
    sample_metrics["filter_preview_reason"] = ""

    train_scores = scores_by_split["train"]
    train_labels = np.asarray(y[train_indices], dtype=int)
    if len(train_scores) < 2:
        return

    label_names = create_label_names(sample_metrics)
    point_labels = get_labels_for_class_names(
        label_names=label_names,
        class_names=filter_preview_config["point_neighbor_classes"],
    )
    if len(point_labels) == 0:
        return

    k_neighbors = min(int(neighbor_config["k_neighbors"]), len(train_scores) - 1)
    neighbor_model = NearestNeighbors(n_neighbors=k_neighbors + 1)
    neighbor_model.fit(train_scores)

    split_offsets = create_split_offsets(sample_metrics)
    apply_splits = set(filter_preview_config["apply_splits"])
    for split_name, scores in scores_by_split.items():
        if split_name not in apply_splits:
            continue

        split_mask = sample_metrics["split"].to_numpy() == split_name
        row_indices = np.flatnonzero(split_mask)
        if len(row_indices) == 0:
            continue

        if split_name == "train":
            query_train_positions = np.arange(len(scores), dtype=int)
        else:
            query_train_positions = None

        point_fractions, nearest_point_classes = compute_point_neighbor_metrics(
            neighbor_model=neighbor_model,
            query_scores=scores,
            train_labels=train_labels,
            point_labels=point_labels,
            label_names=label_names,
            k_neighbors=k_neighbors,
            batch_size=neighbor_config["batch_size"],
            query_train_positions=query_train_positions,
        )

        start, stop = split_offsets[split_name]
        sample_metrics.loc[start:stop - 1, "point_neighbor_fraction"] = (
            point_fractions
        )
        sample_metrics.loc[start:stop - 1, "nearest_point_class"] = (
            nearest_point_classes
        )

    candidate_mask = create_filter_preview_candidate_mask(
        sample_metrics=sample_metrics,
        filter_preview_config=filter_preview_config,
    )
    if candidate_mask.any():
        sample_metrics.loc[candidate_mask, "filter_preview_candidate"] = True
        sample_metrics.loc[candidate_mask, "filter_preview_reason"] = (
            create_filter_preview_reasons(
                rows=sample_metrics.loc[candidate_mask],
                filter_preview_config=filter_preview_config,
            )
        )


def save_pca_plots(
    sample_metrics,
    output_dir,
    plot_config,
    filter_preview_config=None,
    embedding_plot_config=None,
    scores_by_split=None,
    split_indices=None,
):
    """
    Save PCA scatter plots.

    Parameters
    ----------
    sample_metrics : pandas.DataFrame
        Per-sample PCA metric rows.
    output_dir : str or pathlib.Path
        Directory where plots are saved.
    plot_config : dict
        Plot settings.
    filter_preview_config : dict, optional
        Filter-preview settings.
    embedding_plot_config : dict, optional
        Embedding plot settings.
    scores_by_split : dict, optional
        Mapping from split name to PCA scores.
    split_indices : dict, optional
        Mapping from split name to sample indices.

    Returns
    -------
    list of pathlib.Path
        Saved plot paths.
    """

    output_dir = Path(output_dir)
    plot_rows = subsample_plot_rows(
        sample_metrics=sample_metrics,
        max_points_per_class=plot_config["max_points_per_class"],
        random_state=plot_config["random_state"],
    )

    plot_paths = []
    train_rows = plot_rows[plot_rows["split"] == "train"]
    train_by_class_path = output_dir / "pca_train_by_class.png"
    if plot_pca_by_class(
        rows=train_rows,
        output_path=train_by_class_path,
        title="Training PCA by class",
        alpha=plot_config["alpha"],
        point_size=plot_config["point_size"],
    ):
        plot_paths.append(train_by_class_path)

    all_by_class_path = output_dir / "pca_all_by_class.png"
    if plot_pca_by_class(
        rows=plot_rows,
        output_path=all_by_class_path,
        title="PCA by class",
        alpha=plot_config["alpha"],
        point_size=plot_config["point_size"],
    ):
        plot_paths.append(all_by_class_path)

    purity_path = output_dir / "pca_train_neighbor_purity.png"
    if "same_class_fraction" in train_rows.columns and plot_pca_neighbor_purity(
        rows=train_rows,
        output_path=purity_path,
        alpha=plot_config["alpha"],
        point_size=plot_config["point_size"],
    ):
        plot_paths.append(purity_path)

    if (
        filter_preview_config is not None
        and filter_preview_config["enabled"]
        and "filter_preview_candidate" in plot_rows.columns
    ):
        preview_rows = plot_rows[
            plot_rows["split"].isin(filter_preview_config["apply_splits"])
        ]
        preview_candidate_rows = sample_metrics[
            sample_metrics["split"].isin(filter_preview_config["apply_splits"])
            & sample_metrics["filter_preview_candidate"]
        ]
        preview_rows = pd.concat(
            [preview_rows, preview_candidate_rows],
            ignore_index=True,
        ).drop_duplicates()
        preview_path = output_dir / "pca_filter_preview.png"
        if plot_pca_filter_preview(
            rows=preview_rows,
            output_path=preview_path,
            filter_preview_config=filter_preview_config,
            alpha=plot_config["alpha"],
            point_size=plot_config["point_size"],
        ):
            plot_paths.append(preview_path)

    if (
        embedding_plot_config is not None
        and embedding_plot_config["enabled"]
        and scores_by_split is not None
        and split_indices is not None
    ):
        embedding_path = output_dir / (
            f"pca_{embedding_plot_config['method']}_by_class.png"
        )
        if plot_pca_embedding(
            sample_metrics=sample_metrics,
            scores_by_split=scores_by_split,
            split_indices=split_indices,
            output_path=embedding_path,
            embedding_plot_config=embedding_plot_config,
            alpha=plot_config["alpha"],
            point_size=plot_config["point_size"],
        ):
            plot_paths.append(embedding_path)

    return plot_paths


def save_pca_outputs(
    output_dir,
    sample_metrics,
    scores_by_split,
    split_indices,
    explained_variance_ratio,
    pca_summary,
    metrics_text,
):
    """
    Save PCA metrics, scores, and text summary.

    Parameters
    ----------
    output_dir : str or pathlib.Path
        Directory where PCA outputs are saved.
    sample_metrics : pandas.DataFrame
        Per-sample PCA metric rows.
    scores_by_split : dict
        Mapping from split name to PCA scores.
    split_indices : dict
        Mapping from split name to sample indices.
    explained_variance_ratio : numpy.ndarray
        Explained variance ratio for PCA components.
    pca_summary : dict
        PCA backend summary.
    metrics_text : str
        Text summary.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sample_metrics.to_csv(output_dir / "pca_sample_metrics.csv", index=False)

    scores = np.concatenate(
        [scores_by_split[split_name] for split_name in split_indices],
        axis=0,
    )
    sample_indices = np.concatenate(
        [np.asarray(split_indices[split_name], dtype=int) for split_name in split_indices]
    )
    labels = sample_metrics["label"].to_numpy(dtype=int)
    splits = sample_metrics["split"].to_numpy(dtype=str)
    explained_variance_ratio = np.asarray(explained_variance_ratio, dtype=np.float64)

    score_data = {
        "scores": np.asarray(scores, dtype=np.float32),
        "sample_indices": sample_indices,
        "labels": labels,
        "splits": splits,
        "explained_variance_ratio": explained_variance_ratio,
        "cumulative_explained_variance_ratio": np.cumsum(explained_variance_ratio),
        "pca_backend": np.asarray(str(pca_summary.get("backend", ""))),
        "pca_algorithm": np.asarray(str(pca_summary.get("algorithm", ""))),
        "pca_device": np.asarray(str(pca_summary.get("device", ""))),
    }
    if "point_neighbor_fraction" in sample_metrics.columns:
        score_data["point_neighbor_fraction"] = sample_metrics[
            "point_neighbor_fraction"
        ].to_numpy(dtype=np.float32)
    if "filter_preview_candidate" in sample_metrics.columns:
        score_data["filter_preview_candidate"] = sample_metrics[
            "filter_preview_candidate"
        ].to_numpy(dtype=bool)

    np.savez(output_dir / "pca_scores.npz", **score_data)

    with open(output_dir / "pca_metrics.txt", "w") as metrics_file:
        metrics_file.write(metrics_text)


def transform_indices(
    X,
    indices,
    scaler,
    pca,
    feature_config,
    progress_start=None,
    progress_end=None,
    progress_stage=None,
):
    """
    Transform sample indices into PCA scores.

    Parameters
    ----------
    X : numpy.ndarray
        Memory-mapped VDF samples.
    indices : numpy.ndarray
        Sample indices.
    scaler : sklearn.preprocessing.StandardScaler
        Scaler fitted on training features.
    pca : sklearn.decomposition.IncrementalPCA
        PCA model fitted on training features.
    feature_config : dict
        Feature extraction settings.
    progress_start : float, optional
        Progress value at the start of the transform.
    progress_end : float, optional
        Progress value at the end of the transform.
    progress_stage : str, optional
        Stage name for progress logging.

    Returns
    -------
    numpy.ndarray
        PCA scores.
    """

    scores = np.empty((len(indices), int(pca.n_components_)), dtype=np.float32)
    write_start = 0
    n_batches = count_index_batches(
        indices=indices,
        batch_size=feature_config["batch_size"],
    )
    for batch_number, features in enumerate(iter_feature_batches(
        X=X,
        indices=indices,
        batch_size=feature_config["batch_size"],
        downsample_factor=feature_config["downsample_factor"],
        n_jobs=feature_config["n_jobs"],
        log_eps=feature_config["log_eps"],
        sample_normalization=feature_config["sample_normalization"],
        sample_norm_eps=feature_config["sample_norm_eps"],
    ), start=1):
        scaled_features = scaler.transform(features)
        score_batch = pca.transform(scaled_features).astype(np.float32, copy=False)
        write_end = write_start + len(score_batch)
        scores[write_start:write_end] = score_batch
        write_start = write_end
        if progress_stage is not None:
            print_batch_progress(
                stage=progress_stage,
                batch_number=batch_number,
                n_batches=n_batches,
                start=progress_start,
                end=progress_end,
            )

    return scores


def iter_feature_batches(
    X,
    indices,
    batch_size,
    downsample_factor,
    n_jobs,
    log_eps,
    sample_normalization,
    sample_norm_eps,
    min_batch_size=None,
):
    """
    Iterate log-scaled xz-slice feature batches from a VDF memmap.

    Parameters
    ----------
    X : numpy.ndarray
        Memory-mapped VDF samples with shape ``(n_samples, vx, vy, vz)``.
    indices : array-like of int
        Sample indices.
    batch_size : int
        Number of samples per feature batch.
    downsample_factor : int
        Xz-slice downsampling factor.
    n_jobs : int
        Number of thread workers used inside each feature batch.
    log_eps : float
        Small value used for non-positive values before log scaling.
    sample_normalization : str
        Per-sample normalization mode.
    sample_norm_eps : float
        Small value used to avoid division by zero during sample normalization.
    min_batch_size : int, optional
        Minimum batch size. A short final batch is merged with the previous
        batch when possible.

    Yields
    ------
    numpy.ndarray
        Feature batch.
    """

    indices = np.asarray(indices, dtype=int)
    for batch_indices in iter_index_batches_with_min_size(
        indices=indices,
        batch_size=batch_size,
        min_batch_size=min_batch_size,
    ):
        yield create_log_xz_feature_batch(
            X=X,
            indices=batch_indices,
            downsample_factor=downsample_factor,
            n_jobs=n_jobs,
            log_eps=log_eps,
            sample_normalization=sample_normalization,
            sample_norm_eps=sample_norm_eps,
        )


def create_log_xz_feature_batch(
    X,
    indices,
    downsample_factor=1,
    n_jobs=1,
    log_eps=1e-30,
    sample_normalization="none",
    sample_norm_eps=1e-6,
):
    """
    Create flattened log-scaled xz-slice features for one batch.

    Parameters
    ----------
    X : numpy.ndarray
        Memory-mapped VDF samples with shape ``(n_samples, vx, vy, vz)``.
    indices : array-like of int
        Sample indices.
    downsample_factor : int, optional
        Xz-slice downsampling factor.
    n_jobs : int, optional
        Number of thread workers used inside the batch.
    log_eps : float, optional
        Small value used for non-positive values before log scaling.
    sample_normalization : str, optional
        Per-sample normalization mode.
    sample_norm_eps : float, optional
        Small value used to avoid division by zero during sample normalization.

    Returns
    -------
    numpy.ndarray
        Feature matrix.
    """

    indices = np.asarray(indices, dtype=int)
    n_jobs = int(n_jobs)
    if n_jobs == 1 or len(indices) <= 1:
        return create_log_xz_feature_batch_chunk(
            X=X,
            indices=indices,
            downsample_factor=downsample_factor,
            log_eps=log_eps,
            sample_normalization=sample_normalization,
            sample_norm_eps=sample_norm_eps,
        )

    worker_count = (os.cpu_count() or 1) if n_jobs < 0 else n_jobs
    worker_count = max(1, min(len(indices), int(worker_count)))
    index_chunks = np.array_split(indices, worker_count)

    feature_chunks = Parallel(
        n_jobs=n_jobs,
        prefer="threads",
    )(
        delayed(create_log_xz_feature_batch_chunk)(
            X=X,
            indices=index_chunk,
            downsample_factor=downsample_factor,
            log_eps=log_eps,
            sample_normalization=sample_normalization,
            sample_norm_eps=sample_norm_eps,
        )
        for index_chunk in index_chunks
        if len(index_chunk) > 0
    )

    return np.concatenate(feature_chunks, axis=0)


def create_log_xz_feature_batch_chunk(
    X,
    indices,
    downsample_factor=1,
    log_eps=1e-30,
    sample_normalization="none",
    sample_norm_eps=1e-6,
):
    """
    Create flattened log-scaled xz-slice features without reading full VDF cubes.

    Parameters
    ----------
    X : numpy.ndarray
        Memory-mapped VDF samples with shape ``(n_samples, vx, vy, vz)``.
    indices : array-like of int
        Sample indices.
    downsample_factor : int, optional
        Xz-slice downsampling factor.
    log_eps : float, optional
        Small value used for non-positive values before log scaling.
    sample_normalization : str, optional
        Per-sample normalization mode.
    sample_norm_eps : float, optional
        Small value used to avoid division by zero during sample normalization.

    Returns
    -------
    numpy.ndarray
        Feature matrix.
    """

    indices = np.asarray(indices, dtype=int)
    batch_slice = create_contiguous_slice(indices)
    sample_selection = batch_slice if batch_slice is not None else indices
    mid_y = X.shape[2] // 2

    slices = np.array(
        X[sample_selection, :, mid_y, :],
        dtype=np.float32,
        copy=True,
    )
    slices[slices <= 0.0] = float(log_eps)
    np.log10(slices, out=slices)

    slices = downsample_2d_batch(slices, factor=downsample_factor)

    features = slices.reshape(len(indices), -1)

    return normalize_feature_samples(
        features=features,
        sample_normalization=sample_normalization,
        sample_norm_eps=sample_norm_eps,
    )


def downsample_2d_batch(arrays, factor):
    """
    Downsample a batch of 2D arrays by block averaging.

    Parameters
    ----------
    arrays : numpy.ndarray
        Array batch with shape ``(n_samples, nx, ny)``.
    factor : int
        Downsampling factor.

    Returns
    -------
    numpy.ndarray
        Downsampled arrays.
    """

    factor = int(factor)
    if factor == 1:
        return arrays

    _, nx, ny = arrays.shape
    nx_trim = nx - (nx % factor)
    ny_trim = ny - (ny % factor)
    arrays = arrays[:, :nx_trim, :ny_trim]

    return arrays.reshape(
        arrays.shape[0],
        nx_trim // factor,
        factor,
        ny_trim // factor,
        factor,
    ).mean(axis=(2, 4), dtype=np.float32)


def normalize_feature_samples(features, sample_normalization, sample_norm_eps):
    """
    Normalize feature rows independently when configured.

    Parameters
    ----------
    features : numpy.ndarray
        Feature matrix.
    sample_normalization : str
        Per-sample normalization mode.
    sample_norm_eps : float
        Small value used to avoid division by zero.

    Returns
    -------
    numpy.ndarray
        Feature matrix after optional per-sample normalization.
    """

    if sample_normalization == "none":
        return features

    if sample_normalization != "standard":
        raise ValueError("sample_normalization must be 'none' or 'standard'")

    means = features.mean(axis=1, keepdims=True, dtype=np.float32)
    features = features - means
    stds = features.std(axis=1, keepdims=True, dtype=np.float32)
    stds[stds < float(sample_norm_eps)] = 1.0
    features = features / stds

    return features.astype(np.float32, copy=False)


def iter_index_batches_with_min_size(indices, batch_size, min_batch_size=None):
    """
    Iterate index batches, merging a short final batch into the previous batch.

    Parameters
    ----------
    indices : numpy.ndarray
        Sample indices.
    batch_size : int
        Target batch size.
    min_batch_size : int, optional
        Minimum yielded batch size when enough total samples are available.

    Yields
    ------
    numpy.ndarray
        Index batch.
    """

    indices = np.asarray(indices, dtype=int)
    if len(indices) == 0:
        return

    batch_size = int(batch_size)
    if min_batch_size is None or len(indices) <= min_batch_size:
        yield from iter_index_batches(indices, batch_size)
        return

    starts = list(range(0, len(indices), batch_size))
    if len(starts) >= 2 and len(indices) - starts[-1] < int(min_batch_size):
        starts.pop()

    for start_index, start in enumerate(starts):
        if start_index + 1 < len(starts):
            stop = starts[start_index + 1]
        else:
            stop = len(indices)
        yield indices[start:stop]


def count_index_batches(indices, batch_size, min_batch_size=None):
    """
    Count index batches produced by ``iter_index_batches_with_min_size``.

    Parameters
    ----------
    indices : array-like of int
        Sample indices.
    batch_size : int
        Target batch size.
    min_batch_size : int, optional
        Minimum yielded batch size when enough total samples are available.

    Returns
    -------
    int
        Number of yielded batches.
    """

    return sum(
        1
        for _ in iter_index_batches_with_min_size(
            indices=np.asarray(indices, dtype=int),
            batch_size=batch_size,
            min_batch_size=min_batch_size,
        )
    )


def get_feature_count(slice_shape, downsample_factor):
    """
    Return the flattened feature count after downsampling.

    Parameters
    ----------
    slice_shape : tuple of int
        Shape of one xz slice.
    downsample_factor : int
        Downsampling factor.

    Returns
    -------
    int
        Number of flattened features.
    """

    nx, ny = (int(value) for value in slice_shape)
    factor = int(downsample_factor)
    return (nx - nx % factor) // factor * ((ny - ny % factor) // factor)


def compute_neighbor_purity(
    neighbor_model,
    query_scores,
    query_labels,
    train_labels,
    label_names,
    k_neighbors,
    batch_size,
    query_train_positions=None,
):
    """
    Compute nearest-neighbor same-class fractions in PCA space.

    Parameters
    ----------
    neighbor_model : sklearn.neighbors.NearestNeighbors
        Model fitted on training PCA scores.
    query_scores : numpy.ndarray
        PCA scores to query.
    query_labels : numpy.ndarray
        Labels for query samples.
    train_labels : numpy.ndarray
        Labels for training samples.
    label_names : dict
        Mapping from label to class name.
    k_neighbors : int
        Number of neighbors used after excluding self matches.
    batch_size : int
        Query batch size.
    query_train_positions : numpy.ndarray, optional
        Training-row positions for train queries, used to remove self.

    Returns
    -------
    same_class_fraction : numpy.ndarray
        Fraction of neighbors with the same label.
    nearest_other_class : numpy.ndarray
        Most common different neighbor class name.
    """

    same_class_fraction = np.empty(len(query_scores), dtype=np.float32)
    nearest_other_class = np.empty(len(query_scores), dtype=object)

    for batch_indices in iter_index_batches(np.arange(len(query_scores)), batch_size):
        _, neighbor_indices = neighbor_model.kneighbors(query_scores[batch_indices])
        for row_offset, query_index in enumerate(batch_indices):
            neighbors = neighbor_indices[row_offset]
            if query_train_positions is not None:
                train_position = int(query_train_positions[query_index])
                neighbors = neighbors[neighbors != train_position]
            neighbors = neighbors[:k_neighbors]

            neighbor_labels = train_labels[neighbors]
            query_label = int(query_labels[query_index])
            same_class_fraction[query_index] = np.mean(
                neighbor_labels == query_label
            )
            nearest_other_class[query_index] = get_nearest_other_class(
                neighbor_labels=neighbor_labels,
                query_label=query_label,
                label_names=label_names,
            )

    return same_class_fraction, nearest_other_class


def compute_point_neighbor_metrics(
    neighbor_model,
    query_scores,
    train_labels,
    point_labels,
    label_names,
    k_neighbors,
    batch_size,
    query_train_positions=None,
):
    """
    Compute PCA-neighbor fractions belonging to point classes.

    Parameters
    ----------
    neighbor_model : sklearn.neighbors.NearestNeighbors
        Model fitted on training PCA scores.
    query_scores : numpy.ndarray
        PCA scores to query.
    train_labels : numpy.ndarray
        Labels for training samples.
    point_labels : numpy.ndarray
        Labels treated as point classes.
    label_names : dict
        Mapping from label to class name.
    k_neighbors : int
        Number of neighbors used after excluding self matches.
    batch_size : int
        Query batch size.
    query_train_positions : numpy.ndarray, optional
        Training-row positions for train queries, used to remove self.

    Returns
    -------
    point_neighbor_fraction : numpy.ndarray
        Fraction of neighbors whose labels are point labels.
    nearest_point_class : numpy.ndarray
        Nearest point class name, or an empty string.
    """

    point_labels = np.asarray(point_labels, dtype=int)
    point_neighbor_fraction = np.empty(len(query_scores), dtype=np.float32)
    nearest_point_class = np.empty(len(query_scores), dtype=object)

    for batch_indices in iter_index_batches(np.arange(len(query_scores)), batch_size):
        _, neighbor_indices = neighbor_model.kneighbors(query_scores[batch_indices])
        for row_offset, query_index in enumerate(batch_indices):
            neighbors = neighbor_indices[row_offset]
            if query_train_positions is not None:
                train_position = int(query_train_positions[query_index])
                neighbors = neighbors[neighbors != train_position]
            neighbors = neighbors[:k_neighbors]

            neighbor_labels = train_labels[neighbors]
            point_mask = np.isin(neighbor_labels, point_labels)
            point_neighbor_fraction[query_index] = np.mean(point_mask)
            nearest_point_class[query_index] = get_nearest_point_class(
                point_neighbor_labels=neighbor_labels[point_mask],
                label_names=label_names,
            )

    return point_neighbor_fraction, nearest_point_class


def get_nearest_other_class(neighbor_labels, query_label, label_names):
    """
    Return the most common neighbor class different from the query label.

    Parameters
    ----------
    neighbor_labels : numpy.ndarray
        Neighbor labels.
    query_label : int
        Query sample label.
    label_names : dict
        Mapping from label to class name.

    Returns
    -------
    str
        Most common different neighbor class, or an empty string.
    """

    other_labels = neighbor_labels[neighbor_labels != int(query_label)]
    if len(other_labels) == 0:
        return ""

    labels, counts = np.unique(other_labels, return_counts=True)
    label = int(labels[int(np.argmax(counts))])

    return str(label_names.get(label, label))


def get_nearest_point_class(point_neighbor_labels, label_names):
    """
    Return the nearest point-neighbor class name.

    Parameters
    ----------
    point_neighbor_labels : numpy.ndarray
        Point-neighbor labels in nearest-neighbor order.
    label_names : dict
        Mapping from label to class name.

    Returns
    -------
    str
        Nearest point class, or an empty string.
    """

    if len(point_neighbor_labels) == 0:
        return ""

    label = int(point_neighbor_labels[0])

    return str(label_names.get(label, label))


def create_label_names(sample_metrics):
    """
    Create class-name lookup from sample metrics.

    Parameters
    ----------
    sample_metrics : pandas.DataFrame
        Per-sample PCA metric rows.

    Returns
    -------
    dict
        Mapping from integer label to class name.
    """

    if "class_name" not in sample_metrics.columns:
        return {}

    rows = sample_metrics[["label", "class_name"]].drop_duplicates("label")
    return {
        int(row["label"]): str(row["class_name"])
        for _, row in rows.iterrows()
    }


def get_labels_for_class_names(label_names, class_names):
    """
    Return labels matching configured class names.

    Parameters
    ----------
    label_names : dict
        Mapping from integer label to class name.
    class_names : list of str
        Class names to find.

    Returns
    -------
    numpy.ndarray
        Integer labels for matching class names.
    """

    requested_class_names = {str(class_name) for class_name in class_names}

    return np.asarray(
        [
            int(label)
            for label, class_name in label_names.items()
            if str(class_name) in requested_class_names
        ],
        dtype=int,
    )


def create_filter_preview_candidate_mask(sample_metrics, filter_preview_config):
    """
    Create the boolean preview mask for possible future filtering.

    Parameters
    ----------
    sample_metrics : pandas.DataFrame
        Per-sample PCA metric rows.
    filter_preview_config : dict
        Filter-preview settings.

    Returns
    -------
    pandas.Series
        Boolean mask where rows are preview candidates.
    """

    apply_splits = set(filter_preview_config["apply_splits"])
    candidate_classes = set(filter_preview_config["candidate_classes"])
    protected_classes = set(filter_preview_config["protected_classes"])
    point_thresholds = create_class_threshold_series(
        rows=sample_metrics,
        default_threshold=filter_preview_config["min_point_neighbor_fraction"],
        class_thresholds=filter_preview_config[
            "min_point_neighbor_fraction_by_class"
        ],
    )

    candidate_mask = (
        sample_metrics["split"].isin(apply_splits)
        & sample_metrics["class_name"].isin(candidate_classes)
        & ~sample_metrics["class_name"].isin(protected_classes)
        & sample_metrics["point_neighbor_fraction"].ge(point_thresholds)
    )

    if has_same_class_filter(filter_preview_config):
        if "same_class_fraction" not in sample_metrics.columns:
            raise ValueError(
                "filter_preview max_same_class_fraction requires "
                "neighbor_metrics.enabled: true"
            )
        same_class_thresholds = create_optional_class_threshold_series(
            rows=sample_metrics,
            default_threshold=filter_preview_config["max_same_class_fraction"],
            class_thresholds=filter_preview_config[
                "max_same_class_fraction_by_class"
            ],
        )
        candidate_mask = candidate_mask & (
            same_class_thresholds.isna()
            | sample_metrics["same_class_fraction"].le(same_class_thresholds)
        )

    return candidate_mask


def has_same_class_filter(filter_preview_config):
    """
    Return whether same-class purity is part of the preview rule.

    Parameters
    ----------
    filter_preview_config : dict
        Filter-preview settings.

    Returns
    -------
    bool
        Whether same-class filtering is configured.
    """

    return (
        filter_preview_config["max_same_class_fraction"] is not None
        or bool(filter_preview_config["max_same_class_fraction_by_class"])
    )


def create_class_threshold_series(rows, default_threshold, class_thresholds):
    """
    Create a row-aligned threshold series with class-specific overrides.

    Parameters
    ----------
    rows : pandas.DataFrame
        Per-sample metric rows.
    default_threshold : float
        Default threshold.
    class_thresholds : dict
        Mapping from class name to threshold.

    Returns
    -------
    pandas.Series
        Row-aligned thresholds.
    """

    thresholds = pd.Series(float(default_threshold), index=rows.index)
    for class_name, threshold in class_thresholds.items():
        thresholds.loc[rows["class_name"] == class_name] = float(threshold)

    return thresholds


def create_optional_class_threshold_series(rows, default_threshold, class_thresholds):
    """
    Create a row-aligned optional threshold series.

    Parameters
    ----------
    rows : pandas.DataFrame
        Per-sample metric rows.
    default_threshold : float or None
        Default threshold. ``None`` leaves rows unconstrained unless their class
        has an override.
    class_thresholds : dict
        Mapping from class name to threshold.

    Returns
    -------
    pandas.Series
        Row-aligned thresholds with NaN for unconstrained rows.
    """

    if default_threshold is None:
        thresholds = pd.Series(np.nan, index=rows.index, dtype=float)
    else:
        thresholds = pd.Series(float(default_threshold), index=rows.index)

    for class_name, threshold in class_thresholds.items():
        thresholds.loc[rows["class_name"] == class_name] = float(threshold)

    return thresholds


def create_filter_preview_reasons(rows, filter_preview_config):
    """
    Create preview reason strings for marked samples.

    Parameters
    ----------
    rows : pandas.DataFrame
        Marked preview rows.
    filter_preview_config : dict
        Filter-preview settings.

    Returns
    -------
    list of str
        Reason strings.
    """

    reasons = []
    for _, row in rows.iterrows():
        class_name = str(row["class_name"])
        point_threshold = filter_preview_config[
            "min_point_neighbor_fraction_by_class"
        ].get(class_name, filter_preview_config["min_point_neighbor_fraction"])
        reason_parts = [f"point_neighbor_fraction>={point_threshold:g}"]

        if has_same_class_filter(filter_preview_config):
            same_threshold = filter_preview_config[
                "max_same_class_fraction_by_class"
            ].get(class_name, filter_preview_config["max_same_class_fraction"])
            if same_threshold is not None:
                reason_parts.append(f"same_class_fraction<={same_threshold:g}")

        reasons.append("; ".join(reason_parts))

    return reasons


def create_split_offsets(sample_metrics):
    """
    Create contiguous row offsets for each split in the sample metrics table.

    Parameters
    ----------
    sample_metrics : pandas.DataFrame
        Per-sample PCA metric rows.

    Returns
    -------
    dict
        Mapping from split name to ``(start, stop)`` row offsets.
    """

    offsets = {}
    start = 0
    for split_name, rows in sample_metrics.groupby("split", sort=False):
        stop = start + len(rows)
        offsets[str(split_name)] = (start, stop)
        start = stop
    return offsets


def create_pca_metrics_text(
    dataset_dir,
    output_dir,
    X,
    sample_metrics,
    explained_variance_ratio,
    n_components,
    feature_config,
    pca_summary,
    pca_fit_config,
    split_config,
    neighbor_config,
    filter_preview_config,
    embedding_plot_config,
):
    """
    Create a text summary for PCA plots and metrics.

    Parameters
    ----------
    dataset_dir : str or pathlib.Path
        Dataset directory.
    output_dir : str or pathlib.Path
        PCA output directory.
    X : numpy.ndarray
        VDF samples.
    sample_metrics : pandas.DataFrame
        Per-sample PCA metric rows.
    explained_variance_ratio : numpy.ndarray
        Explained variance ratio for PCA components.
    n_components : int
        Number of PCA components.
    feature_config : dict
        Feature extraction settings.
    pca_summary : dict
        PCA backend summary.
    pca_fit_config : dict
        PCA fit-sample settings.
    split_config : dict
        Split settings and split timestep arrays.
    neighbor_config : dict
        Neighbor metric settings.
    filter_preview_config : dict
        Filter-preview settings.
    embedding_plot_config : dict
        Embedding plot settings.

    Returns
    -------
    str
        PCA metrics text.
    """

    explained = np.asarray(explained_variance_ratio, dtype=float)
    explained_rows = pd.DataFrame(
        {
            "component": np.arange(1, len(explained) + 1),
            "explained_variance_ratio": explained,
            "cumulative_explained_variance_ratio": np.cumsum(explained),
        }
    )
    count_table = pd.crosstab(
        sample_metrics["split"],
        sample_metrics["class_name"],
        margins=True,
    )

    lines = [
        "VDF dataset PCA metrics",
        "=" * 70,
        f"Dataset directory: {dataset_dir}",
        f"Output directory: {output_dir}",
        f"Raw dataset shape: {X.shape}",
        f"PCA backend: {pca_summary.get('backend', 'unknown')}",
        f"PCA algorithm: {pca_summary.get('algorithm', 'unknown')}",
        f"PCA device: {pca_summary.get('device', 'unknown')}",
        f"PyTorch version: {pca_summary.get('torch_version', 'not used')}",
        f"PCA oversamples: {pca_summary.get('oversamples', 'not used')}",
        f"PCA power iterations: {pca_summary.get('niter', 'not used')}",
        f"PCA components: {n_components}",
        f"PCA fit balanced: {pca_fit_config['balanced']}",
        f"PCA fit samples: {pca_summary.get('fit_sample_count', 'unknown')}",
        f"PCA fit samples per class: {pca_fit_config['samples_per_class']}",
        f"PCA fit replacement: {pca_fit_config['replace']}",
        "PCA fit class names: "
        f"{format_config_list(pca_fit_config['class_names'], default='all')}",
        f"Embedding plot enabled: {embedding_plot_config['enabled']}",
        f"Embedding plot method: {embedding_plot_config['method']}",
        f"Downsample factor: {feature_config['downsample_factor']}",
        f"Feature batch size: {feature_config['batch_size']}",
        f"Feature extraction jobs: {feature_config['n_jobs']}",
        f"Log epsilon: {feature_config['log_eps']}",
        f"Sample normalization: {feature_config['sample_normalization']}",
        f"Sample normalization epsilon: {feature_config['sample_norm_eps']}",
        f"Train fraction of usable timesteps: {split_config['train_fraction']}",
        "Validation fraction of usable timesteps: "
        f"{split_config['validation_fraction']}",
        f"Gap timesteps per boundary: {split_config['gap_timesteps']}",
        "Train timesteps: "
        f"{format_timestep_range(split_config['train_timesteps'])}",
        "Train-validation gap: "
        f"{format_timestep_range(split_config['train_validation_gap'])}",
        "Validation timesteps: "
        f"{format_timestep_range(split_config['validation_timesteps'])}",
        "Validation-test gap: "
        f"{format_timestep_range(split_config['validation_test_gap'])}",
        "Test timesteps: "
        f"{format_timestep_range(split_config['test_timesteps'])}",
        "",
        "Sample counts by split and class",
        "=" * 70,
        count_table.to_string(),
        "",
        "Explained variance",
        "=" * 70,
        explained_rows.to_string(index=False),
    ]

    if neighbor_config["enabled"] and "same_class_fraction" in sample_metrics.columns:
        lines.extend(create_neighbor_metric_lines(sample_metrics, neighbor_config))

    if (
        filter_preview_config["enabled"]
        and "filter_preview_candidate" in sample_metrics.columns
    ):
        lines.extend(
            create_filter_preview_metric_lines(
                sample_metrics=sample_metrics,
                filter_preview_config=filter_preview_config,
            )
        )

    embedding_plot_path = (
        Path(output_dir) / f"pca_{embedding_plot_config['method']}_by_class.png"
    )
    lines.extend(
        [
            "",
            "Saved outputs",
            "=" * 70,
            f"Sample metrics: {Path(output_dir) / 'pca_sample_metrics.csv'}",
            f"PCA scores: {Path(output_dir) / 'pca_scores.npz'}",
            f"Train class plot: {Path(output_dir) / 'pca_train_by_class.png'}",
            f"All class plot: {Path(output_dir) / 'pca_all_by_class.png'}",
            "Train neighbor-purity plot: "
            f"{Path(output_dir) / 'pca_train_neighbor_purity.png'}",
            f"Filter-preview plot: {Path(output_dir) / 'pca_filter_preview.png'}",
            f"Embedding plot: {embedding_plot_path}",
        ]
    )

    return "\n".join(lines)


def create_neighbor_metric_lines(sample_metrics, neighbor_config):
    """
    Create text sections for nearest-neighbor label-purity metrics.

    Parameters
    ----------
    sample_metrics : pandas.DataFrame
        Per-sample PCA metric rows.
    neighbor_config : dict
        Neighbor metric settings.

    Returns
    -------
    list of str
        Text lines.
    """

    purity_summary = (
        sample_metrics.groupby(["split", "class_name"], sort=True)[
            "same_class_fraction"
        ]
        .agg(["count", "mean", "median", "min"])
        .reset_index()
    )

    threshold_rows = []
    for threshold in neighbor_config["low_purity_thresholds"]:
        low_purity = sample_metrics[
            sample_metrics["same_class_fraction"] < float(threshold)
        ]
        counts = (
            low_purity.groupby(["split", "class_name"], sort=True)
            .size()
            .reset_index(name="count")
        )
        counts["threshold"] = float(threshold)
        threshold_rows.append(counts)
    if threshold_rows:
        threshold_summary = pd.concat(threshold_rows, ignore_index=True)
        threshold_summary = threshold_summary[
            ["threshold", "split", "class_name", "count"]
        ]
    else:
        threshold_summary = pd.DataFrame(
            columns=["threshold", "split", "class_name", "count"]
        )

    other_class_rows = sample_metrics[
        sample_metrics["nearest_other_class"].astype(str) != ""
    ]
    nearest_other_summary = (
        other_class_rows.groupby(
            ["split", "class_name", "nearest_other_class"],
            sort=True,
        )
        .size()
        .reset_index(name="count")
    )

    return [
        "",
        "Nearest-neighbor label purity",
        "=" * 70,
        f"Neighbor count: {neighbor_config['k_neighbors']}",
        purity_summary.to_string(index=False),
        "",
        "Low-purity sample counts",
        "=" * 70,
        threshold_summary.to_string(index=False),
        "",
        "Nearest other class summary",
        "=" * 70,
        nearest_other_summary.to_string(index=False),
    ]


def create_filter_preview_metric_lines(sample_metrics, filter_preview_config):
    """
    Create text sections for PCA-space filter-preview metrics.

    Parameters
    ----------
    sample_metrics : pandas.DataFrame
        Per-sample PCA metric rows.
    filter_preview_config : dict
        Filter-preview settings.

    Returns
    -------
    list of str
        Text lines.
    """

    preview_rows = sample_metrics[
        sample_metrics["split"].isin(filter_preview_config["apply_splits"])
    ]
    fraction_summary = (
        preview_rows.groupby(["split", "class_name"], sort=True)[
            "point_neighbor_fraction"
        ]
        .agg(["count", "mean", "median", "max"])
        .reset_index()
    )

    candidate_rows = preview_rows[preview_rows["filter_preview_candidate"]]
    candidate_summary = (
        candidate_rows.groupby(["split", "class_name"], sort=True)
        .size()
        .reset_index(name="count")
    )
    nearest_point_summary = (
        candidate_rows.groupby(
            ["split", "class_name", "nearest_point_class"],
            sort=True,
        )
        .size()
        .reset_index(name="count")
    )

    return [
        "",
        "PCA-space filter preview",
        "=" * 70,
        "Preview only: no dataset samples are removed by this script.",
        "Candidate classes: "
        f"{', '.join(filter_preview_config['candidate_classes'])}",
        "Point-neighbor classes: "
        f"{', '.join(filter_preview_config['point_neighbor_classes'])}",
        "Protected classes: "
        f"{', '.join(filter_preview_config['protected_classes'])}",
        "Preview splits: "
        f"{', '.join(filter_preview_config['apply_splits'])}",
        "Minimum point-neighbor fraction: "
        f"{filter_preview_config['min_point_neighbor_fraction']}",
        "Minimum point-neighbor fraction by class: "
        f"{format_threshold_mapping(filter_preview_config['min_point_neighbor_fraction_by_class'])}",
        "Maximum same-class fraction: "
        f"{format_optional_threshold(filter_preview_config['max_same_class_fraction'])}",
        "Maximum same-class fraction by class: "
        f"{format_threshold_mapping(filter_preview_config['max_same_class_fraction_by_class'])}",
        "",
        "Point-neighbor fraction summary",
        "=" * 70,
        fraction_summary.to_string(index=False),
        "",
        "Preview candidate counts",
        "=" * 70,
        candidate_summary.to_string(index=False),
        "",
        "Preview candidates by nearest point class",
        "=" * 70,
        nearest_point_summary.to_string(index=False),
    ]


def subsample_plot_rows(sample_metrics, max_points_per_class, random_state):
    """
    Subsample PCA rows for plotting.

    Parameters
    ----------
    sample_metrics : pandas.DataFrame
        Per-sample PCA metric rows.
    max_points_per_class : int
        Maximum plotted samples per class and split.
    random_state : int
        Random seed.

    Returns
    -------
    pandas.DataFrame
        Subsampled rows.
    """

    rng = np.random.default_rng(int(random_state))
    selected_rows = []
    group_columns = ["split", "class_name"]
    for _, rows in sample_metrics.groupby(group_columns, sort=False):
        if len(rows) <= max_points_per_class:
            selected_rows.append(rows)
            continue
        selected_index = rng.choice(
            rows.index.to_numpy(),
            size=max_points_per_class,
            replace=False,
        )
        selected_rows.append(rows.loc[np.sort(selected_index)])

    result = pd.concat(selected_rows, ignore_index=False).copy()
    result["sample_metric_index"] = result.index

    return result.reset_index(drop=True)


def plot_pca_by_class(rows, output_path, title, alpha, point_size):
    """
    Plot PCA coordinates colored by class.

    Parameters
    ----------
    rows : pandas.DataFrame
        Rows to plot.
    output_path : str or pathlib.Path
        Output plot path.
    title : str
        Plot title.
    alpha : float
        Scatter alpha.
    point_size : float
        Scatter point size.

    Returns
    -------
    bool
        Whether the plot was saved.
    """

    if rows.empty:
        return False

    fig, ax = plt.subplots(figsize=(8, 6))
    for class_name, class_rows in rows.groupby("class_name", sort=True):
        ax.scatter(
            class_rows["pca_0"],
            class_rows["pca_1"],
            s=point_size,
            alpha=alpha,
            label=str(class_name),
        )

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    return True


def plot_pca_neighbor_purity(rows, output_path, alpha, point_size):
    """
    Plot training PCA coordinates colored by same-class neighbor fraction.

    Parameters
    ----------
    rows : pandas.DataFrame
        Training rows to plot.
    output_path : str or pathlib.Path
        Output plot path.
    alpha : float
        Scatter alpha.
    point_size : float
        Scatter point size.

    Returns
    -------
    bool
        Whether the plot was saved.
    """

    if rows.empty or rows["same_class_fraction"].isna().all():
        return False

    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(
        rows["pca_0"],
        rows["pca_1"],
        c=rows["same_class_fraction"],
        s=point_size,
        alpha=alpha,
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
    )
    colorbar = fig.colorbar(scatter, ax=ax)
    colorbar.set_label("Same-class neighbor fraction")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Training PCA neighbor purity")
    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    return True


def plot_pca_filter_preview(
    rows,
    output_path,
    filter_preview_config,
    alpha,
    point_size,
):
    """
    Plot PCA coordinates with preview filtering candidates overlaid.

    Parameters
    ----------
    rows : pandas.DataFrame
        Rows to plot.
    output_path : str or pathlib.Path
        Output plot path.
    filter_preview_config : dict
        Filter-preview settings.
    alpha : float
        Scatter alpha.
    point_size : float
        Scatter point size.

    Returns
    -------
    bool
        Whether the plot was saved.
    """

    if rows.empty or "filter_preview_candidate" not in rows.columns:
        return False

    fig, ax = plt.subplots(figsize=(8, 6))
    for class_name, class_rows in rows.groupby("class_name", sort=True):
        ax.scatter(
            class_rows["pca_0"],
            class_rows["pca_1"],
            s=point_size,
            alpha=0.35 * alpha,
            label=str(class_name),
        )

    point_rows = rows[
        rows["class_name"].isin(filter_preview_config["point_neighbor_classes"])
    ]
    if not point_rows.empty:
        ax.scatter(
            point_rows["pca_0"],
            point_rows["pca_1"],
            s=point_size * 4.0,
            facecolors="none",
            edgecolors="black",
            linewidths=0.8,
            label="point-neighbor classes",
        )

    candidate_rows = rows[rows["filter_preview_candidate"]]
    if not candidate_rows.empty:
        ax.scatter(
            candidate_rows["pca_0"],
            candidate_rows["pca_1"],
            s=point_size * 5.0,
            c="red",
            marker="x",
            linewidths=0.9,
            label="filter preview candidates",
        )

    threshold = filter_preview_config["min_point_neighbor_fraction"]
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(f"PCA filter preview, point-neighbor fraction >= {threshold:g}")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    return True


def plot_pca_embedding(
    sample_metrics,
    scores_by_split,
    split_indices,
    output_path,
    embedding_plot_config,
    alpha,
    point_size,
):
    """
    Plot t-SNE or UMAP coordinates computed from PCA scores.

    Parameters
    ----------
    sample_metrics : pandas.DataFrame
        Per-sample PCA metric rows.
    scores_by_split : dict
        Mapping from split name to PCA scores.
    split_indices : dict
        Mapping from split name to sample indices.
    output_path : str or pathlib.Path
        Output plot path.
    embedding_plot_config : dict
        Embedding plot settings.
    alpha : float
        Scatter alpha.
    point_size : float
        Scatter point size.

    Returns
    -------
    bool
        Whether the plot was saved.
    """

    selected_metrics = sample_metrics[
        sample_metrics["split"].isin(embedding_plot_config["splits"])
    ]
    if selected_metrics.empty:
        return False

    plot_rows = subsample_plot_rows(
        sample_metrics=selected_metrics,
        max_points_per_class=embedding_plot_config["max_points_per_class"],
        random_state=embedding_plot_config["random_state"],
    )
    if len(plot_rows) < 3:
        return False

    scores = np.concatenate(
        [scores_by_split[split_name] for split_name in split_indices],
        axis=0,
    )
    selected_scores = scores[
        plot_rows["sample_metric_index"].to_numpy(dtype=int)
    ]
    embedding = compute_score_embedding(
        scores=selected_scores,
        embedding_plot_config=embedding_plot_config,
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    for class_name, class_rows in plot_rows.groupby("class_name", sort=True):
        row_indices = class_rows.index.to_numpy(dtype=int)
        ax.scatter(
            embedding[row_indices, 0],
            embedding[row_indices, 1],
            s=point_size,
            alpha=alpha,
            label=str(class_name),
        )

    method = embedding_plot_config["method"].upper()
    ax.set_xlabel(f"{method} 1")
    ax.set_ylabel(f"{method} 2")
    ax.set_title(f"{method} of PCA scores")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    return True


def compute_score_embedding(scores, embedding_plot_config):
    """
    Compute a 2D nonlinear embedding from PCA scores.

    Parameters
    ----------
    scores : numpy.ndarray
        PCA scores.
    embedding_plot_config : dict
        Embedding plot settings.

    Returns
    -------
    numpy.ndarray
        Two-dimensional embedding.
    """

    method = embedding_plot_config["method"]
    if method == "tsne":
        from sklearn.manifold import TSNE

        perplexity = min(
            float(embedding_plot_config["perplexity"]),
            max(1.0, (len(scores) - 1) / 3.0),
        )
        embedding = TSNE(
            n_components=2,
            perplexity=perplexity,
            init="pca",
            learning_rate="auto",
            random_state=embedding_plot_config["random_state"],
        ).fit_transform(scores)

        return np.asarray(embedding, dtype=np.float32)

    try:
        import umap
    except ImportError as exc:
        raise ImportError(
            "embedding_plot.method: umap requires the optional umap-learn package"
        ) from exc

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=min(int(embedding_plot_config["n_neighbors"]), len(scores) - 1),
        min_dist=float(embedding_plot_config["min_dist"]),
        random_state=embedding_plot_config["random_state"],
    )

    return np.asarray(reducer.fit_transform(scores), dtype=np.float32)


def format_timestep_range(timesteps):
    """
    Format a timestep array as a compact range.

    Parameters
    ----------
    timesteps : array-like of int
        Timestep values.

    Returns
    -------
    str
        Formatted range.
    """

    timesteps = np.asarray(timesteps)
    if len(timesteps) == 0:
        return "none"
    if len(timesteps) == 1:
        return str(int(timesteps[0]))

    return f"{int(timesteps[0])} ... {int(timesteps[-1])}"


def format_config_list(values, default="none"):
    """
    Format a config list for metrics text.

    Parameters
    ----------
    values : list
        Config values.
    default : str, optional
        Text returned when the list is empty.

    Returns
    -------
    str
        Comma-separated values or default text.
    """

    if not values:
        return str(default)

    return ", ".join(str(value) for value in values)


def format_optional_threshold(value):
    """
    Format an optional threshold value.

    Parameters
    ----------
    value : float or None
        Threshold value.

    Returns
    -------
    str
        Formatted threshold or ``disabled``.
    """

    if value is None:
        return "disabled"

    return f"{float(value):g}"


def format_threshold_mapping(thresholds):
    """
    Format class-specific threshold mapping.

    Parameters
    ----------
    thresholds : dict
        Mapping from class name to threshold.

    Returns
    -------
    str
        Formatted threshold mapping.
    """

    if not thresholds:
        return "none"

    return ", ".join(
        f"{class_name}: {float(threshold):g}"
        for class_name, threshold in thresholds.items()
    )
