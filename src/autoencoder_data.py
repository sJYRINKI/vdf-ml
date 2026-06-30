from pathlib import Path
import time

from joblib import Parallel, delayed
import numpy as np
import torch
from torch.utils.data import Dataset

from src.dataset_io import load_dataset
from src.model_split import split_by_timestep
from src.timesteps import create_path


def load_autoencoder_training_data(config, dataset_id, model_id):
    """
    Load VDF arrays and create autoencoder datasets.

    Parameters
    ----------
    config : dict
        Autoencoder training config.
    dataset_id : str
        Dataset identifier.
    model_id : str
        Model identifier.

    Returns
    -------
    dict
        Dataset arrays, split indices, normalization values, and PyTorch
        datasets.
    """

    dataset_dir = create_path(
        path_template=config["dataset_dir"],
        dataset_id=dataset_id,
        model_id=model_id,
    )
    output_dir = create_path(
        path_template=config["output_dir"],
        dataset_id=dataset_id,
        model_id=model_id,
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    input_config = resolve_input_config(config.get("input", {}))
    cache_config = resolve_cache_config(
        config=config.get("cache", {}),
        dataset_dir=dataset_dir,
        dataset_id=dataset_id,
        model_id=model_id,
    )
    split_config = config.get("split", {})
    train_fraction = float(split_config.get("train_fraction", 0.6))
    validation_fraction = float(split_config.get("validation_fraction", 0.2))
    gap_timesteps = int(split_config.get("gap_timesteps", 10))

    X, y, metadata = load_dataset(dataset_dir, mmap=True)

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

    X_log, cache_metadata = create_or_load_log_slice_cache(
        X=X,
        input_config=input_config,
        cache_config=cache_config,
    )

    stats_batch_size = int(
        config.get("training", {}).get("normalization_batch_size", 64)
    )
    if X_log is None:
        normalization = compute_log_slice_normalization(
            X=X,
            indices=train_indices,
            log_eps=input_config["log_eps"],
            clip_negative_to_zero=input_config["clip_negative_to_zero"],
            batch_size=stats_batch_size,
        )
    else:
        normalization = compute_cached_log_slice_normalization(
            X_log=X_log,
            indices=train_indices,
            batch_size=stats_batch_size,
        )

    train_dataset = VdfSliceDataset(
        X=X,
        X_log=X_log,
        y=y,
        indices=train_indices,
        normalization=normalization,
        input_config=input_config,
    )
    validation_dataset = VdfSliceDataset(
        X=X,
        X_log=X_log,
        y=y,
        indices=validation_indices,
        normalization=normalization,
        input_config=input_config,
    )
    test_dataset = VdfSliceDataset(
        X=X,
        X_log=X_log,
        y=y,
        indices=test_indices,
        normalization=normalization,
        input_config=input_config,
    )

    return {
        "X": X,
        "y": y,
        "X_log": X_log,
        "metadata": metadata,
        "dataset_dir": Path(dataset_dir),
        "output_dir": Path(output_dir),
        "input_config": input_config,
        "cache_config": cache_config,
        "cache_metadata": cache_metadata,
        "normalization": normalization,
        "train_dataset": train_dataset,
        "validation_dataset": validation_dataset,
        "test_dataset": test_dataset,
        "train_indices": train_indices,
        "validation_indices": validation_indices,
        "test_indices": test_indices,
        "train_timesteps": train_timesteps,
        "validation_timesteps": validation_timesteps,
        "test_timesteps": test_timesteps,
        "train_validation_gap": train_validation_gap,
        "validation_test_gap": validation_test_gap,
        "train_fraction": train_fraction,
        "validation_fraction": validation_fraction,
        "gap_timesteps": gap_timesteps,
        "slice_shape": train_dataset.slice_shape,
    }


def resolve_cache_config(config, dataset_dir, dataset_id, model_id):
    """
    Resolve VDF log-slice cache settings.

    Parameters
    ----------
    config : dict
        Cache configuration.
    dataset_dir : str or pathlib.Path
        Dataset directory.
    dataset_id : str
        Dataset identifier.
    model_id : str
        Model identifier.

    Returns
    -------
    dict
        Validated cache settings.
    """

    enabled = bool(config.get("enabled", False))
    if config.get("dir") is None:
        cache_dir = Path(dataset_dir) / "cache"
    else:
        cache_dir = create_path(
            path_template=config["dir"],
            dataset_id=dataset_id,
            model_id=model_id,
        )

    filename = config.get("filename", "autoencoder_xz_log.npy")
    metadata_filename = config.get(
        "metadata_filename",
        "autoencoder_xz_log_metadata.npz",
    )
    batch_size = int(config.get("batch_size", 64))
    n_jobs = int(config.get("n_jobs", 1))

    if batch_size <= 0:
        raise ValueError("cache.batch_size must be positive")
    if n_jobs == 0:
        raise ValueError("cache.n_jobs must be non-zero")

    return {
        "enabled": enabled,
        "cache_dir": Path(cache_dir),
        "cache_path": Path(cache_dir) / filename,
        "metadata_path": Path(cache_dir) / metadata_filename,
        "rebuild": bool(config.get("rebuild", False)),
        "batch_size": batch_size,
        "n_jobs": n_jobs,
    }


def resolve_input_config(config):
    """
    Resolve and validate VDF input preprocessing settings.

    Parameters
    ----------
    config : dict
        Input config values.

    Returns
    -------
    dict
        Validated input settings.
    """

    slice_name = config.get("slice", "xz")
    orientation = config.get("orientation", "plot")
    normalization = config.get("normalization", "train_global_standard")
    log_eps = float(config.get("log_eps", 1e-30))

    if slice_name != "xz":
        raise ValueError("Only input.slice='xz' is supported")
    if orientation != "plot":
        raise ValueError("Only input.orientation='plot' is supported")
    if normalization != "train_global_standard":
        raise ValueError(
            "Only input.normalization='train_global_standard' is supported"
        )
    if log_eps <= 0.0:
        raise ValueError("input.log_eps must be positive")

    return {
        "slice": slice_name,
        "orientation": orientation,
        "normalization": normalization,
        "log_eps": log_eps,
        "clip_negative_to_zero": bool(
            config.get("clip_negative_to_zero", True)
        ),
    }


class VdfSliceDataset(Dataset):
    """
    Stream normalized xz VDF slices from a memory-mapped dataset.

    Parameters
    ----------
    X : numpy.ndarray
        Memory-mapped VDF samples with shape ``(n_samples, vx, vy, vz)``.
    X_log : numpy.ndarray or None
        Optional cached log-scaled xz slices with shape
        ``(n_samples, 1, vz, vx)``.
    y : numpy.ndarray
        Integer labels.
    indices : array-like of int
        Sample indices included in this dataset.
    normalization : dict
        Log-slice normalization values.
    input_config : dict
        VDF preprocessing settings.
    """

    def __init__(self, X, X_log, y, indices, normalization, input_config):
        self.X = X
        self.X_log = X_log
        self.y = y
        self.indices = np.asarray(indices, dtype=int)
        self.normalization = normalization
        self.input_config = input_config
        if self.X_log is None:
            self.slice_shape = infer_plot_xz_slice_shape(X)
        else:
            self.slice_shape = tuple(int(value) for value in self.X_log.shape[2:])

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        sample_index = int(self.indices[int(index)])
        if self.X_log is None:
            slice_log = create_log_plot_xz_slice_from_dataset(
                X=self.X,
                sample_index=sample_index,
                log_eps=self.input_config["log_eps"],
                clip_negative_to_zero=self.input_config["clip_negative_to_zero"],
            )
        else:
            slice_log = np.asarray(
                self.X_log[sample_index, 0],
                dtype=np.float32,
            )
        normalized_slice = normalize_log_slice(
            slice_log=slice_log,
            mean=self.normalization["mean"],
            std=self.normalization["std"],
        )
        tensor = torch.from_numpy(normalized_slice[None, ...])

        return {
            "input": tensor,
            "target": tensor,
            "sample_index": sample_index,
            "label": int(self.y[sample_index]),
        }


def create_or_load_log_slice_cache(X, input_config, cache_config):
    """
    Create or load cached log-scaled xz VDF slices.

    Parameters
    ----------
    X : numpy.ndarray
        VDF samples with shape ``(n_samples, vx, vy, vz)``.
    input_config : dict
        VDF preprocessing settings.
    cache_config : dict
        Cache settings.

    Returns
    -------
    X_log : numpy.ndarray or None
        Cached log slices, or ``None`` when caching is disabled.
    cache_metadata : dict
        Cache metadata.
    """

    if not cache_config["enabled"]:
        return None, {"enabled": False}

    cache_path = cache_config["cache_path"]
    metadata_path = cache_config["metadata_path"]

    if (
        not cache_config["rebuild"]
        and is_log_slice_cache_valid(
            X=X,
            input_config=input_config,
            cache_path=cache_path,
            metadata_path=metadata_path,
        )
    ):
        X_log = np.load(cache_path, mmap_mode="r")
        return X_log, load_log_slice_cache_metadata(metadata_path)

    cache_metadata = create_log_slice_cache(
        X=X,
        input_config=input_config,
        cache_config=cache_config,
    )
    X_log = np.load(cache_path, mmap_mode="r")

    return X_log, cache_metadata


def create_log_slice_cache(X, input_config, cache_config):
    """
    Create a memory-mapped cache of log-scaled xz VDF slices.

    Parameters
    ----------
    X : numpy.ndarray
        VDF samples with shape ``(n_samples, vx, vy, vz)``.
    input_config : dict
        VDF preprocessing settings.
    cache_config : dict
        Cache settings.

    Returns
    -------
    dict
        Cache metadata.
    """

    cache_path = cache_config["cache_path"]
    metadata_path = cache_config["metadata_path"]
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    cache_shape = (int(X.shape[0]), 1, *infer_plot_xz_slice_shape(X))
    X_log = np.lib.format.open_memmap(
        cache_path,
        mode="w+",
        dtype=np.float32,
        shape=cache_shape,
    )

    start = time.perf_counter()
    sample_indices = np.arange(int(X.shape[0]), dtype=int)
    batches = list(
        iter_index_batches(
            sample_indices,
            batch_size=cache_config["batch_size"],
        )
    )

    print(f"Creating autoencoder log-slice cache: {cache_path}")
    print(f"Cache shape: {cache_shape}")
    print(f"Cache jobs: {cache_config['n_jobs']}")

    if cache_config["n_jobs"] == 1:
        for batch_indices in batches:
            write_log_slice_cache_batch(
                X=X,
                X_log=X_log,
                batch_indices=batch_indices,
                input_config=input_config,
            )
    else:
        Parallel(
            n_jobs=cache_config["n_jobs"],
            prefer="threads",
            require="sharedmem",
        )(
            delayed(write_log_slice_cache_batch)(
                X=X,
                X_log=X_log,
                batch_indices=batch_indices,
                input_config=input_config,
            )
            for batch_indices in batches
        )

    X_log.flush()
    elapsed = time.perf_counter() - start
    metadata = {
        "enabled": True,
        "cache_path": str(cache_path),
        "metadata_path": str(metadata_path),
        "raw_vdf_shape": tuple(int(value) for value in X.shape),
        "cache_shape": tuple(int(value) for value in cache_shape),
        "log_eps": float(input_config["log_eps"]),
        "clip_negative_to_zero": bool(input_config["clip_negative_to_zero"]),
        "slice": input_config["slice"],
        "orientation": input_config["orientation"],
        "elapsed_seconds": float(elapsed),
    }
    save_log_slice_cache_metadata(metadata_path, metadata)

    print(f"Created autoencoder log-slice cache in {elapsed:.2f} s")

    return metadata


def write_log_slice_cache_batch(X, X_log, batch_indices, input_config):
    """
    Write one batch of log-scaled xz slices into the cache.

    Parameters
    ----------
    X : numpy.ndarray
        VDF samples with shape ``(n_samples, vx, vy, vz)``.
    X_log : numpy.ndarray
        Log-slice cache.
    batch_indices : array-like of int
        Sample indices to write.
    input_config : dict
        VDF preprocessing settings.
    """

    for sample_index in batch_indices:
        X_log[int(sample_index), 0] = create_log_plot_xz_slice_from_dataset(
            X=X,
            sample_index=int(sample_index),
            log_eps=input_config["log_eps"],
            clip_negative_to_zero=input_config["clip_negative_to_zero"],
        )


def is_log_slice_cache_valid(X, input_config, cache_path, metadata_path):
    """
    Return whether an existing log-slice cache matches current settings.

    Parameters
    ----------
    X : numpy.ndarray
        VDF samples with shape ``(n_samples, vx, vy, vz)``.
    input_config : dict
        VDF preprocessing settings.
    cache_path : str or pathlib.Path
        Cache array path.
    metadata_path : str or pathlib.Path
        Cache metadata path.

    Returns
    -------
    bool
        Whether the cache can be reused.
    """

    cache_path = Path(cache_path)
    metadata_path = Path(metadata_path)
    if not cache_path.exists() or not metadata_path.exists():
        return False

    try:
        X_log = np.load(cache_path, mmap_mode="r")
        metadata = load_log_slice_cache_metadata(metadata_path)
    except Exception:
        return False

    expected_shape = (int(X.shape[0]), 1, *infer_plot_xz_slice_shape(X))

    return (
        tuple(X_log.shape) == expected_shape
        and tuple(metadata.get("raw_vdf_shape", ())) == tuple(X.shape)
        and tuple(metadata.get("cache_shape", ())) == expected_shape
        and float(metadata.get("log_eps", np.nan)) == float(input_config["log_eps"])
        and bool(metadata.get("clip_negative_to_zero")) == bool(
            input_config["clip_negative_to_zero"]
        )
        and metadata.get("slice") == input_config["slice"]
        and metadata.get("orientation") == input_config["orientation"]
    )


def save_log_slice_cache_metadata(metadata_path, metadata):
    """
    Save log-slice cache metadata.

    Parameters
    ----------
    metadata_path : str or pathlib.Path
        Cache metadata path.
    metadata : dict
        Cache metadata.
    """

    metadata_path = Path(metadata_path)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        metadata_path,
        enabled=np.asarray(metadata["enabled"]),
        cache_path=np.asarray(metadata["cache_path"]),
        metadata_path=np.asarray(metadata["metadata_path"]),
        raw_vdf_shape=np.asarray(metadata["raw_vdf_shape"], dtype=int),
        cache_shape=np.asarray(metadata["cache_shape"], dtype=int),
        log_eps=np.asarray(metadata["log_eps"]),
        clip_negative_to_zero=np.asarray(metadata["clip_negative_to_zero"]),
        slice=np.asarray(metadata["slice"]),
        orientation=np.asarray(metadata["orientation"]),
        elapsed_seconds=np.asarray(metadata["elapsed_seconds"]),
    )


def load_log_slice_cache_metadata(metadata_path):
    """
    Load log-slice cache metadata.

    Parameters
    ----------
    metadata_path : str or pathlib.Path
        Cache metadata path.

    Returns
    -------
    dict
        Cache metadata.
    """

    with np.load(metadata_path, allow_pickle=False) as metadata:
        return {
            "enabled": bool(metadata["enabled"].item()),
            "cache_path": str(metadata["cache_path"].item()),
            "metadata_path": str(metadata["metadata_path"].item()),
            "raw_vdf_shape": tuple(
                int(value) for value in metadata["raw_vdf_shape"]
            ),
            "cache_shape": tuple(int(value) for value in metadata["cache_shape"]),
            "log_eps": float(metadata["log_eps"].item()),
            "clip_negative_to_zero": bool(
                metadata["clip_negative_to_zero"].item()
            ),
            "slice": str(metadata["slice"].item()),
            "orientation": str(metadata["orientation"].item()),
            "elapsed_seconds": float(metadata["elapsed_seconds"].item()),
        }


def infer_plot_xz_slice_shape(X):
    """
    Infer plot-oriented xz-slice shape from a VDF dataset.

    Parameters
    ----------
    X : numpy.ndarray
        VDF samples with shape ``(n_samples, vx, vy, vz)``.

    Returns
    -------
    tuple of int
        Shape of one plot-oriented xz slice.
    """

    if len(X.shape) != 4:
        raise ValueError("X must have shape (n_samples, vx, vy, vz)")

    return int(X.shape[3]), int(X.shape[1])


def extract_plot_xz_slice_from_dataset(X, sample_index):
    """
    Extract the plot-oriented middle xz slice from a saved VDF sample.

    This matches ``src.dataset_plot._prepare_vdf_xz_plot`` without reading the
    full 3D VDF into memory.

    Parameters
    ----------
    X : numpy.ndarray
        VDF samples with shape ``(n_samples, vx, vy, vz)``.
    sample_index : int
        Dataset sample index.

    Returns
    -------
    numpy.ndarray
        Plot-oriented xz slice with shape ``(vz, vx)``.
    """

    mid_y = X.shape[2] // 2
    return np.asarray(X[int(sample_index), :, mid_y, :].T, dtype=np.float32)


def extract_plot_xz_slice(vdf):
    """
    Extract the plot-oriented middle xz slice from one dense VDF.

    Parameters
    ----------
    vdf : numpy.ndarray
        Dense VDF array with shape ``(vx, vy, vz)``.

    Returns
    -------
    numpy.ndarray
        Plot-oriented xz slice with shape ``(vz, vx)``.
    """

    vdf_swapped = np.swapaxes(vdf, 2, 0)
    mid_y = vdf_swapped.shape[1] // 2

    return np.asarray(vdf_swapped[:, mid_y, :], dtype=np.float32)


def create_log_plot_xz_slice_from_dataset(
    X,
    sample_index,
    log_eps=1e-30,
    clip_negative_to_zero=True,
):
    """
    Create a log-scaled plot-oriented xz slice from a saved sample.

    Parameters
    ----------
    X : numpy.ndarray
        VDF samples with shape ``(n_samples, vx, vy, vz)``.
    sample_index : int
        Dataset sample index.
    log_eps : float, optional
        Small positive value added before log scaling.
    clip_negative_to_zero : bool, optional
        Whether to clip tiny negative values to zero before log scaling.

    Returns
    -------
    numpy.ndarray
        Log-scaled xz slice.
    """

    return create_log_slice(
        physical_slice=extract_plot_xz_slice_from_dataset(
            X=X,
            sample_index=sample_index,
        ),
        log_eps=log_eps,
        clip_negative_to_zero=clip_negative_to_zero,
    )


def create_log_slice(physical_slice, log_eps=1e-30, clip_negative_to_zero=True):
    """
    Convert a physical VDF slice to log10 scale.

    Parameters
    ----------
    physical_slice : numpy.ndarray
        Physical VDF slice.
    log_eps : float, optional
        Small positive value added before log scaling.
    clip_negative_to_zero : bool, optional
        Whether to clip negative values to zero before log scaling.

    Returns
    -------
    numpy.ndarray
        Log-scaled slice.
    """

    physical_slice = np.asarray(physical_slice, dtype=np.float32)
    if clip_negative_to_zero:
        physical_slice = np.maximum(physical_slice, 0.0)

    return np.log10(physical_slice + float(log_eps)).astype(
        np.float32,
        copy=False,
    )


def normalize_log_slice(slice_log, mean, std):
    """
    Normalize one log-scaled VDF slice.

    Parameters
    ----------
    slice_log : numpy.ndarray
        Log-scaled VDF slice.
    mean : float
        Training-set mean.
    std : float
        Training-set standard deviation.

    Returns
    -------
    numpy.ndarray
        Normalized slice.
    """

    return ((slice_log - float(mean)) / float(std)).astype(
        np.float32,
        copy=False,
    )


def denormalize_log_slice(normalized_slice, mean, std):
    """
    Convert a normalized log slice back to log10 scale.

    Parameters
    ----------
    normalized_slice : numpy.ndarray
        Normalized log-scaled VDF slice.
    mean : float
        Training-set mean.
    std : float
        Training-set standard deviation.

    Returns
    -------
    numpy.ndarray
        Log10-scaled VDF slice.
    """

    return np.asarray(normalized_slice, dtype=np.float32) * float(std) + float(mean)


def log_slice_to_physical(slice_log, log_eps=1e-30):
    """
    Convert a log10 VDF slice back to physical values.

    Parameters
    ----------
    slice_log : numpy.ndarray
        Log10-scaled VDF slice.
    log_eps : float, optional
        Small positive offset used during log scaling.

    Returns
    -------
    numpy.ndarray
        Non-negative physical VDF slice.
    """

    slice_log = np.asarray(slice_log, dtype=np.float64)
    slice_log = np.clip(slice_log, -300.0, 30.0)
    physical_slice = np.power(10.0, slice_log) - float(log_eps)
    physical_slice = np.maximum(physical_slice, 0.0)

    return np.asarray(physical_slice, dtype=np.float32)


def normalized_log_slice_to_physical(
    normalized_slice,
    mean,
    std,
    log_eps=1e-30,
):
    """
    Convert a normalized log slice back to physical values.

    Parameters
    ----------
    normalized_slice : numpy.ndarray
        Normalized log-scaled VDF slice.
    mean : float
        Training-set mean.
    std : float
        Training-set standard deviation.
    log_eps : float, optional
        Small positive offset used during log scaling.

    Returns
    -------
    numpy.ndarray
        Non-negative physical VDF slice.
    """

    return log_slice_to_physical(
        slice_log=denormalize_log_slice(
            normalized_slice=normalized_slice,
            mean=mean,
            std=std,
        ),
        log_eps=log_eps,
    )


def compute_log_slice_normalization(
    X,
    indices,
    log_eps=1e-30,
    clip_negative_to_zero=True,
    batch_size=64,
):
    """
    Compute train-only normalization statistics for log xz slices.

    Parameters
    ----------
    X : numpy.ndarray
        VDF samples with shape ``(n_samples, vx, vy, vz)``.
    indices : array-like of int
        Sample indices used for normalization.
    log_eps : float, optional
        Small positive value added before log scaling.
    clip_negative_to_zero : bool, optional
        Whether to clip negative values to zero before log scaling.
    batch_size : int, optional
        Number of samples processed at once.

    Returns
    -------
    dict
        Normalization statistics.
    """

    indices = np.asarray(indices, dtype=int)
    if len(indices) == 0:
        raise ValueError("Cannot compute normalization from zero samples")

    total_sum = 0.0
    total_sum_sq = 0.0
    total_count = 0
    log_min = np.inf
    log_max = -np.inf

    for batch_indices in iter_index_batches(indices, batch_size):
        batch = [
            create_log_plot_xz_slice_from_dataset(
                X=X,
                sample_index=sample_index,
                log_eps=log_eps,
                clip_negative_to_zero=clip_negative_to_zero,
            )
            for sample_index in batch_indices
        ]
        batch = np.asarray(batch, dtype=np.float64)

        total_sum += float(batch.sum())
        total_sum_sq += float(np.square(batch).sum())
        total_count += int(batch.size)
        log_min = min(log_min, float(batch.min()))
        log_max = max(log_max, float(batch.max()))

    mean = total_sum / total_count
    variance = (total_sum_sq / total_count) - mean**2
    variance = max(variance, 0.0)
    std = float(np.sqrt(variance))
    if std == 0.0:
        std = 1.0

    return {
        "mean": float(mean),
        "std": std,
        "log_min": float(log_min),
        "log_max": float(log_max),
        "n_samples": int(len(indices)),
        "n_values": int(total_count),
    }


def compute_cached_log_slice_normalization(X_log, indices, batch_size=64):
    """
    Compute train-only normalization statistics from cached log slices.

    Parameters
    ----------
    X_log : numpy.ndarray
        Cached log-scaled xz slices with shape ``(n_samples, 1, vz, vx)``.
    indices : array-like of int
        Sample indices used for normalization.
    batch_size : int, optional
        Number of samples processed at once.

    Returns
    -------
    dict
        Normalization statistics.
    """

    indices = np.asarray(indices, dtype=int)
    if len(indices) == 0:
        raise ValueError("Cannot compute normalization from zero samples")

    total_sum = 0.0
    total_sum_sq = 0.0
    total_count = 0
    log_min = np.inf
    log_max = -np.inf

    for batch_indices in iter_index_batches(indices, batch_size):
        batch = np.asarray(X_log[batch_indices, 0], dtype=np.float64)

        total_sum += float(batch.sum())
        total_sum_sq += float(np.square(batch).sum())
        total_count += int(batch.size)
        log_min = min(log_min, float(batch.min()))
        log_max = max(log_max, float(batch.max()))

    mean = total_sum / total_count
    variance = (total_sum_sq / total_count) - mean**2
    variance = max(variance, 0.0)
    std = float(np.sqrt(variance))
    if std == 0.0:
        std = 1.0

    return {
        "mean": float(mean),
        "std": std,
        "log_min": float(log_min),
        "log_max": float(log_max),
        "n_samples": int(len(indices)),
        "n_values": int(total_count),
    }


def iter_index_batches(indices, batch_size):
    """
    Iterate sample indices in fixed-size batches.

    Parameters
    ----------
    indices : array-like of int
        Sample indices.
    batch_size : int
        Number of indices per batch.

    Yields
    ------
    numpy.ndarray
        Batch of sample indices.
    """

    indices = np.asarray(indices, dtype=int)
    batch_size = int(batch_size)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    for start in range(0, len(indices), batch_size):
        yield indices[start:start + batch_size]
