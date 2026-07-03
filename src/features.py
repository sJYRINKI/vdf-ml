import os
import numpy as np
from joblib import Parallel, delayed
from src.vdf_helpers import create_xz_slice


def create_features(X, downsample_factor=8, log_eps=1e-30, n_jobs=1):
    """
    Convert 3D VDF into flattened 2D features.

    Parameters
    ----------
    X : numpy.ndarray
        VDF.
    downsample_factor : int, optional
        Factor used to downsample the xz slice.
    log_eps : float, optional
        Small value added before log scaling.
    n_jobs : int, optional
        Number of parallel workers used for feature extraction.

    Returns
    -------
    numpy.ndarray
        Feature matrix.
    """
    n_jobs = int(n_jobs)

    if n_jobs == 1:
        return create_features_chunk(
            X,
            downsample_factor=downsample_factor,
            log_eps=log_eps,
        )

    worker_count = (os.cpu_count() or 1) if n_jobs < 0 else n_jobs
    chunks = np.array_split(X, min(len(X), worker_count))

    feature_chunks = Parallel(
        n_jobs=n_jobs,
        prefer="threads"
    )(
        delayed(create_features_chunk)(
            chunk,
            downsample_factor=downsample_factor,
            log_eps=log_eps,
        )
        for chunk in chunks
        if len(chunk) > 0
    )

    return np.concatenate(feature_chunks)


def create_features_chunk(X, downsample_factor=8, log_eps=1e-30):
    """
    Convert a chunk of 3D VDF samples into flattened 2D features.

    Parameters
    ----------
    X : numpy.ndarray
        VDF samples.
    downsample_factor : int, optional
        Factor used to downsample the xz slice.
    log_eps : float, optional
        Small value added before log scaling.

    Returns
    -------
    numpy.ndarray
        Feature matrix for the chunk.
    """
    features = [
        create_feature(
            vdf,
            downsample_factor=downsample_factor,
            log_eps=log_eps,
        )
        for vdf in X
    ]

    return np.array(features, dtype=np.float32)


def create_feature(vdf, downsample_factor=8, log_eps=1e-30):
    """
    Convert one 3D VDF into flattened 2D features.

    Parameters
    ----------
    vdf : numpy.ndarray
        One VDF sample.
    downsample_factor : int, optional
        Factor used to downsample the xz slice.
    log_eps : float, optional
        Small value added before log scaling.

    Returns
    -------
    numpy.ndarray
        Feature vector.
    """
    xz_slice = create_xz_slice(vdf)
    xz_slice = np.where(xz_slice > 0, xz_slice, log_eps)
    xz_slice = np.log10(xz_slice)

    xz_slice = downsample_2d(
        xz_slice,
        factor=downsample_factor,
    )

    return xz_slice.ravel()


def downsample_2d(array, factor):
    """
    Downsample a 2D array by block averaging.

    Parameters
    ----------
    array : numpy.ndarray
        Input 2D array.
    factor : int
        Downsampling factor.

    Returns
    -------
    numpy.ndarray
        Downsampled 2D array.
    """

    nx, ny = array.shape

    nx_trim = nx - (nx % factor)
    ny_trim = ny - (ny % factor)

    array = array[:nx_trim, :ny_trim]

    return array.reshape(
        nx_trim // factor,
        factor,
        ny_trim // factor,
        factor,
    ).mean(axis=(1, 3))


def create_features_from_log_slice_cache(
    X_log,
    indices,
    downsample_factor=8,
    batch_size=64,
    sample_normalization="none",
    sample_norm_eps=1e-6,
):
    """
    Create flattened features from cached log-scaled xz slices.

    Parameters
    ----------
    X_log : numpy.ndarray
        Cached log-scaled plot-oriented xz slices with shape
        ``(n_samples, 1, vz, vx)``.
    indices : array-like of int
        Sample indices to extract.
    downsample_factor : int, optional
        Factor used to downsample the xz slice.
    batch_size : int, optional
        Number of cached slices to process at once.
    sample_normalization : {"none", "standard"}, optional
        Optional per-sample feature normalization.
    sample_norm_eps : float, optional
        Small value used to avoid division by zero during sample normalization.

    Returns
    -------
    numpy.ndarray
        Feature matrix in the same order as indices.
    """

    indices = np.asarray(indices, dtype=int)
    features = None
    write_start = 0

    for batch_indices in _iter_index_batches(indices, batch_size):
        features_batch = create_log_slice_cache_feature_batch(
            X_log=X_log,
            indices=batch_indices,
            downsample_factor=downsample_factor,
            sample_normalization=sample_normalization,
            sample_norm_eps=sample_norm_eps,
        )

        if features is None:
            features = np.empty(
                (len(indices), features_batch.shape[1]),
                dtype=np.float32,
            )

        write_end = write_start + len(features_batch)
        features[write_start:write_end] = features_batch
        write_start = write_end

    if features is None:
        return np.empty((0, 0), dtype=np.float32)

    return features


def create_log_slice_cache_feature_batch(
    X_log,
    indices,
    downsample_factor=8,
    sample_normalization="none",
    sample_norm_eps=1e-6,
):
    """
    Create one flattened feature batch from cached log-scaled xz slices.

    Parameters
    ----------
    X_log : numpy.ndarray
        Cached log-scaled plot-oriented xz slices with shape
        ``(n_samples, 1, vz, vx)``.
    indices : array-like of int
        Sample indices to extract.
    downsample_factor : int, optional
        Factor used to downsample the xz slice.
    sample_normalization : {"none", "standard"}, optional
        Optional per-sample feature normalization.
    sample_norm_eps : float, optional
        Small value used to avoid division by zero during sample normalization.

    Returns
    -------
    numpy.ndarray
        Feature batch.
    """

    indices = np.asarray(indices, dtype=int)
    batch_slice = _create_contiguous_slice(indices)
    sample_selection = batch_slice if batch_slice is not None else indices

    slices = np.asarray(X_log[sample_selection, 0], dtype=np.float32)
    # The cache is plot-oriented (vz, vx). Existing CNN/PCA features use
    # create_xz_slice orientation (vx, vz), so transpose back before flattening.
    slices = np.swapaxes(slices, 1, 2).copy()
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


def _iter_index_batches(indices, batch_size):
    indices = np.asarray(indices, dtype=int)
    for start in range(0, len(indices), int(batch_size)):
        yield indices[start:start + int(batch_size)]


def _create_contiguous_slice(indices):
    indices = np.asarray(indices)
    if len(indices) == 0:
        return slice(0, 0)

    if not np.issubdtype(indices.dtype, np.integer):
        return None

    start = int(indices[0])
    stop = start + len(indices)

    if start < 0 or int(indices[-1]) != stop - 1:
        return None

    if np.all(indices == np.arange(start, stop)):
        return slice(start, stop)

    return None
