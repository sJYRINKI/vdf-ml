import os

import numpy as np
from joblib import Parallel, delayed
from src.vdf_helpers import create_xz_slice


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