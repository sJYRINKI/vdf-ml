"""Stage 1: open saved PCA representations and yield bounded feature batches.

This stage follows the PCA workflow entry and precedes global feature
scaling. It keeps ``X.npy`` or ``X_hermite.npy`` memory mapped for all PCA
passes and prepares consecutive sample batches without ever constructing the
complete sample-by-feature matrix.

Raw batches retain every ``(vx, vy, vz)`` VDF cell, apply the configured
positive floor and ``log10``, and flatten in stable C order. Hermite batches
retain every physical-VDF coefficient in saved ``(n_x, n_y, n_z)`` order or
optional rotated ``(n_parallel, n_perp1, n_perp2)`` order and flatten without
another numerical transformation. Labels and other metadata are not passed
to the iterator.
"""

import numpy as np

from src.representations.step_01_load_saved_representation import (
    load_saved_representation,
)
from src.representations.step_02_prepare_raw_input import (
    prepare_raw_input,
)


def load_pca_feature_source(dataset_dir, representation):
    """Open one memory-mapped representation for the complete PCA workflow.

    The returned resource owner remains open while stages 2 and 3 recreate
    the batch iterator for their statistics, basis-fitting, and score passes.
    This lifetime avoids copying every prepared row merely to close the source
    before numerical fitting.

    Parameters
    ----------
    dataset_dir : str or pathlib.Path
        Directory containing aligned metadata and the requested saved array.
    representation : {"raw", "hermite"}
        Representation whose samples are processed in saved row order.

    Returns
    -------
    LoadedRepresentation
        Context-managed memory map, aligned metadata, and saved-array
        description.
    """

    return load_saved_representation(dataset_dir, representation)


def iter_pca_feature_batches(
    representation_array,
    representation,
    batch_size,
    log_eps=None,
):
    """Yield prepared PCA feature rows without materializing the dataset.

    The saved representation is memory mapped and read in consecutive sample
    batches. Raw samples retain the complete ``(vx, vy, vz)`` VDF, receive
    the positive logarithmic transform, and are flattened only inside the
    current batch. Hermite samples retain the complete coefficient cube and
    are flattened without further transformation.

    Recreating this iterator for each PCA pass keeps prepared-feature memory
    proportional to ``batch_size * n_features`` rather than
    ``n_samples * n_features``. This is necessary for full-resolution raw
    VDFs, where one complete feature matrix can require hundreds of
    gibibytes.

    Parameters
    ----------
    representation_array : numpy.ndarray
        Read-only memory-mapped raw or Hermite array. Its first dimension is
        saved sample order.
    representation : {"raw", "hermite"}
        Saved representation converted into PCA feature rows.
    batch_size : int
        Number of consecutive samples prepared at one time.
    log_eps : float, optional
        Positive floor applied to raw VDF cells before ``log10``. Hermite
        coefficients do not use this value.

    Yields
    ------
    start : int
        Inclusive row index in the saved dataset.
    stop : int
        Exclusive row index in the saved dataset.
    features : numpy.ndarray
        Writable C-order feature rows with shape
        ``(stop - start, n_features)``.

    Notes
    -----
    The iterator preserves sample and feature order. It never writes a
    prepared feature cache and never removes a velocity-space axis.
    """

    sample_count = int(representation_array.shape[0])
    batch_size = int(batch_size)
    for start in range(0, sample_count, batch_size):
        stop = min(start + batch_size, sample_count)
        if representation == "raw":
            batch = prepare_raw_input(
                representation_array[start:stop],
                log_eps=log_eps,
            )
        else:
            batch = np.array(
                representation_array[start:stop],
                dtype=representation_array.dtype,
                order="C",
                copy=True,
            )
        yield start, stop, batch.reshape(stop - start, -1, order="C")
        del batch


def iter_pca_feature_shard_batches(
    representation_array,
    representation,
    batch_size,
    feature_start,
    feature_stop,
    log_eps=None,
):
    """Yield one contiguous feature shard from every prepared sample batch.

    Each distributed rank receives all saved sample rows but owns only the
    flattened feature columns in ``[feature_start, feature_stop)``. Slicing
    precedes the owned working copy and raw logarithm so host and device
    memory both scale with the local shard width.

    Parameters
    ----------
    representation_array : numpy.ndarray
        Read-only memory-mapped raw or Hermite array with samples first.
    representation : {"raw", "hermite"}
        Preparation branch applied to the local flattened columns.
    batch_size : int
        Maximum number of consecutive saved samples in one batch.
    feature_start : int
        Inclusive flattened feature-column offset owned by this rank.
    feature_stop : int
        Exclusive flattened feature-column offset owned by this rank.
    log_eps : float, optional
        Positive floor applied before ``log10`` for raw VDF values.

    Yields
    ------
    start : int
        Inclusive saved sample index.
    stop : int
        Exclusive saved sample index.
    features : numpy.ndarray
        Writable C-order local features with shape
        ``(stop - start, feature_stop - feature_start)``.
    """

    sample_count = int(representation_array.shape[0])
    batch_size = int(batch_size)
    feature_start = int(feature_start)
    feature_stop = int(feature_stop)
    for start in range(0, sample_count, batch_size):
        stop = min(start + batch_size, sample_count)
        row_count = stop - start
        source_rows = representation_array[start:stop].reshape(
            row_count,
            -1,
            order="C",
        )
        source_shard = source_rows[:, feature_start:feature_stop]
        if representation == "raw":
            batch = np.array(
                source_shard,
                dtype=np.result_type(representation_array.dtype, np.float32),
                order="C",
                copy=True,
            )
            np.maximum(batch, float(log_eps), out=batch)
            np.log10(batch, out=batch)
        else:
            batch = np.array(
                source_shard,
                dtype=representation_array.dtype,
                order="C",
                copy=True,
            )
        yield start, stop, batch
        del batch, source_shard, source_rows


def get_pca_feature_shape(representation_array):
    """Return sample and flattened-feature counts without preparing rows.

    Parameters
    ----------
    representation_array : numpy.ndarray
        Saved raw or Hermite array with samples on the first axis.

    Returns
    -------
    tuple of int
        ``(n_samples, n_features)`` inferred from the memory-map shape.
    """

    return (
        int(representation_array.shape[0]),
        int(np.prod(representation_array.shape[1:], dtype=np.int64)),
    )


def get_pca_feature_dtype(representation_array, representation):
    """Return the prepared feature dtype without reading a sample batch.

    Parameters
    ----------
    representation_array : numpy.ndarray
        Saved raw or Hermite array.
    representation : {"raw", "hermite"}
        Preparation branch that determines whether a raw logarithmic working
        copy must support floating-point values.

    Returns
    -------
    numpy.dtype
        Dtype used by every yielded feature batch.
    """

    if representation == "raw":
        return np.dtype(
            np.result_type(representation_array.dtype, np.float32)
        )
    return np.dtype(representation_array.dtype)
