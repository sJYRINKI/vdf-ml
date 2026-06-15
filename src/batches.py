import numpy as np
from src.features import create_features



def iter_index_batches(indices, batch_size):
    """
    Iterate over index batches.

    Parameters
    ----------
    indices : numpy.ndarray
        Sample indices.
    batch_size : int
        Number of indices per batch.

    Yields
    ------
    numpy.ndarray
        Batch of indices.
    """
    indices = np.asarray(indices)

    for start in range(0, len(indices), int(batch_size)):
        yield indices[start:start + int(batch_size)]

def iter_array_batches(X, indices=None, batch_size=64):
    """
    Iterate over array batches.

    Parameters
    ----------
    X : numpy.ndarray
        Dataset array.
    indices : array-like, optional
        Sample indices to iterate over.
    batch_size : int, optional
        Number of samples per batch.
    Yields
    ------
    batch_indices : numpy.ndarray
        Indices included in the batch.
    batch : numpy.ndarray
        Batch of samples.
    """
    if indices is None:
        indices = np.arange(X.shape[0])

    for batch_indices in iter_index_batches(indices, batch_size):
        yield batch_indices, X[batch_indices]

def predict_in_batches(model_pipeline, X, indices, downsample_factor, batch_size, n_jobs=1):
    """
    Predict labels in batches.

    Parameters
    ----------
    model_pipeline : sklearn.pipeline.Pipeline
        Pipeline containing the trained model and any preprocessing steps.
    X : numpy.ndarray
        VDF samples.
    indices : array-like of int
        Sample indices to predict.
    downsample_factor : int
        Factor used to downsample the xz slice.
    batch_size : int, optional
        Number of samples per batch.
    n_jobs : int, optional
        Number of parallel workers used for feature extraction.

    Returns
    -------
    numpy.ndarray
        Predicted labels.
    """
    y_pred_batches = []

    for batch_indices in iter_index_batches(indices, batch_size):
        features_batch = create_features(
            X[batch_indices],
            downsample_factor=downsample_factor,
            n_jobs=n_jobs,
        )

        y_pred_batch = model_pipeline.predict(features_batch)
        y_pred_batches.append(y_pred_batch)

    return np.concatenate(y_pred_batches)

def create_features_in_batches(X, indices, downsample_factor, batch_size, n_jobs=1, log_eps=1e-30):
    """
    Create a full feature matrix by reading VDF samples in batches.

    Parameters
    ----------
    X : numpy.ndarray
        VDF samples.
    indices : array-like of int
        Sample indices to extract.
    downsample_factor : int
        Factor used to downsample the xz slice.
    batch_size : int
        Number of raw VDF samples to process at once.
    n_jobs : int, optional
        Number of parallel workers used inside each batch.
    log_eps : float, optional
        Small value added before log scaling.

    Returns
    -------
    numpy.ndarray
        Feature matrix in the same order as indices.
    """
    feature_batches = []

    for batch_indices in iter_index_batches(indices, batch_size):
        features_batch = create_features(
            X[batch_indices],
            downsample_factor=downsample_factor,
            log_eps=log_eps,
            n_jobs=n_jobs,
        )
        feature_batches.append(features_batch)

    return np.concatenate(feature_batches, axis=0)