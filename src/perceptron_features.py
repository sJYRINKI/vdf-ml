import numpy as np
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

def create_perceptron_features(X, downsample_factor=8, log_eps=1e-30):
    """
    Convert 3D VDF into flattened 2D features.

    Parameters
    ----------
    X : numpy.ndarray
        VDF.
    downsample_features : int, optional
        Factor used to downsample the xz slice.
    log_eps : float, optional
        Small value added before log scaling.

    Returns
    -------
    numpy.ndarray
        Feature matrix.
    """
    features = []
    
    for vdf in X:
        xz_slice = create_xz_slice(vdf)
        xz_slice = np.where(xz_slice > 0, xz_slice, log_eps)
        xz_slice = np.log10(xz_slice)

        xz_slice = downsample_2d(
            xz_slice,
            factor=downsample_factor,
        )

        features.append(xz_slice.ravel())
    
    features = np.array(features, dtype=np.float32)

    return features

def standardize_features(features):
    """
    Standardize features to zero mean and to unit standard deviation.

    Parameters
    ----------
    features : numpy.ndarray
        Feature matrix.

    Returns
    -------
    features_scaled : numpy.ndarray
        Standardized feature matrix.
    mean : numpy.ndarray
        Feature means.
    std : numpy.ndarray
        Feature standard deviations.
    """
    mean = features.mean(axis=0)
    std = features.std(axis=0)

    std[std == 0] = 1.0

    features_scaled = (features - mean) / std

    return features_scaled, mean, std