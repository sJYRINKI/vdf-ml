"""Stage 2: prepare complete raw VDF inputs for model workflows.

This stage follows saved-representation loading for the raw branch and
precedes PCA scaling or CNN normalization. It applies a positive floor and
``log10`` to every velocity-space cell in the saved
``(sample, vx, vy, vz)`` array.

PCA flattens each transformed volume in stable C order. CNN training and
prediction preserve all three velocity axes and add a channel dimension
later. No velocity-plane slicing, projection, cropping, or downsampling
occurs in this stage.
"""

import numpy as np

from src import RAW

RAW_AXIS_ORDER = ("vx", "vy", "vz")


def prepare_raw_input(vdfs, *, log_eps):
    """Convert complete physical VDF volumes to logarithmic model inputs.

    The raw representation keeps every velocity-space cell in the saved
    ``(vx, vy, vz)`` distribution. A positive floor is applied before
    ``log10`` so values at or below the configured numerical floor do not
    produce undefined logarithms. PCA later flattens the complete volume,
    while CNN training adds a channel dimension and preserves all three
    velocity axes for ``Conv3d``.

    Parameters
    ----------
    vdfs : numpy.ndarray
        One VDF with shape ``(vx, vy, vz)`` or a batch with shape
        ``(n_samples, vx, vy, vz)``. Values are physical phase-space
        densities in ``s^3 m^-6``.
    log_eps : float
        Positive floor in ``s^3 m^-6`` applied to velocity-space values
        below it before the logarithm.

    Returns
    -------
    numpy.ndarray
        Writable logarithmic VDF array with the same shape and
        ``(vx, vy, vz)`` axis order as ``vdfs``.

    Notes
    -----
    This function deliberately performs no velocity-plane slicing,
    projection, cropping, or downsampling. The input array is not mutated.
    """

    values = np.asanyarray(vdfs)
    working = np.array(
        values,
        dtype=np.result_type(values.dtype, np.float32),
        order="C",
        copy=True,
    )
    np.maximum(working, float(log_eps), out=working)
    np.log10(working, out=working)
    return working


def describe_raw_input(raw, *, log_eps):
    """Describe complete-volume raw preprocessing without preparing rows.

    CNN tensor specifications use this description while PCA prepares the
    same complete logarithmic volumes through its bounded Stage 1 iterator.
    Reading only shape and dtype information avoids allocating a
    sample-by-feature matrix.

    Parameters
    ----------
    raw : numpy.ndarray
        Physical VDF samples in ``s^3 m^-6`` with shape
        ``(n_samples, vx, vy, vz)`` in saved sample order.
    log_eps : float
        Positive phase-space-density floor applied to every working value
        below it before ``log10``.

    Returns
    -------
    dict
        Raw representation axes, complete volume shape, floor, and
        source-array identity.

    Notes
    -----
    The description preserves all ``vx * vy * vz`` cells. It performs no
    logarithm, flattening, slicing, cropping, or data copy.
    """

    raw = np.asanyarray(raw)
    volume_shape = tuple(int(value) for value in raw.shape[1:])
    feature_count = int(np.prod(volume_shape))
    return {
        "representation": "raw",
        "representation_version": RAW,
        "source_array_filename": "X.npy",
        "source_axis_order": list(RAW_AXIS_ORDER),
        "source_shape": [int(value) for value in raw.shape],
        "source_dtype": np.dtype(raw.dtype).str,
        "output_axis_order": list(RAW_AXIS_ORDER),
        "log_floor": float(log_eps),
        "volume_shape": list(volume_shape),
        "flattened_feature_count": feature_count,
    }
