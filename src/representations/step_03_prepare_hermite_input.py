"""Stage 3: describe saved physical-VDF Hermite coefficient inputs.

This alternative representation branch follows saved-array loading. CNN and
autoencoder samples retain each complete dimensional coefficient cube, while
PCA flattens the same saved values in stable C order. The stored array shape
is authoritative for the number of modes along every coefficient axis.
"""

import numpy as np


def describe_hermite_input(
    coefficients,
    *,
    rotated,
):
    """Describe a complete saved physical-VDF Hermite representation.

    The saved coefficient array is produced by projecting physical linear
    ``f(v)`` onto physically normalized Hermite bases. CNN and autoencoder
    loaders use this metadata to retain every mode with its original
    dimensional value; the saved shape supplies the actual order rather than
    a training configuration or global constant.

    Parameters
    ----------
    coefficients : numpy.ndarray
        Saved float coefficient cubes with shape
        ``(n_samples, order, order, order)``. Unrotated axes are
        ``(n_x, n_y, n_z)``; optionally rotated axes are
        ``(n_parallel, n_perp1, n_perp2)``.
    rotated : bool
        Whether extraction rotated the physical VDF into the local magnetic
        frame before moment calculation and projection.

    Returns
    -------
    dict
        Actual volume shape and order, physical coefficient axes, rotation
        setting, source dtype, and complete flattened feature count.

    Notes
    -----
    The function reads only array metadata. Dimensional Hermite coefficients
    are not logged, thresholded, normalized, transposed, sliced, or copied.
    """

    coefficients = np.asanyarray(coefficients)
    volume_shape = tuple(int(value) for value in coefficients.shape[1:])
    order = int(volume_shape[0])
    axis_order = (
        ("n_parallel", "n_perp1", "n_perp2")
        if rotated
        else ("n_x", "n_y", "n_z")
    )
    return {
        "representation": "hermite",
        "source_array_filename": "X_hermite.npy",
        "representation_version": "hermite",
        "volume_shape": list(volume_shape),
        "hermite_order": order,
        "hermite_rotate": bool(rotated),
        "axis_order": [str(value) for value in axis_order],
        "source_shape": [int(value) for value in coefficients.shape],
        "source_dtype": np.dtype(coefficients.dtype).str,
        "flattened_feature_count": int(np.prod(volume_shape)),
    }
