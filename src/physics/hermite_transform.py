"""Transform physical VDFs into dimensional Hermite coefficient cubes.

The active ``hermite`` representation projects the supplied linear physical
VDF without logarithms, sparsity thresholds, clipping, or per-sample
coefficient normalization.  Unrotated inputs retain ``[vx, vy, vz]`` axis
order; callers may first use :mod:`src.physics.hermite_rotation` to obtain
``[parallel, perp1, perp2]`` input axes.  Extraction stores the resulting
float64 calculation once in its configured dataset dtype, while PCA, CNN,
autoencoder, and prediction preserve the complete saved cube.
"""

import numpy as np

from src.physics.hermite_basis import (
    DEFAULT_HERMITE_ORDER,
    compute_drift_velocity,
    compute_hermite_spectra,
    compute_thermal_velocity,
)


UNROTATED_HERMITE_AXIS_ORDER = ("n_x", "n_y", "n_z")
ROTATED_HERMITE_AXIS_ORDER = ("n_parallel", "n_perp1", "n_perp2")


def vdf_to_hermite(
    vdf,
    shape,
    velocity_limits_mps,
    order=DEFAULT_HERMITE_ORDER,
):
    """Project one physical VDF into the supplied Hermite representation.

    Dataset extraction and Hermite prediction call this function after
    choosing either the original physical velocity grid or the optional
    rotated grid.  Drift and isotropic thermal velocity are calculated from
    the linear VDF itself.  The same input frame then defines endpoint
    coordinates, physical-velocity basis normalization, and ``dv**3``
    quadrature for the complete coefficient cube.

    Parameters
    ----------
    vdf : numpy.ndarray
        Physical velocity-distribution function in ``s^3 m^-6`` with three
        velocity axes.  Values are projected exactly as supplied.
    shape : sequence of int
        Input VDF shape.  Axis order is ``(vx, vy, vz)`` without rotation or
        ``(parallel, perp1, perp2)`` after optional rotation.
    velocity_limits_mps : array-like
        Endpoint extent for the three input axes in metres per second.
    order : int, optional
        Number of Hermite modes per axis; the current default is 22.

    Returns
    -------
    numpy.ndarray
        Dimensional float64 coefficient cube with shape
        ``(order, order, order)``.  Axes follow the transform input:
        ``(n_x, n_y, n_z)`` unrotated or
        ``(n_parallel, n_perp1, n_perp2)`` after rotation.

    Notes
    -----
    Exactly zero physical density produces a zero drift, zero thermal scale,
    and zero coefficient cube.  No MinValue, logarithm, density tolerance,
    or transform diagnostics enter this operation.
    """

    array = np.asarray(vdf, dtype=np.float64)
    shape = tuple(int(value) for value in shape)
    order = int(order)
    drift_velocity = compute_drift_velocity(
        array,
        shape,
        velocity_limits_mps,
    )
    thermal_velocity = compute_thermal_velocity(
        array,
        shape,
        velocity_limits_mps,
        drift_velocity,
    )
    if thermal_velocity == 0.0:
        return np.zeros((order, order, order), dtype=float)
    return compute_hermite_spectra(
        array,
        shape,
        velocity_limits_mps,
        order,
        thermal_velocity,
        drift_velocity,
    )
