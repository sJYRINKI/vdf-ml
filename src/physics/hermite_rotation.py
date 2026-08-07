"""Rotate physical VDFs onto an optional bulk-flow magnetic frame.

When extraction configuration enables Hermite rotation, this module builds
the supplied ``(parallel, perpendicular bulk flow, cross-product)`` frame,
expands the velocity bounds to contain all rotated corners, and resamples the
physical VDF with trilinear interpolation.  The returned grid axes determine
the coefficient order ``(n_parallel, n_perp1, n_perp2)``.  Unrotated Hermite
generation bypasses this module entirely.
"""

import numpy as np
from scipy.interpolate import RegularGridInterpolator


def build_rotation_matrix(magnetic_field, bulk_velocity):
    """Build the supplied global-to-rotated velocity transformation.

    The first row is the magnetic-field direction.  The second is the part
    of total bulk velocity perpendicular to that field, and the third is
    their cross product.  The matrix maps column vectors through
    ``v_rotated = rotation_matrix @ v_original``.  Dataset extraction and
    prediction call this only when ``hermite.rotate`` is true.

    Parameters
    ----------
    magnetic_field : array-like
        Same-cell magnetic field ``(Bx, By, Bz)``; only direction is used.
    bulk_velocity : array-like
        Same-cell total bulk flow ``(Vx, Vy, Vz)`` in metres per second.

    Returns
    -------
    numpy.ndarray
        Float64 orthonormal matrix with rows ``(parallel, perp1, perp2)``.

    Notes
    -----
    The operation intentionally supplies no fallback frame for parallel
    field and velocity vectors; direct NumPy numerical behavior applies.
    """

    magnetic_field = np.asarray(magnetic_field, dtype=np.float64)
    bulk_velocity = np.asarray(bulk_velocity, dtype=np.float64)
    parallel = magnetic_field / np.linalg.norm(magnetic_field)
    perpendicular_1 = bulk_velocity - np.dot(bulk_velocity, parallel) * parallel
    perpendicular_1 /= np.linalg.norm(perpendicular_1)
    perpendicular_2 = np.cross(parallel, perpendicular_1)
    return np.stack((parallel, perpendicular_1, perpendicular_2), axis=0)


def rotate_velocity_bounds(velocity_limits_mps, rotation_matrix):
    """Find endpoint bounds containing the rotated velocity cuboid.

    All eight corners of the original axis-aligned velocity extent are
    transformed.  Their componentwise minima and maxima define the expanded
    rotated grid used for trilinear interpolation.

    Parameters
    ----------
    velocity_limits_mps : array-like
        Original extent ``[vxmin, vymin, vzmin, vxmax, vymax, vzmax]`` in
        metres per second.
    rotation_matrix : numpy.ndarray
        Global-to-rotated ``(3, 3)`` frame with row axis order
        ``(parallel, perp1, perp2)``.

    Returns
    -------
    numpy.ndarray
        Rotated endpoint extent in metres per second.
    """

    limits = np.asarray(velocity_limits_mps, dtype=np.float64)
    low = limits[:3]
    high = limits[3:]
    corners = np.asarray(
        [
            [
                high[0] if corner_index & 1 else low[0],
                high[1] if corner_index & 2 else low[1],
                high[2] if corner_index & 4 else low[2],
            ]
            for corner_index in range(8)
        ],
        dtype=np.float64,
    )
    rotated_corners = corners @ np.asarray(rotation_matrix, dtype=np.float64).T
    return np.concatenate(
        (rotated_corners.min(axis=0), rotated_corners.max(axis=0))
    )


def calculate_rotated_shape(shape, velocity_limits_mps, rotated_limits_mps):
    """Choose a rotated grid shape at the original approximate spacing.

    Original endpoint spacing is ``(vmax - vmin) / (shape - 1)`` on each
    axis.  The expanded rotated extent is divided by those spacings, floored,
    and incremented once, exactly matching the supplied interpolation grid.

    Parameters
    ----------
    shape : sequence of int
        Original VDF shape ``(nx, ny, nz)``.
    velocity_limits_mps : array-like
        Original endpoint velocity extent in metres per second.
    rotated_limits_mps : array-like
        Rotated endpoint velocity extent in metres per second.

    Returns
    -------
    tuple of int
        Rotated ``(n_parallel, n_perp1, n_perp2)`` grid shape.
    """

    shape_array = np.asarray(shape, dtype=np.float64)
    limits = np.asarray(velocity_limits_mps, dtype=np.float64)
    rotated_limits = np.asarray(rotated_limits_mps, dtype=np.float64)
    original_spacing = (limits[3:] - limits[:3]) / (shape_array - 1.0)
    rotated_extent = rotated_limits[3:] - rotated_limits[:3]
    rotated_shape = np.floor(rotated_extent / original_spacing).astype(int) + 1
    return tuple(int(value) for value in rotated_shape)


def rotate_vdf(
    vdf,
    shape,
    velocity_limits_mps,
    magnetic_field,
    bulk_velocity,
):
    """Rotate and interpolate one physical VDF onto expanded bounds.

    The new endpoint grid covers the complete rotated original cuboid at
    approximately the original coordinate spacing.  New-grid coordinates
    are mapped back through the inverse orthonormal rotation and sampled by
    ``RegularGridInterpolator`` using linear interpolation and zero outside
    the original domain.  No density renormalization or density-tolerance
    calculation is performed.

    Parameters
    ----------
    vdf : numpy.ndarray
        Physical VDF in ``s^3 m^-6`` and ``[vx, vy, vz]`` order.
    shape : sequence of int
        Original VDF shape ``(nx, ny, nz)``.
    velocity_limits_mps : array-like
        Original endpoint velocity extent in metres per second.
    magnetic_field : array-like
        Same-cell magnetic field used for the parallel direction.
    bulk_velocity : array-like
        Same-cell total bulk flow in metres per second used for ``perp1``.

    Returns
    -------
    rotated_vdf : numpy.ndarray
        Interpolated physical VDF with rotated-grid shape and input dtype.
    rotated_shape : tuple of int
        Grid shape in ``(parallel, perp1, perp2)`` axis order.
    rotated_velocity_limits_mps : tuple of float
        Expanded rotated endpoint extent in metres per second.
    rotation_matrix : numpy.ndarray
        Global-to-rotated matrix with rows ``(parallel, perp1, perp2)``.
    """

    array = np.asarray(vdf)
    shape = tuple(int(value) for value in shape)
    limits = np.asarray(velocity_limits_mps, dtype=np.float64)
    rotation_matrix = build_rotation_matrix(magnetic_field, bulk_velocity)
    rotated_limits = rotate_velocity_bounds(limits, rotation_matrix)
    rotated_shape = calculate_rotated_shape(shape, limits, rotated_limits)
    new_axes = tuple(
        np.linspace(
            rotated_limits[axis],
            rotated_limits[axis + 3],
            rotated_shape[axis],
            dtype=np.float64,
        )
        for axis in range(3)
    )
    new_grid = np.stack(
        np.meshgrid(*new_axes, indexing="ij"),
        axis=-1,
    ).reshape(-1, 3)
    original_grid_coordinates = new_grid @ rotation_matrix
    original_axes = tuple(
        np.linspace(
            limits[axis],
            limits[axis + 3],
            shape[axis],
            dtype=np.float64,
        )
        for axis in range(3)
    )
    interpolator = RegularGridInterpolator(
        original_axes,
        array,
        method="linear",
        bounds_error=False,
        fill_value=0.0,
    )
    rotated_vdf = interpolator(original_grid_coordinates).reshape(rotated_shape)
    return (
        rotated_vdf.astype(array.dtype, copy=False),
        rotated_shape,
        tuple(float(value) for value in rotated_limits),
        rotation_matrix,
    )
