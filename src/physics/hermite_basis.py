"""Calculate physical VDF moments and dimensional Hermite coefficients.

The Hermite representation projects a linear velocity-distribution function
``f(vx, vy, vz)`` in ``s^3 m^-6`` onto Gaussian-weighted physicists' Hermite
polynomials.  Velocity coordinates include both saved mesh endpoints, the
basis carries physical-velocity normalization through ``sqrt(vth)``, and the
three-dimensional projection uses the supplied ``dv**3`` quadrature.  Dataset
extraction, CNN and autoencoder inputs, and prediction all consume the
resulting ``(order, order, order)`` coefficient cube without another
coefficient normalization.
"""

import math

import numpy as np


DEFAULT_HERMITE_ORDER = 22


def hermite_polynomials(normalized_velocity, order):
    """Evaluate Gaussian-weighted physicists' Hermite polynomials.

    The recurrence evaluates ``H_n(x) exp(-x**2 / 2)`` for normalized
    velocity ``x = (v - u) / vth``.  Physical-velocity normalization is
    applied separately by :func:`normalized_hermite_basis` so this function
    reproduces the supplied polynomial arrays directly.

    Parameters
    ----------
    normalized_velocity : array-like
        Dimensionless one-dimensional velocity coordinates.
    order : int
        Number of Hermite modes, beginning with order zero.

    Returns
    -------
    numpy.ndarray
        Float64 values with shape ``(order, n_velocity_points)``.
    """

    coordinates = np.asarray(normalized_velocity, dtype=np.float64)
    polynomials = np.zeros((int(order), len(coordinates)), dtype=np.float64)
    polynomials[0] = np.exp(-0.5 * coordinates**2)
    if int(order) > 1:
        polynomials[1] = 2.0 * coordinates * np.exp(-0.5 * coordinates**2)
    for mode in range(2, int(order)):
        polynomials[mode] = (
            2.0 * coordinates * polynomials[mode - 1]
            - 2.0 * (mode - 1) * polynomials[mode - 2]
        )
    return polynomials


def normalized_hermite_basis(
    shape,
    velocity_limits_mps,
    order,
    thermal_velocity_mps,
    drift_velocity_mps,
    axis,
):
    """Build one physical-velocity Hermite basis on endpoint coordinates.

    The velocity axis is ``np.linspace(vmin, vmax, n_cells)`` in metres per
    second, including both configured endpoints.  Each Gaussian-weighted
    physicists' polynomial is divided by
    ``sqrt(2**n * n! * sqrt(pi) * vth)``.  This ``sqrt(vth)`` factor gives
    the basis its physical-velocity normalization and produces dimensional
    coefficients when the physical VDF is projected.

    Parameters
    ----------
    shape : sequence of int
        VDF shape ``(nx, ny, nz)`` in input velocity-axis order.
    velocity_limits_mps : array-like
        Endpoint extent ``[vxmin, vymin, vzmin, vxmax, vymax, vzmax]`` in
        metres per second.
    order : int
        Number of Hermite modes along this axis.
    thermal_velocity_mps : float
        Isotropic thermal standard deviation ``vth`` in metres per second.
    drift_velocity_mps : array-like
        VDF first-moment velocity ``(ux, uy, uz)`` in metres per second.
    axis : int
        Input velocity axis: ``0`` for x, ``1`` for y, or ``2`` for z.

    Returns
    -------
    numpy.ndarray
        Float64 basis with shape ``(order, shape[axis])``.
    """

    shape = tuple(int(value) for value in shape)
    limits = np.asarray(velocity_limits_mps, dtype=np.float64)
    drift = np.asarray(drift_velocity_mps, dtype=np.float64)
    thermal_velocity_mps = float(thermal_velocity_mps)
    velocity_axis = np.linspace(
        limits[axis],
        limits[axis + 3],
        shape[axis],
        dtype=np.float64,
    )
    normalized_axis = (velocity_axis - drift[axis]) / thermal_velocity_mps
    basis = hermite_polynomials(normalized_axis, int(order))
    for mode in range(int(order)):
        normalization = math.sqrt(
            (2**mode)
            * math.factorial(mode)
            * math.sqrt(math.pi)
            * thermal_velocity_mps
        )
        basis[mode] /= normalization
    return basis


def compute_drift_velocity(vdf, shape, velocity_limits_mps):
    """Calculate drift velocity from the physical VDF first moment.

    Endpoint ``vx``, ``vy``, and ``vz`` axes are weighted by the physical
    linear VDF.  The density quadrature uses the product of the three axis
    spacings, where each spacing is the corresponding extent divided by its
    sample count.  No logarithm, sparsity threshold, clipping, or fluid
    ``rho_v / rho`` value enters this Hermite moment.

    Parameters
    ----------
    vdf : numpy.ndarray
        Physical VDF in ``s^3 m^-6`` with shape ``(nx, ny, nz)`` and axis
        order ``[vx, vy, vz]`` or the optional rotated-grid equivalent.
    shape : sequence of int
        Three-dimensional VDF shape in input axis order.
    velocity_limits_mps : array-like
        Endpoint velocity extent in metres per second.

    Returns
    -------
    numpy.ndarray
        Float64 drift vector in metres per second.  Exactly zero density
        returns ``np.zeros(3)`` as specified by the supplied transform.
    """

    array = np.asarray(vdf, dtype=np.float64)
    shape = tuple(int(value) for value in shape)
    limits = np.asarray(velocity_limits_mps, dtype=np.float64)
    axes = tuple(
        np.linspace(
            limits[axis],
            limits[axis + 3],
            shape[axis],
            dtype=np.float64,
        )
        for axis in range(3)
    )
    spacings = (limits[3:] - limits[:3]) / np.asarray(shape, dtype=np.float64)
    cell_volume = float(np.prod(spacings))
    density = float(np.sum(array, dtype=np.float64)) * cell_volume
    if density == 0.0:
        return np.zeros(3, dtype=np.float64)
    momentum = np.asarray(
        [
            np.sum(array * axes[0][:, None, None], dtype=np.float64),
            np.sum(array * axes[1][None, :, None], dtype=np.float64),
            np.sum(array * axes[2][None, None, :], dtype=np.float64),
        ],
        dtype=np.float64,
    )
    return momentum * cell_volume / density


def compute_thermal_velocity(vdf, shape, velocity_limits_mps, drift_velocity_mps):
    """Calculate the supplied isotropic physical-VDF thermal velocity.

    The thermal scale is the square root of one third of the density-weighted
    peculiar-speed variance.  Endpoint velocity coordinates are used, while
    density and all diagonal pressure moments deliberately use ``dv**3``
    from the first velocity-axis extent and sample count.  The returned scale
    is the standard deviation itself, without a ``sqrt(2)`` factor.

    Parameters
    ----------
    vdf : numpy.ndarray
        Physical VDF in ``s^3 m^-6`` and input velocity-axis order.
    shape : sequence of int
        VDF shape ``(nx, ny, nz)`` in input axis order.
    velocity_limits_mps : array-like
        Endpoint velocity extent in metres per second.
    drift_velocity_mps : array-like
        First-moment drift ``(ux, uy, uz)`` in metres per second.

    Returns
    -------
    float
        Isotropic thermal standard deviation in metres per second.  Exactly
        zero density returns ``0.0``.
    """

    array = np.asarray(vdf, dtype=np.float64)
    shape = tuple(int(value) for value in shape)
    limits = np.asarray(velocity_limits_mps, dtype=np.float64)
    drift = np.asarray(drift_velocity_mps, dtype=np.float64)
    axes = tuple(
        np.linspace(
            limits[axis],
            limits[axis + 3],
            shape[axis],
            dtype=np.float64,
        )
        for axis in range(3)
    )
    dv = (limits[3] - limits[0]) / shape[0]
    dv_cubed = float(dv**3)
    density = float(np.sum(array, dtype=np.float64)) * dv_cubed
    if density == 0.0:
        return 0.0
    diagonal_pressure = np.asarray(
        [
            np.sum(
                array * (axes[0][:, None, None] - drift[0]) ** 2,
                dtype=np.float64,
            ),
            np.sum(
                array * (axes[1][None, :, None] - drift[1]) ** 2,
                dtype=np.float64,
            ),
            np.sum(
                array * (axes[2][None, None, :] - drift[2]) ** 2,
                dtype=np.float64,
            ),
        ],
        dtype=np.float64,
    ) * dv_cubed
    return float(np.sqrt(np.sum(diagonal_pressure) / (3.0 * density)))


def compute_hermite_spectra(
    vdf,
    shape,
    velocity_limits_mps,
    order,
    thermal_velocity_mps,
    drift_velocity_mps,
):
    """Project one physical VDF onto separable Hermite bases.

    The supplied optimized contraction projects linear ``f(v)`` directly.
    Basis and coefficient accumulation use float64, and the result is
    multiplied by ``dv**3`` from the first velocity-axis extent and sample
    count.  No thresholding, logarithm, density normalization, or
    coefficient normalization is applied.

    Parameters
    ----------
    vdf : numpy.ndarray
        Physical VDF in ``s^3 m^-6`` with three input velocity axes.
    shape : sequence of int
        VDF shape in the same input axis order.
    velocity_limits_mps : array-like
        Endpoint velocity extent in metres per second.
    order : int
        Number of Hermite modes per input axis.
    thermal_velocity_mps : float
        Isotropic thermal standard deviation in metres per second.
    drift_velocity_mps : array-like
        Physical VDF drift in metres per second.

    Returns
    -------
    numpy.ndarray
        Dimensional float64 coefficients with shape
        ``(order, order, order)``.  Output axes correspond in order to
        transform input axes 0, 1, and 2: ``(n_x, n_y, n_z)`` without
        rotation and ``(n_parallel, n_perp1, n_perp2)`` after rotation.
    """

    array = np.asarray(vdf, dtype=np.float64)
    shape = tuple(int(value) for value in shape)
    limits = np.asarray(velocity_limits_mps, dtype=np.float64)
    bases = tuple(
        normalized_hermite_basis(
            shape,
            limits,
            int(order),
            float(thermal_velocity_mps),
            drift_velocity_mps,
            axis,
        )
        for axis in range(3)
    )
    spectra = np.einsum(
        "ijk,ni,mj,lk->nml",
        array,
        bases[0],
        bases[1],
        bases[2],
        optimize=True,
    )
    dv = (limits[3] - limits[0]) / shape[0]
    return np.asarray(spectra * dv**3, dtype=np.float64)
