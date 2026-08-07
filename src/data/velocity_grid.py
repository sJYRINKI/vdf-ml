"""Describe velocity meshes and interpolate complete physical VDFs.

This module owns the ``[vx, vy, vz]`` velocity-grid descriptor used by
extraction, representations, and prediction. Grid extents are expressed in
metres per second. Raw CNN prediction uses cell-centred trilinear
interpolation to reproduce the complete training-grid VDF before logarithmic
preprocessing.
"""

import numpy as np


VELOCITY_AXIS_ORDER = ("vx", "vy", "vz")


def create_velocity_grid_descriptor(
    reader,
    pop,
    *,
    velocity_mesh_shape=None,
):
    """Create a velocity-grid descriptor from an open VLSV reader.

    Dense extraction calls this once per source mesh, and dataset planning
    stores the resulting descriptor for all aligned rows. It combines block
    dimensions with physical extents and records the saved middle-``vy`` cell
    used by raw representation consumers.

    Parameters
    ----------
    reader : analysator.vlsvfile.VlsvReader
        Open VLSV reader that owns the selected velocity mesh.
    pop : str
        Particle population already selected for the operation.
    velocity_mesh_shape : tuple of int, optional
        Positive ``(vx, vy, vz)`` mesh shape already resolved for ``reader``
        and ``pop``. When omitted, the reader's mesh and block sizes define
        the shape.

    Returns
    -------
    dict
        Velocity-grid shape, six extents in metres per second, population,
        ``[vx, vy, vz]`` axis order, and middle-``vy`` slice definition.

    """

    pop = str(pop)
    if velocity_mesh_shape is None:
        mesh_size = np.asarray(
            reader.get_velocity_mesh_size(pop),
            dtype=np.int64,
        )
        block_size = np.asarray(
            reader.get_velocity_block_size(pop),
            dtype=np.int64,
        )
        shape = mesh_size * block_size
    else:
        shape = np.asarray(velocity_mesh_shape, dtype=np.int64)

    extent_mps = np.asarray(
        reader.get_velocity_mesh_extent(pop),
        dtype=np.float64,
    )
    return normalize_velocity_grid_descriptor(
        {
            "shape": shape,
            "extent_mps": extent_mps,
            "population": pop,
            "axis_order": VELOCITY_AXIS_ORDER,
            "slice_axis": "vy",
            "slice_index": int(shape[1] // 2),
        }
    )


def normalize_velocity_grid_descriptor(descriptor):
    """Normalize one current ``[vx, vy, vz]`` velocity-grid descriptor.

    Loading from NPZ archives produces NumPy scalar and array wrappers, while
    reader construction produces ordinary Python values. This function makes
    those storage types consistent; it does not rescale physical velocities
    or alter the grid geometry.

    Parameters
    ----------
    descriptor : dict
        Mapping containing ``shape``, ``extent_mps``, ``population``,
        ``axis_order``, ``slice_axis``, and ``slice_index``.

    Returns
    -------
    dict
        Descriptor with integer shape, float64 extents in metres per second,
        and normalized scalar values.

    """

    normalized = normalize_velocity_grid_geometry(descriptor)
    slice_axis = str(np.asarray(descriptor["slice_axis"]).item()).strip()
    slice_index = int(np.asarray(descriptor["slice_index"]).item())

    return {
        **normalized,
        "slice_axis": slice_axis,
        "slice_index": slice_index,
    }


def normalize_velocity_grid_geometry(descriptor):
    """Normalize the complete three-dimensional velocity-grid geometry.

    Full-volume prediction and checkpoint serialization use only the mesh
    shape, physical extents, population, and axis order. Keeping these
    fields separate from the dataset's retained visualization/autoencoder
    slice metadata prevents a 2-D convention from entering raw CNN input
    preparation.

    Parameters
    ----------
    descriptor : mapping
        Velocity-grid shape, six extents in metres per second, population,
        and ``(vx, vy, vz)`` axis order.

    Returns
    -------
    dict
        Geometry with integer shape, float64 extents, normalized population,
        and axis-order values.
    """

    shape = np.asarray(descriptor["shape"], dtype=np.int64)
    extent_mps = np.asarray(descriptor["extent_mps"], dtype=np.float64)
    axis_order = tuple(
        str(value) for value in np.asarray(descriptor["axis_order"]).tolist()
    )
    population = str(np.asarray(descriptor["population"]).item()).strip()
    return {
        "shape": shape,
        "extent_mps": extent_mps,
        "population": population,
        "axis_order": np.asarray(axis_order),
    }


def create_trilinear_volume_plan(source_grid, target_grid):
    """Plan cell-centred interpolation of one complete physical VDF.

    Raw prediction constructs this plan once per source/training grid pair
    and reuses it for every selected spatial cell. Each velocity axis owns
    source index pairs, float32 weights, and a coverage mask, so the complete
    ``(vx, vy, vz)`` target volume can be resampled without rebuilding
    coordinate arrays.

    Parameters
    ----------
    source_grid : mapping
        Native VLSV velocity-grid geometry in metres per second.
    target_grid : mapping
        Training velocity-grid geometry stored in the CNN checkpoint.

    Returns
    -------
    dict
        Three-axis interpolation indices, weights, coverage masks, target
        shape, and identity-grid flag.

    Notes
    -----
    Target cell centres outside the source centre range are zero-filled.
    Interpolation is trilinear in physical phase-space density and is not a
    conservative velocity-moment remapping.
    """

    source_grid = normalize_velocity_grid_geometry(source_grid)
    target_grid = normalize_velocity_grid_geometry(target_grid)
    source_shape = source_grid["shape"]
    target_shape = target_grid["shape"]
    source_extent = source_grid["extent_mps"]
    target_extent = target_grid["extent_mps"]
    identity = (
        np.array_equal(source_shape, target_shape)
        and np.allclose(
            source_extent,
            target_extent,
            rtol=1e-12,
            atol=0.0,
        )
    )

    axis_plans = []
    target_centers = _create_velocity_cell_centers(target_grid)
    for axis_index in range(3):
        lower, upper, weight, valid = _create_axis_interpolation(
            source_min=source_extent[axis_index],
            source_max=source_extent[axis_index + 3],
            source_size=source_shape[axis_index],
            target_centers=target_centers[axis_index],
        )
        axis_plans.append(
            {
                "lower": lower,
                "upper": upper,
                "weight": weight,
                "valid": valid,
            }
        )
    coverage_fraction = float(
        np.prod(
            [
                np.mean(axis_plan["valid"], dtype=float)
                for axis_plan in axis_plans
            ]
        )
    )
    return {
        "source_grid": source_grid,
        "source_shape": source_shape.copy(),
        "target_shape": target_shape.copy(),
        "axis_plans": tuple(axis_plans),
        "identity": bool(identity),
        "coverage_fraction": coverage_fraction,
    }


def interpolate_volume(source_volume, interpolation_plan, dtype):
    """Interpolate a complete physical VDF with a cached three-axis plan.

    Interpolation is applied successively along ``vx``, ``vy``, and ``vz``.
    This separable evaluation is algebraically trilinear while avoiding a
    materialized target coordinate array with three values per voxel.

    Parameters
    ----------
    source_volume : numpy.ndarray
        Physical VDF in ``s^3 m^-6`` with shape
        ``(source_vx, source_vy, source_vz)``.
    interpolation_plan : dict
        Plan returned by :func:`create_trilinear_volume_plan`.
    dtype : numpy.dtype
        Floating-point dtype of the returned target volume.

    Returns
    -------
    numpy.ndarray
        Resampled physical VDF with target ``(vx, vy, vz)`` shape.
    """

    source_volume = np.asarray(source_volume, dtype=dtype)
    if interpolation_plan["identity"]:
        return source_volume.copy(order="C")

    x_plan, y_plan, z_plan = interpolation_plan["axis_plans"]
    x_weight = x_plan["weight"][:, None, None]
    interpolated_x = (
        (1.0 - x_weight) * source_volume[x_plan["lower"], :, :]
        + x_weight * source_volume[x_plan["upper"], :, :]
    )
    interpolated_x[~x_plan["valid"], :, :] = 0.0

    y_weight = y_plan["weight"][None, :, None]
    interpolated_y = (
        (1.0 - y_weight) * interpolated_x[:, y_plan["lower"], :]
        + y_weight * interpolated_x[:, y_plan["upper"], :]
    )
    interpolated_y[:, ~y_plan["valid"], :] = 0.0

    z_weight = z_plan["weight"][None, None, :]
    result = (
        (1.0 - z_weight) * interpolated_y[:, :, z_plan["lower"]]
        + z_weight * interpolated_y[:, :, z_plan["upper"]]
    )
    result[:, :, ~z_plan["valid"]] = 0.0
    return np.asarray(result, dtype=dtype, order="C")


def _create_velocity_cell_centers(velocity_grid):
    """Create ``vx``, ``vy``, and ``vz`` cell centres in metres per second.

    Interpolation planning uses these one-dimensional arrays to map target
    centres onto source-grid fractional indices without materializing a full
    three-dimensional coordinate volume.
    """

    shape = velocity_grid["shape"]
    extent = velocity_grid["extent_mps"]
    cell_size = (extent[3:] - extent[:3]) / shape
    return tuple(
        extent[axis_index]
        + (np.arange(shape[axis_index], dtype=np.float64) + 0.5)
        * cell_size[axis_index]
        for axis_index in range(3)
    )


def _create_axis_interpolation(
    source_min,
    source_max,
    source_size,
    target_centers,
):
    """Plan linear interpolation from source cells to target-axis centres.

    The returned lower and upper indices and float32 weights are reused across
    the orthogonal plane. A Boolean mask marks target centres outside the
    source centre range so the caller can preserve the established zero-fill
    behavior.
    """

    source_size = int(source_size)
    source_cell_size = (float(source_max) - float(source_min)) / source_size
    first_source_center = float(source_min) + 0.5 * source_cell_size
    continuous_indices = (
        np.asarray(target_centers, dtype=np.float64) - first_source_center
    ) / source_cell_size
    tolerance = 32.0 * np.finfo(np.float64).eps * max(1, source_size)
    valid = (
        (continuous_indices >= -tolerance)
        & (continuous_indices <= source_size - 1 + tolerance)
    )
    clipped_indices = np.clip(continuous_indices, 0.0, source_size - 1)
    lower = np.floor(clipped_indices).astype(np.int64)
    upper = np.minimum(lower + 1, source_size - 1)
    weights = (clipped_indices - lower).astype(np.float32)
    return lower, upper, weights, valid
