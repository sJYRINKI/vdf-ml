import numpy as np
import analysator as pt

R_EARTH = 6.371e6


def coord_re_to_m(coord_re):
    """
    Convert coordinates from Earth radii to meters.

    Parameters
    ----------
    coord_re : array-like of float
        Coordinate in Earth radii, given as ``[x, y, z]``.

    Returns
    -------
    numpy.ndarray
        Coordinate in meters.
    """
    return np.array(coord_re, dtype=float) *R_EARTH


def create_coordinate_name(coord_re):
    """
    Create a string name for the coordinate.

    Parameters
    ----------
    coord_re : array-like of float
        Coordinate in Earth radii, given as ``[x, y, z]``.

    Returns
    -------
    str
        Coordinate name.
    """
    return (
        f"x{coord_re[0]:g}_y{coord_re[1]:g}_z{coord_re[2]:g}"
        .replace(".", "p")
        .replace("-", "m")
    )


def get_cellid_with_vdf(reader, coord_re, pop="avgs"):
    """
    Find the nearest cell ID with a VDF.

    Parameters
    ----------
    reader : analysator.vlsvfile.VlsvReader
        Open VLSV file reader.
    coord_re : array-like of float
        Coordinate in Earth radii, given as ``[x, y, z]``.
    pop : str, optional
        Particle population name.

    Returns
    -------
    int
        Cell ID with a VDF
    """

    coord_m = coord_re_to_m(coord_re)
    cid = reader.get_cellid_with_vdf(coord_m, pop=pop)

    return int(cid)


def cell_has_vdf(reader, cid, pop="avgs"):
    """
    Check whether a cell contains a VDF.

    Parameters
    ----------
    reader : analysator.vlsvfile.VlsvReader
        Open VLSV file reader.
    cid : int
        Spatial cell ID.
    pop : str, optional
        Particle population name.

    Returns
    -------
    bool
        Whether the cell contains velocity-space data.
    """

    try:
        velocity_cells = reader.read_velocity_cells(int(cid), pop)
    except Exception:
        return False

    return len(velocity_cells) > 0


def get_vdf_cellid_set(reader, pop="avgs"):
    """
    Return all spatial cell IDs that contain VDF blocks.

    Parameters
    ----------
    reader : analysator.vlsvfile.VlsvReader
        Open VLSV file reader.
    pop : str, optional
        Particle population name.

    Returns
    -------
    set of int
        Cell IDs with velocity-space blocks.
    """

    try:
        cellids = reader.read(
            mesh="SpatialGrid",
            tag="CELLSWITHBLOCKS",
            name=pop,
        )
    except Exception:
        cellids = reader.read(
            mesh="SpatialGrid",
            tag="CELLSWITHBLOCKS",
        )

    return {int(cid) for cid in np.atleast_1d(cellids)}


def get_vdf_cells_with_coords_re(reader, pop="avgs"):
    """
    Return VDF cell IDs and their coordinates in Earth radii.

    Parameters
    ----------
    reader : analysator.vlsvfile.VlsvReader
        Open VLSV file reader.
    pop : str, optional
        Particle population name.

    Returns
    -------
    cellids : numpy.ndarray
        Spatial cell IDs with VDF data.
    coords_re : numpy.ndarray
        Cell center coordinates in Earth radii with shape ``(n_cells, 3)``.
    """

    cellids = np.asarray(sorted(get_vdf_cellid_set(reader, pop=pop)), dtype=int)
    if len(cellids) == 0:
        return cellids, np.empty((0, 3), dtype=float)

    try:
        coords = reader.get_cell_coordinates(cellids)
    except Exception:
        coords = [reader.get_cell_coordinates(int(cid)) for cid in cellids]

    coords_re = np.asarray(coords, dtype=float) / R_EARTH

    return cellids, coords_re


def get_nearest_vdf_cellid(coord_re, vdf_cellids, vdf_coords_re):
    """
    Return the VDF cell ID nearest to a coordinate.

    Parameters
    ----------
    coord_re : array-like of float
        Coordinate in Earth radii, given as ``[x, y, z]``.
    vdf_cellids : numpy.ndarray
        Spatial cell IDs with VDF data.
    vdf_coords_re : numpy.ndarray
        VDF cell coordinates in Earth radii with shape ``(n_cells, 3)``.

    Returns
    -------
    int
        Nearest VDF cell ID.
    """

    if len(vdf_cellids) == 0:
        raise ValueError("No velocity distributions found")

    coord_re = np.asarray(coord_re, dtype=float)
    distances_squared = np.sum((vdf_coords_re - coord_re) ** 2, axis=1)
    nearest_index = int(np.argmin(distances_squared))

    return int(vdf_cellids[nearest_index])


def get_spatial_index_range(reader, axis_name, axis_index, min_value, max_value):
    """
    Convert spatial coordinate bounds to clipped fsgrid index bounds.

    Parameters
    ----------
    reader : analysator.vlsvfile.VlsvReader
        Open VLSV file reader.
    axis_name : str
        Spatial axis name, one of ``x``, ``y``, or ``z``.
    axis_index : int
        Axis index in the spatial mesh size.
    min_value : float
        Lower coordinate bound in meters.
    max_value : float
        Upper coordinate bound in meters.

    Returns
    -------
    tuple of int
        Inclusive lower and upper fsgrid index bounds.
    """

    mesh_size = np.asarray(reader.get_spatial_mesh_size(), dtype=int)
    n_cells = int(mesh_size[axis_index])
    axis_min = float(reader.read_parameter(f"{axis_name}min"))
    axis_max = float(reader.read_parameter(f"{axis_name}max"))
    cell_size = (axis_max - axis_min) / n_cells

    lower_index = int(np.floor((min_value - axis_min) / cell_size))
    upper_index = int(np.floor((max_value - axis_min) / cell_size))

    lower_index = max(0, min(n_cells - 1, lower_index))
    upper_index = max(0, min(n_cells - 1, upper_index))

    return min(lower_index, upper_index), max(lower_index, upper_index)


def get_vdf_cellids_in_box(
    reader,
    coord_re,
    box_config,
    pop="avgs",
    cell_has_vdf_func=None,
    vdf_cellids=None,
    vdf_coords_re=None,
):
    """
    Find all VDF cell IDs inside a spatial box around a coordinate.

    Parameters
    ----------
    reader : analysator.vlsvfile.VlsvReader
        Open VLSV file reader.
    coord_re : array-like of float
        Box center in Earth radii, given as ``[x, y, z]``.
    box_config : dict
        Box half-widths in Earth radii.
    pop : str, optional
        Particle population name.
    cell_has_vdf_func : callable, optional
        Function taking a cell ID and returning whether it has VDF data. Kept
        for compatibility; precomputed VDF cell IDs already satisfy this.
    vdf_cellids : numpy.ndarray, optional
        Spatial cell IDs with VDF data.
    vdf_coords_re : numpy.ndarray, optional
        VDF cell coordinates in Earth radii with shape ``(n_cells, 3)``.

    Returns
    -------
    dict
        Mapping from box position name to VDF cell ID.
    """

    if vdf_cellids is None or vdf_coords_re is None:
        vdf_cellids, vdf_coords_re = get_vdf_cells_with_coords_re(
            reader=reader,
            pop=pop,
        )

    if len(vdf_cellids) == 0:
        return {}

    center_re = np.asarray(coord_re, dtype=float)
    half_widths_re = np.array(
        [
            float(box_config["x_half_width_re"]),
            float(box_config["y_half_width_re"]),
            float(box_config["z_half_width_re"]),
        ],
        dtype=float,
    )

    lower_bounds = center_re - half_widths_re
    upper_bounds = center_re + half_widths_re
    active_axes = half_widths_re > 0

    if np.any(active_axes):
        selected = np.all(
            (vdf_coords_re[:, active_axes] >= lower_bounds[active_axes])
            & (vdf_coords_re[:, active_axes] <= upper_bounds[active_axes]),
            axis=1,
        )
    else:
        selected = np.ones(len(vdf_cellids), dtype=bool)

    selected_cellids = vdf_cellids[selected]
    if cell_has_vdf_func is not None:
        selected_cellids = np.asarray(
            [cid for cid in selected_cellids if cell_has_vdf_func(int(cid))],
            dtype=int,
        )

    return {
        f"box_{index:04d}": int(cid)
        for index, cid in enumerate(selected_cellids)
    }


def get_min_value_from_file(file_location, cid):
    """
    Read the VDF minimum value.

    Parameters
    ----------
    file_location : str
        Path to VLSV file.
    cid : int
        Spatial cell ID.

    Returns
    -------
    float
        VDF minimum value threshold.
    """
    reader = pt.vlsvfile.VlsvReader(str(file_location))
    return float(reader.read_variable("MinValue", int(cid)))


def get_velocity_cell_size_from_extent(extent, vdf_shape, axis="vy"):
    """
    Compute velocity cell size from velocity mesh and VDF shape.

    Parameters
    ----------
    extent : array-like of float
        Velocity mesh extent.
    vdf_shape : tuple of int
        Shape of the VDF array.
    axis : {"vx", "vy", "vz"}, optional
        Velocity axis.

    Returns
    -------
    float
        Velocity cell size.
    """

    extent = np.asarray(extent, dtype=float)

    axis_map = {
        "vx": 0,
        "vy": 1,
        "vz": 2,
    }

    axis_index = axis_map[axis]

    vmin = extent[axis_index]
    vmax = extent[axis_index + 3]

    return float((vmax - vmin) / vdf_shape[axis_index])


def get_vdf_plot_parameters_from_file(file_location, cid, vdf_shape, pop="avgs"):
    """
    Read sample specific parameters from a VLSV file.

    Parameters
    ----------
    file_location : str
        Path to VLSV file.
    cid : int
        Spatial cell ID..
    vdf_shape : tuple of int
        Shape of the VDF array.
    pop : str, optional
        Particle population name.

    Returns
    -------
    extent : numpy.ndarray
        Velocity mesh extent.
    dc : float
        Velocity cell size.
    threshold : float
        VDF sparsity threshold.
    """

    reader = pt.vlsvfile.VlsvReader(str(file_location))

    extent = np.asarray(
        reader.get_velocity_mesh_extent(pop=pop),
        dtype=float
    )

    dv = get_velocity_cell_size_from_extent(
        extent=extent,
        vdf_shape=vdf_shape,
    )

    threshold = float(reader.read_variable("MinValue", int(cid)))

    return extent, dv, threshold


def create_xz_slice(vdf):
    """
    Extract the middle xz slice from 3D VDF.

    Parameters
    ----------
    vdf:numpy.ndarray
        VDF array.

    Returns
    -------
    numpy.ndarray
        xz slice.
    """

    mid_y = vdf.shape[1] // 2
    return vdf[:, mid_y, :]
