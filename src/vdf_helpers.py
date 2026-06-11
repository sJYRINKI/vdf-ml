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

    threshold = get_min_value_from_file(
        file_location=file_location,
        cid=cid
    )

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