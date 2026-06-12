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

def get_vdf_ranks_in_direction(
        reader,
        center_indices,
        direction,
        ranks,
        max_index_steps,
        pop="avgs",
):
    """
    Find ranked VDF cell IDs by walking in one spatial-index direction.

    Parameters
    ----------
    reader : analysator.vlsvfile.VlsvReader
        Open VLSV file reader.
    center_indices : array-like of int
        Spatial index of the center cell.
    direction : tuple of int
        Spatial-index direction ``(di, dj, dk)``.
    ranks : iterable of int
        VDF ranks to return from this direction.
    max_index_steps : int
        Maximum number of spatial index steps to scan.
    pop : str, optional
        Particle population name.

    Returns
    -------
    dict
        Mapping from requested rank to VDF cell ID.
    """

    ranks = sorted(int(rank) for rank in ranks)

    if not ranks:
        return {}

    max_rank = max(ranks)
    i, j, k = map(int, center_indices)
    di, dj, dk = direction
    found_cellids = []

    for step in range(1, int(max_index_steps) + 1):
        try:
            cid = int(
                reader.get_cellid_at_fsgrid_index(
                    i + step * di,
                    j + step * dj,
                    k + step * dk,
                )
            )
        except Exception:
            break

        if cid <= 0:
            break
        if not cell_has_vdf(reader, cid, pop=pop):
            continue

        found_cellids.append(cid)

        if len(found_cellids) >= max_rank:
            break

    return {
        rank: found_cellids[rank - 1]
        for rank in ranks
        if len(found_cellids) >= rank
    }

def get_neighbor_vdf_cellids(reader, coord_re, neighborhood_config=None, pop="avgs"):
    """
    Find VDF cell IDs around the nearest VDF cell.

    The input coordinate is first mapped to the nearest cell with a VDF. Each
    configured direction is then scanned until the requested VDF ranks are
    found or the maximum scan distance is reached.

    Parameters
    ----------
    reader : analysator.vlsvfile.VlsvReader
        Open VLSV file reader.
    coord_re : array-like of float
        Coordinate in Earth radii, given as ``[x, y, z]``.
    neighborhood_config : dict, optional
        Neighborhood configuration.
    pop : str, optional
        Particle population name.

    Returns
    -------
    dict
        Mapping from neighborhood position name to VDF cell ID.
    """

    center_cid = get_cellid_with_vdf(
        reader=reader,
        coord_re=coord_re,
        pop=pop,
    )

    center_indices = np.asarray(
        reader.get_cell_indices(center_cid)
    ).squeeze()

    neighborhood_config = neighborhood_config or {}

    if not neighborhood_config.get("enabled", False):
        return {"center": center_cid}

    max_index_steps = int(neighborhood_config.get("max_index_steps", 200))
    x_ranks = neighborhood_config.get("x_ranks", [1, 2])
    z_ranks = neighborhood_config.get("z_ranks", [1])

    directions = {
        "left": ((-1, 0, 0), x_ranks),
        "right": ((1, 0, 0), x_ranks),
        "bottom": ((0, 0, -1), z_ranks),
        "top": ((0, 0, 1), z_ranks),
    }

    cellids = {"center": center_cid}

    for direction_name, (direction, ranks) in directions.items():
        ranked_cellids = get_vdf_ranks_in_direction(
            reader=reader,
            center_indices=center_indices,
            direction=direction,
            ranks=ranks,
            max_index_steps=max_index_steps,
            pop=pop,
        )

        for rank, cid in ranked_cellids.items():
            cellids[f"{direction_name}_{rank}"] = cid

    return cellids

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