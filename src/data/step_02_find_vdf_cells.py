"""Stage 2: find VDF-carrying cells and spatial regions.

This stage follows opening a timestep VLSV reader and precedes dataset
sample planning. It converts configured Earth-radius regions to masks,
discovers spatial cells with velocity blocks, and maps coordinates to the
nearest VDF cell.

Inputs are one open reader plus coordinates in Earth radii. Outputs are
stable cell-ID arrays, coordinate arrays, and spatial selections used by
the planning stage.
"""

import numpy as np

from src.data.dense_vdf import resolve_velocity_population

R_EARTH = 6.371e6


def coord_re_to_m(coord_re):
    """
    Convert coordinates from Earth radii to meters.

    VLSV reader methods accept SI spatial coordinates, while extraction
    configuration and saved metadata use Earth radii. This conversion is the
    boundary between those conventions and preserves the input ``[x, y, z]``
    axis order.

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

    Plot and sample identifiers use this deterministic encoding to place an
    Earth-radius coordinate in a filename-safe token. Decimal points become
    ``p`` and minus signs become ``m`` without changing the coordinate itself.

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


def iter_enabled_regions_re(points_config, names_key, default_region_name="tail"):
    """
    Iterate configured spatial regions in Earth radii.

    Point and background selectors share this iterator so region order has one
    definition. A configuration containing one direct ``region_re`` mapping is
    exposed under ``default_region_name``; otherwise ``names_key`` controls
    the ordered subset of ``regions_re``.

    Parameters
    ----------
    points_config : dict
        Point sampling configuration.
    names_key : str
        Key containing the ordered region names to use.
    default_region_name : str, optional
        Region name assigned when configuration provides one direct
        ``region_re`` mapping.

    Yields
    ------
    region_name : str
        Region name.
    region_re : dict
        Region bounds in Earth radii.
    """

    # Retrieve the points configuration or initialize it as an empty dictionary if not provided
    points_config = points_config or {}
    regions_re = points_config.get("regions_re")

    # If no regions are configured, check for a single region configuration and yield it with the default region name
    if regions_re is None:
        region_re = points_config.get("region_re")
        if region_re is not None:
            yield default_region_name, region_re
        return

    # Retrieve the list of region names
    region_names = points_config.get(names_key)
    # If no specific region names are provided, use all configured regions
    if region_names is None:
        region_names = list(regions_re)
    # If a single region name is provided as a string, convert it to a list for iteration
    elif isinstance(region_names, str):
        region_names = [region_names]

    # Iterate over the region names and yield the corresponding bounds.
    for region_name in region_names:
        yield str(region_name), regions_re[region_name]


def find_matching_region_name_re(coord_re, points_config, names_key):
    """
    Return the first configured region containing a coordinate.

    Topology detection uses the first match to retain configured region
    priority when bounds overlap. With no region bounds configured, the
    coordinate belongs to the synthetic ``"all"`` region.

    Parameters
    ----------
    coord_re : array-like of float
        Coordinate in Earth radii, given as ``[x, y, z]``.
    points_config : dict
        Point sampling configuration.
    names_key : str
        Key containing the ordered region names to use.

    Returns
    -------
    str or None
        Matching region name, or ``None`` when the coordinate is outside all
        enabled regions.
    """

    # Retrieve the points configuration or initialize it as an empty dictionary if not provided
    points_config = points_config or {}
    if (
            points_config.get("regions_re") is None
            and points_config.get("region_re") is None
    ):
        return "all"

    # Iterate over the enabled regions and check if the coordinate is inside any of them
    for region_name, region_re in iter_enabled_regions_re(
            points_config=points_config,
            names_key=names_key,
    ):
        # Check if the coordinate is inside the current region using the helper function
        if is_coord_in_region_re(coord_re, region_re):
            return region_name

    return None


def is_coord_in_region_re(coord_re, region_re):
    """
    Check whether a coordinate is inside a region.

    Spatial selection calls this scalar form for one ``[x, y, z]`` location.
    Bounds are inclusive and axes without a configured lower or upper limit do
    not restrict the coordinate.

    Parameters
    ----------
    coord_re : array-like of float
        Coordinate in Earth radii, given as ``[x, y, z]``.
    region_re : dict
        Region bounds in Earth radii.

    Returns
    -------
    bool
        Whether the coordinate is inside the configured bounds.
    """

    # Convert the coordinate to a NumPy array
    coord_re = np.asarray(coord_re, dtype=float)

    # Iterate over the three spatial axes (x, y, z) and check if the coordinate is within the bounds of the region for each axis
    for axis_index, axis_name in enumerate(("x", "y", "z")):
        lower_re, upper_re = get_region_axis_bounds_re(region_re, axis_name)

        if lower_re is not None and coord_re[axis_index] < lower_re:
            return False

        if upper_re is not None and coord_re[axis_index] > upper_re:
            return False

    return True


def create_region_mask_re(coords_re, region_re):
    """
    Create a boolean mask for coordinates inside a region.

    Background and topology workflows use this vectorized counterpart to
    select aligned cell-coordinate rows without changing their original
    order. Region bounds are inclusive on every configured axis.

    Parameters
    ----------
    coords_re : numpy.ndarray
        Coordinates in Earth radii with shape ``(n_cells, 3)``.
    region_re : dict
        Region bounds in Earth radii.

    Returns
    -------
    numpy.ndarray
        Boolean mask selecting coordinates inside the configured region.
    """

    coords_re = np.asarray(coords_re, dtype=float)
    selected = np.ones(coords_re.shape[0], dtype=bool)

    for axis_index, axis_name in enumerate(("x", "y", "z")):
        lower_re, upper_re = get_region_axis_bounds_re(region_re, axis_name)

        if lower_re is not None:
            selected &= coords_re[:, axis_index] >= lower_re

        if upper_re is not None:
            selected &= coords_re[:, axis_index] <= upper_re

    return selected


def get_region_axis_bounds_re(region_re, axis_name):
    """
    Return optional sorted bounds for one region axis.

    This resolver centralizes the accepted interval, absolute-maximum, and
    separate lower/upper spellings so scalar and vector region selection use
    identical Earth-radius bounds.

    Parameters
    ----------
    region_re : dict
        Region bounds in Earth radii.
    axis_name : {"x", "y", "z"}
        Coordinate axis name.

    Returns
    -------
    tuple
        Pair of optional lower and upper bounds in Earth radii.
    """

    between = region_re.get(f"{axis_name}_between")
    if between is not None:
        lower_re, upper_re = between
        return min(lower_re, upper_re), max(lower_re, upper_re)

    abs_max_re = region_re.get(f"{axis_name}_abs_max")
    if abs_max_re is not None:
        abs_max_re = float(abs_max_re)
        return -abs_max_re, abs_max_re

    lower_re = region_re.get(f"{axis_name}_min")
    upper_re = region_re.get(f"{axis_name}_max")
    if lower_re is not None:
        lower_re = float(lower_re)
    if upper_re is not None:
        upper_re = float(upper_re)

    if lower_re is not None and upper_re is not None:
        return min(lower_re, upper_re), max(lower_re, upper_re)

    return lower_re, upper_re


def get_cellid_with_vdf(reader, coord_re, pop=None):
    """
    Find the nearest cell ID with a VDF.

    Static coordinates and detected critical points call this mapping before
    sample planning. The Earth-radius coordinate is converted to metres and
    Analysator selects the nearest spatial cell that stores velocity blocks.

    Parameters
    ----------
    reader : analysator.vlsvfile.VlsvReader
        Open VLSV file reader.
    coord_re : array-like of float
        Coordinate in Earth radii, given as ``[x, y, z]``.
    pop : str, optional
        Explicit particle population override. If omitted, ``avgs`` is used
        when available and ``proton`` otherwise.

    Returns
    -------
    int
        Cell ID with a VDF
    """

    pop = resolve_velocity_population(reader=reader, pop=pop)
    coord_m = coord_re_to_m(coord_re)
    cid = reader.get_cellid_with_vdf(coord_m, pop=pop)

    return int(cid)


def cell_has_vdf(reader, cid, pop=None):
    """
    Check whether a cell contains a VDF.

    Optional spatial selectors use this focused reader query when they have
    not already received the file-wide set of VDF-carrying cell IDs.

    Parameters
    ----------
    reader : analysator.vlsvfile.VlsvReader
        Open VLSV file reader.
    cid : int
        Spatial cell ID.
    pop : str, optional
        Explicit particle population override. If omitted, ``avgs`` is used
        when available and ``proton`` otherwise.

    Returns
    -------
    bool
        Whether the cell contains velocity-space data.
    """

    pop = resolve_velocity_population(reader=reader, pop=pop)
    velocity_cells = reader.read_velocity_cells(int(cid), pop)
    return len(velocity_cells) > 0


def get_vdf_cellid_set(reader, pop=None):
    """
    Return all spatial cell IDs that contain VDF blocks.

    Stage 2 reads the file-level ``CELLSWITHBLOCKS`` record once so subsequent
    coordinate and region selection can operate on arrays rather than issuing
    one sparse-VDF read per spatial cell.

    Parameters
    ----------
    reader : analysator.vlsvfile.VlsvReader
        Open VLSV file reader.
    pop : str, optional
        Explicit particle population override. If omitted, ``avgs`` is used
        when available and ``proton`` otherwise.

    Returns
    -------
    set of int
        Cell IDs with velocity-space blocks.
    """

    # Resolve the particle population to use for retrieving VDF data
    pop = resolve_velocity_population(reader=reader, pop=pop)

    # Retrieve cell IDs containing VDF blocks for the selected population.
    cellids = reader.read(
        mesh="SpatialGrid",
        tag="CELLSWITHBLOCKS",
        name=pop,
    )

    return {int(cid) for cid in np.atleast_1d(cellids)}


def get_vdf_cells_with_coords_re(reader, pop=None):
    """
    Return VDF cell IDs and their coordinates in Earth radii.

    Sample planning uses these aligned arrays as its stable spatial index.
    Sorting IDs before the single coordinate read makes background and
    point-selection order deterministic.

    Parameters
    ----------
    reader : analysator.vlsvfile.VlsvReader
        Open VLSV file reader.
    pop : str, optional
        Explicit particle population override. If omitted, ``avgs`` is used
        when available and ``proton`` otherwise.

    Returns
    -------
    cellids : numpy.ndarray
        Spatial cell IDs with VDF data.
    coords_re : numpy.ndarray
        Cell center coordinates in Earth radii with shape ``(n_cells, 3)``.
    """

    # Resolve the particle population to use for retrieving VDF data
    pop = resolve_velocity_population(reader=reader, pop=pop)

    # Retrieve the set of cell IDs that contain VDF blocks for the specified population
    cellids = np.asarray(sorted(get_vdf_cellid_set(reader, pop=pop)), dtype=int)
    if len(cellids) == 0:
        return cellids, np.empty((0, 3), dtype=float)

    # Retrieve the coordinates of the VDF cells in Earth radii.
    coords = reader.get_cell_coordinates(cellids)

    # Convert the coordinates from meters to Earth radii
    coords_re = np.asarray(coords, dtype=float) / R_EARTH

    return cellids, coords_re


def get_nearest_vdf_cellid(coord_re, vdf_cellids, vdf_coords_re):
    """
    Return the VDF cell ID nearest to a coordinate.

    Static-label planning applies this Euclidean Earth-radius lookup to the
    precomputed file-wide VDF-cell arrays, avoiding repeated Analysator
    searches while preserving the caller's cell-ID ordering for ties.

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

    coord_re = np.asarray(coord_re, dtype=float)
    distances_squared = np.sum((vdf_coords_re - coord_re) ** 2, axis=1)
    nearest_index = int(np.argmin(distances_squared))

    return int(vdf_cellids[nearest_index])


def get_spatial_index_range(reader, axis_name, axis_index, min_value, max_value):
    """
    Convert spatial coordinate bounds to clipped fsgrid index bounds.

    Physics routines use this conversion when a physical box must address the
    regular field-solver grid. The returned pair is inclusive and clipped to
    the available cells on the requested spatial axis.

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
    pop=None,
    cell_has_vdf_func=None,
    vdf_cellids=None,
    vdf_coords_re=None,
):
    """
    Find all VDF cell IDs inside a spatial box around a coordinate.

    Manual sample selection applies the configured Earth-radius half-widths to
    aligned VDF-cell centres. Zero half-width axes are ignored, and returned
    position names retain the selected input order for deterministic metadata.

    Parameters
    ----------
    reader : analysator.vlsvfile.VlsvReader
        Open VLSV file reader.
    coord_re : array-like of float
        Box center in Earth radii, given as ``[x, y, z]``.
    box_config : dict
        Box half-widths in Earth radii.
    pop : str, optional
        Explicit particle population override. If omitted, ``avgs`` is used
        when available and ``proton`` otherwise.
    cell_has_vdf_func : callable, optional
        Additional predicate applied after the geometric selection.
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
