import numpy as np
import analysator as pt


VELOCITY_AXIS_ORDER = ("vx", "vy", "vz")
EQUIVALENT_VELOCITY_POPULATIONS = ("avgs", "proton")


class VdfExtractor:
    """
    Extract dense VDF arrays or interpolated target-grid xz slices.

    Parameters
    ----------
    reader : analysator.vlsvfile.VlsvReader
        Open VLSV file reader.
    pop : str, optional
        Explicit particle population override. If omitted, ``avgs`` is used
        when available and ``proton`` otherwise.
    dtype : data-type, optional
        Output data type for dense VDF arrays.
    """

    def __init__(self, reader, pop=None, dtype=np.float32):
        self.reader = reader
        self.dtype = dtype

        self.velocity_grid = create_velocity_grid_descriptor(
            reader=reader,
            pop=pop,
        )
        self.pop = self.velocity_grid["population"]
        self.vdf_shape = tuple(
            int(value) for value in self.velocity_grid["shape"]
        )
        self.n_velocity_cells = int(np.prod(self.vdf_shape))
        self.sorted_velocity_indices = None

    def extract(self, cid, box=-1):
        """
        Extract a 3D VDF from one Vlasiator cell.

        Parameters
        ----------
        cid : int
            Spatial cell ID from which the VDF is extracted.
        box : int, optional
            If positive, crop the VDF around its maximum value using this value
            as the half-width of the crop in index space. If ``-1``, return the
            full VDF.

        Returns
        -------
        numpy.ndarray
            Dense 3D VDF array with dtype ``float32`` by default. The returned
            axis order is ``[vx, vy, vz]``.
        """

        if int(cid) <= 0:
            raise ValueError("cid must be positive")

        velocity_cells = self.reader.read_velocity_cells(int(cid), self.pop)
        dist = np.zeros(self.n_velocity_cells, dtype=self.dtype)
        dist[list(velocity_cells.keys())] = list(velocity_cells.values())

        if self.sorted_velocity_indices is None:
            self.sorted_velocity_indices = create_sorted_velocity_indices(
                reader=self.reader,
                n_velocity_cells=self.n_velocity_cells,
                pop=self.pop,
            )

        nx, ny, nz = self.vdf_shape
        vdf = dist[self.sorted_velocity_indices].reshape((nz, ny, nx))
        vdf = np.transpose(vdf, (2, 1, 0))
        max_indices = np.unravel_index(np.argmax(vdf), vdf.shape)
        box = int(box)

        if box > 0:
            i, j, k = max_indices
            data = vdf[
                (i - box):(i + box),
                (j - box):(j + box),
                (k - box):(k + box),
            ]
        else:
            data = vdf

        return np.asarray(data, dtype=self.dtype)

    def extract_interpolated_xz(self, cid, interpolation_plan):
        """
        Extract and interpolate one VDF onto a target xz plane.

        Only the two source ``vy`` planes required by trilinear interpolation
        are materialized. Missing sparse velocity cells and target centers
        outside the source center range are treated as zero. Interpolation is
        applied to raw VDF values and is not conservative for velocity moments.

        Parameters
        ----------
        cid : int
            Spatial cell ID from which the VDF is extracted.
        interpolation_plan : dict
            Cached plan returned by ``create_trilinear_xz_plan``.

        Returns
        -------
        numpy.ndarray
            Raw target-grid xz slice with axis order ``[vx, vz]``.
        """

        if int(cid) <= 0:
            raise ValueError("cid must be positive")

        plan_source_grid = interpolation_plan.get("source_grid")
        if plan_source_grid is None or not velocity_grid_descriptors_match(
            plan_source_grid,
            self.velocity_grid,
        ):
            raise ValueError(
                "Interpolation source grid does not match the VDF extractor"
            )

        source_y_indices = tuple(
            int(value) for value in interpolation_plan["source_y_indices"]
        )
        source_planes = np.zeros(
            (2, self.vdf_shape[0], self.vdf_shape[2]),
            dtype=self.dtype,
        )
        velocity_cells = self.reader.read_velocity_cells(int(cid), self.pop)
        if not velocity_cells:
            return _interpolate_xz_planes(
                source_planes=source_planes,
                interpolation_plan=interpolation_plan,
                dtype=self.dtype,
            )

        velocity_cell_ids = np.fromiter(
            velocity_cells.keys(),
            dtype=np.int64,
            count=len(velocity_cells),
        )
        velocity_values = np.fromiter(
            velocity_cells.values(),
            dtype=self.dtype,
            count=len(velocity_cells),
        )
        if not np.all(np.isfinite(velocity_values)):
            raise ValueError(f"VDF cell {int(cid)} contains non-finite values")
        velocity_values = np.maximum(velocity_values, 0.0)

        velocity_coords = self.reader.get_velocity_cell_coordinates(
            velocity_cell_ids,
            self.pop,
        )
        velocity_indices = self.reader.get_velocity_cell_indices(
            velocity_coords,
            self.pop,
        )
        velocity_indices = np.asarray(velocity_indices, dtype=np.int64)
        shape = np.asarray(self.vdf_shape, dtype=np.int64)
        if np.any(velocity_indices < 0) or np.any(velocity_indices >= shape):
            raise ValueError(
                f"VDF cell {int(cid)} contains out-of-range velocity indices"
            )

        for plane_index, source_y_index in enumerate(source_y_indices):
            plane_mask = velocity_indices[:, 1] == source_y_index
            plane_indices = velocity_indices[plane_mask]
            source_planes[
                plane_index,
                plane_indices[:, 0],
                plane_indices[:, 2],
            ] = velocity_values[plane_mask]

        return _interpolate_xz_planes(
            source_planes=source_planes,
            interpolation_plan=interpolation_plan,
            dtype=self.dtype,
        )


def resolve_velocity_population(reader, pop=None):
    """
    Resolve the equivalent proton VDF population in a VLSV file.

    Parameters
    ----------
    reader : analysator.vlsvfile.VlsvReader
        Open VLSV file reader.
    pop : str, optional
        Explicit population override. ``None`` and ``"auto"`` select
        ``avgs`` when available and ``proton`` otherwise.

    Returns
    -------
    str
        Population name available through the reader.
    """

    requested_population = None if pop is None else str(pop).strip()
    use_automatic_selection = (
        not requested_population
        or requested_population.lower() == "auto"
    )
    if use_automatic_selection:
        candidates = EQUIVALENT_VELOCITY_POPULATIONS
    else:
        candidates = (requested_population,)

    active_populations = tuple(
        str(population)
        for population in getattr(reader, "active_populations", ())
    )
    active_by_lower_name = {
        population.lower(): population
        for population in active_populations
    }
    for candidate in candidates:
        if hasattr(reader, "check_population"):
            if reader.check_population(candidate):
                return candidate
        elif candidate.lower() in active_by_lower_name:
            return active_by_lower_name[candidate.lower()]

    if not hasattr(reader, "check_population") and not active_populations:
        if use_automatic_selection:
            raise ValueError(
                "Cannot determine whether the VLSV file contains an avgs or "
                "proton population"
            )
        return requested_population

    if use_automatic_selection:
        raise ValueError(
            "Neither 'avgs' nor 'proton' is available in the VLSV file. "
            f"Available populations: {list(active_populations)}"
        )
    raise ValueError(
        f"Velocity population {requested_population!r} is unavailable. "
        f"Available populations: {list(active_populations)}"
    )


def create_velocity_grid_descriptor(reader, pop=None):
    """
    Create a velocity-grid descriptor from an open VLSV reader.

    Parameters
    ----------
    reader : analysator.vlsvfile.VlsvReader
        Open VLSV file reader.
    pop : str, optional
        Explicit particle population override. If omitted, ``avgs`` is used
        when available and ``proton`` otherwise.

    Returns
    -------
    dict
        Velocity-grid shape, extent in m/s, population, axis order, and
        middle-``vy`` slice definition.
    """

    pop = resolve_velocity_population(reader=reader, pop=pop)

    mesh_size = np.asarray(
        reader.get_velocity_mesh_size(pop),
        dtype=np.int64,
    )
    block_size = np.asarray(
        reader.get_velocity_block_size(pop),
        dtype=np.int64,
    )
    if hasattr(reader, "get_WID"):
        scalar_block_size = int(reader.get_WID())
        if not np.all(block_size == scalar_block_size):
            raise ValueError(
                "Anisotropic velocity blocks are unsupported by the "
                "Analysator velocity-cell indexing API"
            )
    shape = mesh_size * block_size
    extent_mps = np.asarray(
        reader.get_velocity_mesh_extent(pop),
        dtype=np.float64,
    )
    descriptor = normalize_velocity_grid_descriptor(
        {
            "shape": shape,
            "extent_mps": extent_mps,
            "population": pop,
            "axis_order": VELOCITY_AXIS_ORDER,
            "slice_axis": "vy",
            "slice_index": int(shape[1] // 2),
        }
    )

    reported_cell_size = np.asarray(
        reader.get_velocity_mesh_dv(pop),
        dtype=np.float64,
    )
    calculated_cell_size = (
        descriptor["extent_mps"][3:] - descriptor["extent_mps"][:3]
    ) / descriptor["shape"]
    if not np.allclose(
        reported_cell_size,
        calculated_cell_size,
        rtol=1e-12,
        atol=0.0,
    ):
        raise ValueError(
            "Velocity mesh extent, shape, and cell sizes are inconsistent"
        )

    return descriptor


def normalize_velocity_grid_descriptor(descriptor):
    """
    Validate and normalize a velocity-grid descriptor.

    Parameters
    ----------
    descriptor : dict
        Velocity-grid descriptor values.

    Returns
    -------
    dict
        Normalized descriptor with NumPy shape and extent arrays.
    """

    required_keys = {
        "shape",
        "extent_mps",
        "population",
        "axis_order",
        "slice_axis",
        "slice_index",
    }
    missing_keys = required_keys - set(descriptor)
    if missing_keys:
        raise ValueError(
            f"Velocity-grid descriptor is missing keys: {sorted(missing_keys)}"
        )

    shape = np.asarray(descriptor["shape"], dtype=np.int64)
    extent_mps = np.asarray(descriptor["extent_mps"], dtype=np.float64)
    axis_order = tuple(
        str(value) for value in np.asarray(descriptor["axis_order"]).tolist()
    )
    population = str(np.asarray(descriptor["population"]).item()).strip()
    slice_axis = str(np.asarray(descriptor["slice_axis"]).item()).strip()
    slice_index = int(np.asarray(descriptor["slice_index"]).item())

    if shape.shape != (3,) or np.any(shape <= 0):
        raise ValueError("Velocity-grid shape must contain three positive values")
    if extent_mps.shape != (6,) or not np.all(np.isfinite(extent_mps)):
        raise ValueError(
            "Velocity-grid extent must contain six finite values in m/s"
        )
    if np.any(extent_mps[3:] <= extent_mps[:3]):
        raise ValueError("Velocity-grid maximum extents must exceed minimums")
    if axis_order != VELOCITY_AXIS_ORDER:
        raise ValueError(
            f"Velocity-grid axis order must be {VELOCITY_AXIS_ORDER}"
        )
    if not population:
        raise ValueError("Velocity-grid population must not be empty")
    if slice_axis != "vy":
        raise ValueError("Velocity-grid slice axis must be 'vy'")
    if slice_index < 0 or slice_index >= int(shape[1]):
        raise ValueError("Velocity-grid slice index is outside the vy axis")

    return {
        "shape": shape,
        "extent_mps": extent_mps,
        "population": population,
        "axis_order": np.asarray(axis_order),
        "slice_axis": slice_axis,
        "slice_index": slice_index,
    }


def velocity_grid_descriptors_match(first, second, check_population=True):
    """
    Compare two normalized velocity-grid descriptors.

    Parameters
    ----------
    first : dict
        First velocity-grid descriptor.
    second : dict
        Second velocity-grid descriptor.
    check_population : bool, optional
        Whether population names must match.

    Returns
    -------
    bool
        Whether the descriptors define the same velocity grid.
    """

    first = normalize_velocity_grid_descriptor(first)
    second = normalize_velocity_grid_descriptor(second)
    matches = (
        np.array_equal(first["shape"], second["shape"])
        and np.allclose(
            first["extent_mps"],
            second["extent_mps"],
            rtol=1e-12,
            atol=0.0,
        )
        and np.array_equal(first["axis_order"], second["axis_order"])
        and first["slice_axis"] == second["slice_axis"]
        and first["slice_index"] == second["slice_index"]
    )
    if check_population:
        matches = matches and first["population"] == second["population"]
    return bool(matches)


def create_trilinear_xz_plan(source_grid, target_grid):
    """
    Create a reusable trilinear plan for one target xz slice.

    The plan samples cell-center values and marks target centers outside the
    source center range for zero filling.

    Parameters
    ----------
    source_grid : dict
        Native VLSV velocity-grid descriptor.
    target_grid : dict
        Training velocity-grid descriptor.

    Returns
    -------
    dict
        Cached source indices, weights, masks, and target-grid information.
    """

    source_grid = normalize_velocity_grid_descriptor(source_grid)
    target_grid = normalize_velocity_grid_descriptor(target_grid)
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

    target_centers = _create_velocity_cell_centers(target_grid)
    target_y = target_centers[1][target_grid["slice_index"]]

    if identity:
        x_lower = np.arange(target_shape[0], dtype=np.int64)
        x_upper = x_lower.copy()
        x_weight = np.zeros(target_shape[0], dtype=np.float32)
        x_valid = np.ones(target_shape[0], dtype=bool)
        z_lower = np.arange(target_shape[2], dtype=np.int64)
        z_upper = z_lower.copy()
        z_weight = np.zeros(target_shape[2], dtype=np.float32)
        z_valid = np.ones(target_shape[2], dtype=bool)
        source_y_indices = (
            int(target_grid["slice_index"]),
            int(target_grid["slice_index"]),
        )
        y_weight = 0.0
        y_valid = True
    else:
        x_lower, x_upper, x_weight, x_valid = _create_axis_interpolation(
            source_min=source_extent[0],
            source_max=source_extent[3],
            source_size=source_shape[0],
            target_centers=target_centers[0],
        )
        y_lower, y_upper, y_weights, y_valid_values = (
            _create_axis_interpolation(
                source_min=source_extent[1],
                source_max=source_extent[4],
                source_size=source_shape[1],
                target_centers=np.asarray([target_y]),
            )
        )
        z_lower, z_upper, z_weight, z_valid = _create_axis_interpolation(
            source_min=source_extent[2],
            source_max=source_extent[5],
            source_size=source_shape[2],
            target_centers=target_centers[2],
        )
        source_y_indices = (int(y_lower[0]), int(y_upper[0]))
        y_weight = float(y_weights[0])
        y_valid = bool(y_valid_values[0])

    coverage_fraction = float(
        np.mean(x_valid, dtype=float)
        * np.mean(z_valid, dtype=float)
        * float(y_valid)
    )
    if not y_valid:
        raise ValueError(
            "The training middle-vy slice is outside the source velocity grid"
        )
    if coverage_fraction == 0.0:
        raise ValueError(
            "The source and training velocity grids have no xz-plane overlap"
        )

    return {
        "source_grid": source_grid,
        "source_shape": source_shape.copy(),
        "target_shape": target_shape.copy(),
        "source_y_indices": source_y_indices,
        "y_weight": np.float32(y_weight),
        "x_lower": x_lower,
        "x_upper": x_upper,
        "x_weight": x_weight,
        "x_valid": x_valid,
        "z_lower": z_lower,
        "z_upper": z_upper,
        "z_weight": z_weight,
        "z_valid": z_valid,
        "identity": bool(identity),
        "coverage_fraction": coverage_fraction,
    }


def _create_velocity_cell_centers(velocity_grid):
    """Create velocity-cell-center coordinates for each grid axis."""

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
    """Create lower indices, upper indices, weights, and validity mask."""

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


def _interpolate_xz_planes(source_planes, interpolation_plan, dtype):
    """Interpolate two source xz planes using a cached trilinear plan."""

    if interpolation_plan["identity"]:
        return np.asarray(source_planes[0], dtype=dtype).copy()

    y_weight = interpolation_plan["y_weight"]
    source_plane = (
        (1.0 - y_weight) * source_planes[0]
        + y_weight * source_planes[1]
    )
    x_lower = interpolation_plan["x_lower"][:, None]
    x_upper = interpolation_plan["x_upper"][:, None]
    z_lower = interpolation_plan["z_lower"][None, :]
    z_upper = interpolation_plan["z_upper"][None, :]
    x_weight = interpolation_plan["x_weight"][:, None]
    z_weight = interpolation_plan["z_weight"][None, :]

    result = (
        (1.0 - x_weight)
        * (1.0 - z_weight)
        * source_plane[x_lower, z_lower]
        + x_weight
        * (1.0 - z_weight)
        * source_plane[x_upper, z_lower]
        + (1.0 - x_weight)
        * z_weight
        * source_plane[x_lower, z_upper]
        + x_weight
        * z_weight
        * source_plane[x_upper, z_upper]
    )
    valid = (
        interpolation_plan["x_valid"][:, None]
        & interpolation_plan["z_valid"][None, :]
    )
    result[~valid] = 0.0

    return np.asarray(result, dtype=dtype)


def create_sorted_velocity_indices(reader, n_velocity_cells, pop=None):
    """
    Create the velocity-cell ordering used by VDF extraction.

    Parameters
    ----------
    reader : analysator.vlsvfile.VlsvReader
        Open VLSV file reader.
    n_velocity_cells : int
        Number of cells in the dense velocity mesh.
    pop : str, optional
        Explicit particle population override. If omitted, ``avgs`` is used
        when available and ``proton`` otherwise.

    Returns
    -------
    numpy.ndarray
        Indices that sort velocity cells in the same order as the original
        repeated stable sorting implementation.
    """

    pop = resolve_velocity_population(reader=reader, pop=pop)
    vids = np.arange(int(n_velocity_cells))
    velocity_coords = reader.get_velocity_cell_coordinates(vids, pop)
    sorted_indices = vids

    for axis_index in range(3):
        axis_order = np.argsort(
            velocity_coords[:, axis_index],
            kind="stable",
        )
        velocity_coords = velocity_coords[axis_order]
        sorted_indices = sorted_indices[axis_order]

    return sorted_indices


def extract_vdf_from_reader(reader, cid, box=-1, pop=None):
    """
    Extract a 3D VDF from one Vlasiator cell.

    This function reads the sparse velocity-space data stored in a `.vlsv`
    file, places the values into a full velocity-space grid, sorts the grid by
    velocity coordinates, and returns the VDF as a dense NumPy array.

    Parameters
    ----------
    reader : analysator.vlsvfile.VlsvReader
        Open VLSV file reader.
    cid : int
        Spatial cell ID from which the VDF is extracted.
    box : int, optional
        If positive, crop the VDF around its maximum value using this value as
        the half-width of the crop in index space. If `-1`, return VDF.
    pop : str, optional
        Explicit particle population override. If omitted, ``avgs`` is used
        when available and ``proton`` otherwise.

    Returns
    -------
    numpy.ndarray
        Dense 3D VDF array with dtype `float32`. The returned axis order is
        ``[vx, vy, vz]``.
    """

    extractor = VdfExtractor(reader=reader, pop=pop)

    return extractor.extract(cid=cid, box=box)


def extract_vdf(file_location, cid, box=-1, pop=None):
    """
    Extract a 3D VDF from one Vlasiator cell.

    Parameters
    ----------
    file_location : str
        Path to the `.vlsv` file.
    cid : int
        Spatial cell ID from which the VDF is extracted.
    box : int, optional
        If positive, crop the VDF around its maximum value using this value as
        the half-width of the crop in index space. If `-1`, return VDF.
    pop : str, optional
        Explicit particle population override. If omitted, ``avgs`` is used
        when available and ``proton`` otherwise.

    Returns
    -------
    numpy.ndarray
        Dense 3D VDF array with dtype `float32`.
    """

    reader = pt.vlsvfile.VlsvReader(file_location)
    return extract_vdf_from_reader(
        reader=reader,
        cid=cid,
        box=box,
        pop=pop,
    )
