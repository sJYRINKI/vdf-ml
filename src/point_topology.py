import logging

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.path import Path as MplPath
from scipy.signal import convolve2d
from shapely import geometry

from src.point_selection import get_point_selection_method
from src.vdf_helpers import R_EARTH, find_matching_region_name_re

PROTON_MASS = 1.67262192369e-27
ELEMENTARY_CHARGE = 1.602176634e-19
MU0 = 4.0e-7 * np.pi


def find_point_coords_re(reader, flux_file_location, points_config=None):
    """
    Find X- and O-point coordinates from a flux file.

    Parameters
    ----------
    reader : analysator.vlsvfile.VlsvReader
        Open VLSV file reader for matching timestep.
    flux_file_location : pathlib.Path
        Flux filepath.
    points_config : dict, optional
        Point configuration.

    Returns
    -------
    x_point_coords_re : list of list of float
        X-point coordinates in Earth radii.
    o_point_coords_re : list of list of float
        O-point coordinates in Earth radii.
    """

    x_point_records, o_point_records = find_point_records(
        reader=reader,
        flux_file_location=flux_file_location,
        points_config=points_config,
    )

    return (
        [record["coord_re"] for record in x_point_records],
        [record["coord_re"] for record in o_point_records],
    )


def find_point_records(reader, flux_file_location, points_config=None):
    """
    Find X- and O-point records from a flux file.

    Parameters
    ----------
    reader : analysator.vlsvfile.VlsvReader
        Open VLSV file reader for matching timestep.
    flux_file_location : pathlib.Path
        Flux filepath.
    points_config : dict, optional
        Point configuration.

    Returns
    -------
    x_point_records : list of dict
        X-point records with coordinates, flux, Hessian axes, and optional
        ``d_i`` metadata.
    o_point_records : list of dict
        O-point records with coordinates, flux, and optional island boundary
        contours.
    """

    x_array, z_array, flux_function_zx = read_smoothed_flux_grid(
        reader=reader,
        flux_file_location=flux_file_location,
    )
    x_cells = len(x_array)
    z_cells = len(z_array)
    dx = float(x_array[1] - x_array[0])
    xmin = float(x_array[0])
    zmin = float(z_array[0])

    dfdx, dfdz = np.gradient(flux_function_zx)

    fig, ax = plt.subplots()
    contour1 = ax.contour(x_array, z_array, dfdx, [0])
    contour1_paths = get_contour_paths(contour1)

    contour2 = ax.contour(x_array, z_array, dfdz, [0])
    contour2_paths = get_contour_paths(contour2)
    plt.close(fig)

    x_coords = []
    z_coords = []

    for path1 in contour1_paths:
        for path2 in contour2_paths:
            if path1.intersects_path(path2) and len(path1) > 1 and len(path2) > 1:
                intersection = find_intersection(path1.vertices, path2.vertices)

                for x_coord, z_coord in intersection_to_points(intersection):
                    x_coords.append(x_coord)
                    z_coords.append(z_coord)

    x_point_records = []
    o_point_records = []
    flux_function_xz = flux_function_zx.T
    x_selection_method = get_point_selection_method(
        config=points_config,
        point_kind="x",
    )
    o_selection_method = get_point_selection_method(
        config=points_config,
        point_kind="o",
    )
    physical_selection_methods = {
        "physical",
        "consensus",
        "union_physical_priority",
    }

    for k in range(len(x_coords)):
        coords = [x_coords[k], 0, z_coords[k]]
        cellid = reader.get_cellid(coords)
        i = int((cellid - 1) % x_cells)
        j = (int(cellid) - 1) // x_cells

        if i <= 100 or j <= 100 or i >= x_cells - 100 or j >= z_cells - 100:
            continue

        hessian = calculate_hessian(
            flux_function_xz=flux_function_xz,
            i=i,
            j=j,
            dx=dx,
        )
        det_hessian = np.linalg.det(hessian)
        eigvals, eigvecs = np.linalg.eigh(hessian)

        coord_re = [
            float(coords[0] / R_EARTH),
            float(coords[1] / R_EARTH),
            float(coords[2] / R_EARTH),
        ]

        region_name = find_matching_region_name_re(
            coord_re=coord_re,
            points_config=points_config,
            names_key="point_region_names",
        )
        if region_name is None:
            continue

        interpolated_flux = interpolate_flux(
            flux_function_xz=flux_function_xz,
            x=float(coords[0]),
            z=float(coords[2]),
            xmin=float(xmin),
            zmin=float(zmin),
            dx=float(dx),
        )

        if det_hessian < 0:
            point_record = create_point_record(
                point_kind="x",
                coord_m=coords,
                coord_re=coord_re,
                cellid=cellid,
                flux=interpolated_flux,
                hessian=hessian,
                eigvals=eigvals,
                eigvecs=eigvecs,
            )
            point_record["region_name"] = region_name
            if x_selection_method in physical_selection_methods:
                add_ion_inertial_length(
                    reader=reader,
                    point_record=point_record,
                    points_config=points_config,
                )
                if (
                    point_record.get("di_m") is None
                    and x_selection_method == "physical"
                ):
                    logging.warning(
                        "Skipping X point at %s because local d_i could not be computed",
                        coord_re,
                    )
                    continue
            x_point_records.append(point_record)

        if det_hessian > 0 and hessian[0, 0] < 0:
            point_record = create_point_record(
                point_kind="o",
                coord_m=coords,
                coord_re=coord_re,
                cellid=cellid,
                flux=interpolated_flux,
                hessian=hessian,
                eigvals=eigvals,
                eigvecs=eigvecs,
            )
            point_record["region_name"] = region_name
            o_point_records.append(point_record)

    if o_selection_method in physical_selection_methods:
        add_o_point_island_contours(
            o_point_records=o_point_records,
            x_point_records=x_point_records,
            flux_function_zx=flux_function_zx,
            x_array=x_array,
            z_array=z_array,
            points_config=points_config,
        )

        if o_selection_method == "physical":
            o_point_records = [
                point_record
                for point_record in o_point_records
                if point_record.get("contour_vertices_re") is not None
            ]

    return x_point_records, o_point_records


def read_smoothed_flux_grid(reader, flux_file_location, kernel_size=5):
    """
    Read and smooth the 2D flux grid in simulation coordinates.

    Parameters
    ----------
    reader : analysator.vlsvfile.VlsvReader
        Reader for the timestep VLSV file.
    flux_file_location : pathlib.Path
        Path to the flux file.
    kernel_size : int, optional
        Width of the square smoothing kernel in grid cells.

    Returns
    -------
    x_array : numpy.ndarray
        X coordinates in meters.
    z_array : numpy.ndarray
        Z coordinates in meters.
    flux_function_zx : numpy.ndarray
        Smoothed flux function indexed as ``[z_index, x_index]``.
    """

    x_cells = int(reader.get_spatial_mesh_size()[0])
    z_cells = int(reader.get_spatial_mesh_size()[2])
    xsize = reader.read_parameter("xcells_ini")
    xmax = reader.read_parameter("xmax")
    xmin = reader.read_parameter("xmin")
    zmin = reader.read_parameter("zmin")
    dx = float((xmax - xmin) / xsize)

    x_array = float(xmin) + np.arange(x_cells) * dx
    z_array = float(zmin) + np.arange(z_cells) * dx

    flux_function_zx = np.fromfile(
        flux_file_location,
        dtype="double",
    ).reshape(z_cells, x_cells)

    smoothing_kernel = np.ones((kernel_size, kernel_size)) / (kernel_size**2)

    return x_array, z_array, convolve2d(
        flux_function_zx,
        smoothing_kernel,
        "same",
    )


def get_contour_paths(contour):
    """
    Get contour paths from a Matplotlib contour object.

    Parameters
    ----------
    contour : matplotlib.contour.QuadContourSet
        Contour object returned by ``plt.contour``.

    Returns
    -------
    list
        Contour path objects.
    """

    if hasattr(contour, "collections"):
        return contour.collections[0].get_paths()

    paths = []

    for path in contour.get_paths():
        for vertices in path.to_polygons(closed_only=False):
            if len(vertices) > 1:
                paths.append(MplPath(vertices))

    return paths


def find_intersection(v1, v2):
    """
    Find intersection points of two contour paths.

    Parameters
    ----------
    v1 : numpy.ndarray
        Vertices of the first contour path.
    v2 : numpy.ndarray
        Vertices of the second contour path.

    Returns
    -------
    intersection : shapely.geometry.LineString
        The intersection of the two contour paths.
    """

    poly1 = geometry.LineString(v1)
    poly2 = geometry.LineString(v2)

    intersection = poly1.intersection(poly2)
    return intersection


def intersection_to_points(intersection):
    """
    Convert a Shapely intersection geometry to coordinate points.

    Parameters
    ----------
    intersection : shapely geometry
        Intersection geometry.

    Returns
    -------
    list of tuple
        List of ``(x, z)`` coordinate pairs.
    """

    if intersection.is_empty:
        return []

    if intersection.geom_type == "Point":
        return [(float(intersection.x), float(intersection.y))]

    if intersection.geom_type == "MultiPoint":
        return [
            (float(point.x), float(point.y))
            for point in intersection.geoms
        ]

    if intersection.geom_type in {"LineString", "LinearRing"}:
        return [
            (float(point[0]), float(point[1]))
            for point in intersection.coords
        ]

    if hasattr(intersection, "geoms"):
        points = []

        for geom in intersection.geoms:
            points.extend(intersection_to_points(geom))

        return points

    return []


def calculate_hessian(flux_function_xz, i, j, dx):
    """
    Calculate the Hessian matrix of the flux function.

    Parameters
    ----------
    flux_function_xz : numpy.ndarray
        Flux function indexed as ``[x_index, z_index]``.
    i : int
        X grid index.
    j : int
        Z grid index.
    dx : float
        Spatial cell size in meters.

    Returns
    -------
    numpy.ndarray
        Hessian matrix in the x-z plane.
    """

    delta_psi_xx = (
        flux_function_xz[i + 1, j]
        - 2 * flux_function_xz[i, j]
        + flux_function_xz[i - 1, j]
    ) / dx**2
    delta_psi_zz = (
        flux_function_xz[i, j + 1]
        - 2 * flux_function_xz[i, j]
        + flux_function_xz[i, j - 1]
    ) / dx**2
    delta_psi_xz = (
        flux_function_xz[i + 1, j + 1]
        - flux_function_xz[i + 1, j - 1]
        - flux_function_xz[i - 1, j + 1]
        + flux_function_xz[i - 1, j - 1]
    ) / (4 * dx**2)

    return np.array(
        [
            [delta_psi_xx, delta_psi_xz],
            [delta_psi_xz, delta_psi_zz],
        ],
        dtype=float,
    )


def is_in_point_region(coord_re, points_config):
    """
    Check whether a point coordinate is in the configured region.

    Parameters
    ----------
    coord_re : list of float
        Coordinate in Earth radii, given as ``[x, y, z]``.
    points_config : dict
        Point configuration.

    Returns
    -------
    bool
        Whether the coordinate should be used as a labeled point.
    """

    return find_matching_region_name_re(
        coord_re=coord_re,
        points_config=points_config,
        names_key="point_region_names",
    ) is not None


def interpolate_flux(flux_function_xz, x, z, xmin, zmin, dx):
    """
    Interpolate flux function value at a spatial coordinate.

    Parameters
    ----------
    flux_function_xz : numpy.ndarray
        Flux function indexed as ``[x_index, z_index]``.
    x : float
        X coordinate in meters.
    z : float
        Z coordinate in meters.
    xmin : float
        Minimum x coordinate in meters.
    zmin : float
        Minimum z coordinate in meters.
    dx : float
        Spatial cell size in meters.

    Returns
    -------
    float
        Bilinearly interpolated flux value.
    """

    x_index = (x - xmin) / dx
    z_index = (z - zmin) / dx
    i = int(np.floor(x_index))
    j = int(np.floor(z_index))
    i = max(0, min(flux_function_xz.shape[0] - 2, i))
    j = max(0, min(flux_function_xz.shape[1] - 2, j))
    x_fraction = float(x_index - i)
    z_fraction = float(z_index - j)

    return float(
        (1.0 - z_fraction)
        * (
            (1.0 - x_fraction) * flux_function_xz[i, j]
            + x_fraction * flux_function_xz[i + 1, j]
        )
        + z_fraction
        * (
            (1.0 - x_fraction) * flux_function_xz[i, j + 1]
            + x_fraction * flux_function_xz[i + 1, j + 1]
        )
    )


def create_point_record(
    point_kind,
    coord_m,
    coord_re,
    cellid,
    flux,
    hessian,
    eigvals,
    eigvecs,
):
    """
    Create a magnetic critical-point record.

    Parameters
    ----------
    point_kind : {"x", "o"}
        Critical point type.
    coord_m : array-like of float
        Point coordinate in meters.
    coord_re : array-like of float
        Point coordinate in Earth radii.
    cellid : int
        Spatial cell ID containing the point.
    flux : float
        Interpolated magnetic flux at the point.
    hessian : numpy.ndarray
        Hessian matrix of the flux function.
    eigvals : numpy.ndarray
        Hessian eigenvalues.
    eigvecs : numpy.ndarray
        Hessian eigenvectors.

    Returns
    -------
    dict
        Point record.
    """

    return {
        "is_point_record": True,
        "point_kind": point_kind,
        "coord_m": [float(value) for value in coord_m],
        "coord_re": [float(value) for value in coord_re],
        "cellid": int(cellid),
        "flux": float(flux),
        "hessian": np.asarray(hessian, dtype=float),
        "eigvals": np.asarray(eigvals, dtype=float),
        "eigvecs": np.asarray(eigvecs, dtype=float),
    }


def add_ion_inertial_length(reader, point_record, points_config):
    """
    Add local ion inertial length to an X-point record.

    Parameters
    ----------
    reader : analysator.vlsvfile.VlsvReader
        Open VLSV file reader.
    point_record : dict
        X-point record to update.
    points_config : dict
        Point configuration.
    """

    x_selection_config = (points_config or {}).get("x_selection", {})
    density_variable = x_selection_config.get("density_variable", "rho")

    try:
        number_density = float(
            np.asarray(
                reader.read_variable(density_variable, int(point_record["cellid"]))
            ).squeeze()
        )
        di_m = compute_ion_inertial_length(number_density)
    except Exception as error:
        logging.warning(
            "Could not compute d_i for X point at %s using %s: %s",
            point_record["coord_re"],
            density_variable,
            error,
        )
        point_record["rho"] = None
        point_record["di_m"] = None
        point_record["di_re"] = None
        return

    point_record["rho"] = number_density
    point_record["di_m"] = di_m
    point_record["di_re"] = di_m / R_EARTH


def compute_ion_inertial_length(number_density):
    """
    Compute proton ion inertial length.

    Parameters
    ----------
    number_density : float
        Proton number density in m^-3.

    Returns
    -------
    float
        Ion inertial length in meters.
    """

    if number_density <= 0:
        raise ValueError("Proton number density must be positive")

    return float(
        np.sqrt(
            PROTON_MASS / (MU0 * number_density * ELEMENTARY_CHARGE**2)
        )
    )


def add_o_point_island_contours(
    o_point_records,
    x_point_records,
    flux_function_zx,
    x_array,
    z_array,
    points_config=None,
):
    """
    Attach the smallest X-flux closed contour around each O point.

    Parameters
    ----------
    o_point_records : list of dict
        O-point records to update.
    x_point_records : list of dict
        Candidate X-point records.
    flux_function_zx : numpy.ndarray
        Smoothed flux function indexed as ``[z_index, x_index]``.
    x_array : numpy.ndarray
        X coordinates in meters.
    z_array : numpy.ndarray
        Z coordinates in meters.
    points_config : dict, optional
        Point configuration containing ``o_selection.core_fraction``.
    """

    if not x_point_records:
        for o_point_record in o_point_records:
            o_point_record["contour_vertices_re"] = None
            o_point_record["associated_x_point"] = None
        return

    o_selection_config = (points_config or {}).get("o_selection", {})
    core_fraction = float(o_selection_config.get("core_fraction", 0.4))

    fig, ax = plt.subplots()
    try:
        for o_point_record in o_point_records:
            contour_data = find_island_boundary_contour(
                o_point_record=o_point_record,
                x_point_records=x_point_records,
                flux_function_zx=flux_function_zx,
                x_array=x_array,
                z_array=z_array,
                core_fraction=core_fraction,
                contour_ax=ax,
            )
            if contour_data is None:
                o_point_record["contour_vertices_re"] = None
                o_point_record["associated_x_point"] = None
                continue

            o_point_record.update(contour_data)
    finally:
        plt.close(fig)


def find_island_boundary_contour(
    o_point_record,
    x_point_records,
    flux_function_zx,
    x_array,
    z_array,
    core_fraction,
    contour_ax=None,
):
    """
    Find the smallest closed X-flux contour enclosing an O point.

    Parameters
    ----------
    o_point_record : dict
        O-point record.
    x_point_records : list of dict
        Candidate X-point records.
    flux_function_zx : numpy.ndarray
        Smoothed flux function indexed as ``[z_index, x_index]``.
    x_array : numpy.ndarray
        X coordinates in meters.
    z_array : numpy.ndarray
        Z coordinates in meters.
    core_fraction : float
        Fraction from O-point flux toward boundary X-point flux used for the
        actual O-core search contour.
    contour_ax : matplotlib.axes.Axes, optional
        Reusable Matplotlib axes used for contour extraction.

    Returns
    -------
    dict or None
        Boundary contour metadata, or ``None`` if no valid contour was found.
    """

    core_fraction = min(1.0, max(0.0, float(core_fraction)))
    candidates = []
    flux_min = float(np.nanmin(flux_function_zx))
    flux_max = float(np.nanmax(flux_function_zx))
    o_point_flux = float(o_point_record["flux"])
    o_point_xz_m = (
        float(o_point_record["coord_m"][0]),
        float(o_point_record["coord_m"][2]),
    )

    for x_point_record in x_point_records:
        boundary_flux = float(x_point_record["flux"])
        if boundary_flux < flux_min or boundary_flux > flux_max:
            continue

        boundary_contour = find_smallest_closed_contour(
            x_array=x_array,
            z_array=z_array,
            flux_function_zx=flux_function_zx,
            contour_flux=boundary_flux,
            point_xz=o_point_xz_m,
            contour_ax=contour_ax,
        )
        if boundary_contour is None:
            continue

        boundary_area, _ = boundary_contour
        search_flux = o_point_flux + core_fraction * (boundary_flux - o_point_flux)
        search_contour = find_smallest_closed_contour(
            x_array=x_array,
            z_array=z_array,
            flux_function_zx=flux_function_zx,
            contour_flux=search_flux,
            point_xz=o_point_xz_m,
            contour_ax=contour_ax,
        )
        if search_contour is None:
            continue

        search_area, search_vertices = search_contour
        distance = np.linalg.norm(
            np.asarray(o_point_record["coord_re"], dtype=float)
            - np.asarray(x_point_record["coord_re"], dtype=float)
        )
        flux_difference = abs(o_point_flux - boundary_flux)
        candidates.append(
            {
                "boundary_area": boundary_area,
                "search_area": search_area,
                "distance": float(distance),
                "flux_difference": float(flux_difference),
                "boundary_flux": boundary_flux,
                "search_flux": float(search_flux),
                "core_fraction": core_fraction,
                "contour_vertices_re": search_vertices / R_EARTH,
                "associated_x_point": {
                    "coord_re": x_point_record["coord_re"],
                    "flux": float(x_point_record["flux"]),
                },
            }
        )

    if not candidates:
        return None

    candidates.sort(
        key=lambda candidate: (
            candidate["boundary_area"],
            candidate["flux_difference"],
            candidate["distance"],
        )
    )
    best_candidate = candidates[0]

    return {
        "boundary_flux": best_candidate["boundary_flux"],
        "search_flux": best_candidate["search_flux"],
        "core_fraction": best_candidate["core_fraction"],
        "contour_vertices_re": best_candidate["contour_vertices_re"],
        "associated_x_point": best_candidate["associated_x_point"],
    }


def find_smallest_closed_contour(
    x_array,
    z_array,
    flux_function_zx,
    contour_flux,
    point_xz,
    contour_ax=None,
):
    """
    Find the smallest closed contour at one flux level enclosing a point.

    Parameters
    ----------
    x_array : numpy.ndarray
        X coordinates in meters.
    z_array : numpy.ndarray
        Z coordinates in meters.
    flux_function_zx : numpy.ndarray
        Flux function indexed as ``[z_index, x_index]``.
    contour_flux : float
        Flux level to contour.
    point_xz : tuple of float
        Point coordinate in the same units as ``x_array`` and ``z_array``.
    contour_ax : matplotlib.axes.Axes, optional
        Reusable Matplotlib axes used for contour extraction. If omitted, a
        temporary figure and axes are created for this contour.

    Returns
    -------
    tuple or None
        Pair of ``(area, vertices)`` for the smallest enclosing contour.
    """

    fig = None
    if contour_ax is None:
        fig, contour_ax = plt.subplots()

    try:
        contour = contour_ax.contour(
            x_array,
            z_array,
            flux_function_zx,
            [contour_flux],
        )
        contour_paths = get_contour_paths(contour)

        candidates = []
        for contour_path in contour_paths:
            vertices = np.asarray(contour_path.vertices, dtype=float)
            if len(vertices) < 3:
                continue

            if not np.allclose(vertices[0], vertices[-1]):
                continue

            if not MplPath(vertices).contains_point(point_xz):
                continue

            area = polygon_area(vertices)
            if area <= 0:
                continue

            candidates.append((area, vertices))
    finally:
        contour_ax.cla()
        if fig is not None:
            plt.close(fig)

    if not candidates:
        return None

    candidates.sort(key=lambda candidate: candidate[0])
    return candidates[0]


def polygon_area(vertices):
    """
    Compute polygon area with the shoelace formula.

    Parameters
    ----------
    vertices : numpy.ndarray
        Polygon vertices with shape ``(n_vertices, 2)``.

    Returns
    -------
    float
        Absolute polygon area.
    """

    x_coords = vertices[:, 0]
    z_coords = vertices[:, 1]

    return float(
        0.5
        * abs(
            np.dot(x_coords, np.roll(z_coords, -1))
            - np.dot(z_coords, np.roll(x_coords, -1))
        )
    )
