"""Read magnetic-flux grids and locate gradient-contour intersections.

Flux arrays are indexed as ``[z_index, x_index]`` when read and smoothed.
Finite-difference Hessians and bilinear interpolation use the transposed
``[x_index, z_index]`` view in physical metres.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.path import Path as MplPath
from scipy.signal import convolve2d
from shapely import geometry


def read_smoothed_flux_grid(reader, flux_file_location, kernel_size=5):
    """Read and square-kernel smooth one two-dimensional flux grid.

    Point-topology detection begins with this I/O and conditioning stage.
    Smoothing suppresses grid-scale contour structure before zero-gradient
    intersections are found, while the returned coordinate arrays preserve
    the source file's physical x-z indexing.

    Parameters
    ----------
    reader : analysator.vlsvfile.VlsvReader
        Matching timestep reader used for spatial dimensions and extents.
    flux_file_location : str or pathlib.Path
        Raw double-precision flux-function file.
    kernel_size : int, optional
        Width of the normalized square smoothing kernel in spatial cells.

    Returns
    -------
    x_array : numpy.ndarray
        X coordinates in metres.
    z_array : numpy.ndarray
        Z coordinates in metres.
    flux_function_zx : numpy.ndarray
        Smoothed flux indexed as ``[z_index, x_index]``.
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
    smoothing_kernel = np.ones(
        (kernel_size, kernel_size)
    ) / (kernel_size**2)
    return x_array, z_array, convolve2d(
        flux_function_zx,
        smoothing_kernel,
        "same",
    )


def find_flux_gradient_intersections(x_array, z_array, flux_function_zx):
    """Find x-z intersections of zero flux-gradient contours.

    Candidate magnetic critical points occur where both in-plane flux
    derivatives vanish. This routine extracts those intersections with
    Matplotlib and Shapely; the topology stage that follows classifies each
    candidate from its physical finite-difference Hessian.

    Parameters
    ----------
    x_array : numpy.ndarray
        X coordinates in metres.
    z_array : numpy.ndarray
        Z coordinates in metres.
    flux_function_zx : numpy.ndarray
        Smoothed flux indexed as ``[z_index, x_index]``.

    Returns
    -------
    list of tuple
        Intersection coordinates ``(x_m, z_m)`` in contour iteration order.
    """

    dfdx, dfdz = np.gradient(flux_function_zx)
    fig, ax = plt.subplots()
    contour1 = ax.contour(x_array, z_array, dfdx, [0])
    contour1_paths = get_contour_paths(contour1)
    contour2 = ax.contour(x_array, z_array, dfdz, [0])
    contour2_paths = get_contour_paths(contour2)
    plt.close(fig)

    points = []
    for path1 in contour1_paths:
        for path2 in contour2_paths:
            if (
                path1.intersects_path(path2)
                and len(path1) > 1
                and len(path2) > 1
            ):
                intersection = find_intersection(
                    path1.vertices,
                    path2.vertices,
                )
                points.extend(intersection_to_points(intersection))
    return points


def get_contour_paths(contour):
    """Return all path objects from a Matplotlib contour set.

    Flux-gradient and island calculations share this adapter because
    Matplotlib may expose one contour level through collections or compound
    paths. The function returns only path geometry and does not retain the
    owning figure.

    Parameters
    ----------
    contour : matplotlib.contour.QuadContourSet
        Contour set produced by Matplotlib.

    Returns
    -------
    list
        Nontrivial Matplotlib paths extracted from the contour level.
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
    """Intersect two contour polylines in the physical x-z plane.

    Zero-gradient contour processing delegates the geometric intersection to
    Shapely so point, overlapping-line, and multipart results can be flattened
    consistently by :func:`intersection_to_points`.

    Parameters
    ----------
    v1 : numpy.ndarray
        First polyline vertices with shape ``(n_vertices, 2)``.
    v2 : numpy.ndarray
        Second polyline vertices with shape ``(n_vertices, 2)``.

    Returns
    -------
    shapely.geometry.base.BaseGeometry
        Shapely intersection geometry.
    """

    return geometry.LineString(v1).intersection(geometry.LineString(v2))


def intersection_to_points(intersection):
    """Flatten a Shapely intersection into deterministic x-z points.

    Topology detection requires one simple ordered coordinate list regardless
    of the geometry type returned by Shapely. Multipart and collection
    geometries are traversed recursively without changing component order.

    Parameters
    ----------
    intersection : shapely geometry
        Point, line, multipart, collection, or empty intersection.

    Returns
    -------
    list of tuple
        Physical ``(x, z)`` coordinate pairs.
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
    """Calculate the local two-dimensional magnetic-flux Hessian.

    Point classification calls this after locating a zero-gradient candidate.
    The determinant separates saddles from extrema, and the returned
    eigensystem later orients the physical X-point sampling box.

    Parameters
    ----------
    flux_function_xz : numpy.ndarray
        Flux indexed as ``[x_index, z_index]``.
    i : int
        Interior x grid index.
    j : int
        Interior z grid index.
    dx : float
        Uniform spatial cell width in metres.

    Returns
    -------
    numpy.ndarray
        Symmetric ``(2, 2)`` Hessian in the x-z plane, in source flux units
        per square metre.

    Notes
    -----
    The established centred second differences and mixed four-point stencil
    are retained exactly.
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


def interpolate_flux(flux_function_xz, x, z, xmin, zmin, dx):
    """Bilinearly interpolate magnetic flux at one x-z coordinate.

    Critical-point records use this interpolation because contour
    intersections need not lie at grid nodes. The interpolated value later
    selects X-linked island boundary levels for O-point geometry.

    Parameters
    ----------
    flux_function_xz : numpy.ndarray
        Flux indexed as ``[x_index, z_index]``.
    x : float
        X coordinate in metres.
    z : float
        Z coordinate in metres.
    xmin : float
        Minimum x coordinate in metres.
    zmin : float
        Minimum z coordinate in metres.
    dx : float
        Uniform spatial cell width in metres.

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
