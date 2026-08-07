"""Associate O points with closed magnetic-island flux contours.

Point-topology detection calls this reusable geometry after classifying X and
O critical points. X-point flux levels define candidate island boundaries;
the selected interior contour is saved in Earth radii for physical O-point
VDF-cell selection.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.path import Path as MplPath

from src.data.step_02_find_vdf_cells import R_EARTH
from src.physics.flux_grid import get_contour_paths


def add_o_point_island_contours(
    o_point_records,
    x_point_records,
    flux_function_zx,
    x_array,
    z_array,
    points_config=None,
):
    """Attach one X-linked closed search contour to each O point.

    The topology workflow calls this after both point classes are known. It
    mutates each O-point record with the selected search contour and
    associated X point while reusing one temporary Matplotlib axes for all
    candidates and closing that figure before returning.

    Parameters
    ----------
    o_point_records : list of dict
        O-point records updated in place.
    x_point_records : list of dict
        Candidate X-point records defining boundary flux levels.
    flux_function_zx : numpy.ndarray
        Smoothed flux indexed as ``[z_index, x_index]``.
    x_array : numpy.ndarray
        X coordinates in metres.
    z_array : numpy.ndarray
        Z coordinates in metres.
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
    """Find the smallest X-flux boundary and its O-core search contour.

    Each X-point flux supplies a candidate closed boundary around the O point.
    Candidates are ordered by enclosing area, flux difference, and point
    distance; the chosen core contour lies the configured fraction of the way
    from O flux toward that boundary flux.

    Parameters
    ----------
    o_point_record : dict
        O-point record with physical coordinates and flux.
    x_point_records : list of dict
        Candidate X-point boundary records.
    flux_function_zx : numpy.ndarray
        Smoothed flux indexed as ``[z_index, x_index]``.
    x_array : numpy.ndarray
        X coordinates in metres.
    z_array : numpy.ndarray
        Z coordinates in metres.
    core_fraction : float
        Fraction from O-point flux toward boundary X-point flux.
    contour_ax : matplotlib.axes.Axes, optional
        Reusable axes for contour extraction.

    Returns
    -------
    dict or None
        Boundary/search flux, Earth-radius contour vertices, and associated
        X point, or ``None`` when no enclosing contour exists.
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
        search_flux = (
            o_point_flux
            + core_fraction * (boundary_flux - o_point_flux)
        )
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
    """Find the smallest closed flux contour enclosing one x-z point.

    Island association uses this operation for both boundary and core flux
    levels. Open paths and paths that do not contain the requested physical
    point are ignored; supplied axes are cleared after use, while internally
    created figures are also closed.

    Parameters
    ----------
    x_array : numpy.ndarray
        X coordinates in metres.
    z_array : numpy.ndarray
        Z coordinates in metres.
    flux_function_zx : numpy.ndarray
        Flux indexed as ``[z_index, x_index]``.
    contour_flux : float
        Flux level to contour.
    point_xz : tuple of float
        Test point in the same physical units as the coordinate arrays.
    contour_ax : matplotlib.axes.Axes, optional
        Reusable axes; a temporary figure is created when omitted.

    Returns
    -------
    tuple or None
        ``(area, vertices)`` for the smallest enclosing contour in coordinate
        units, or ``None``.
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
    """Calculate absolute polygon area with the shoelace formula.

    Enclosing contours are ranked by this orientation-independent area so
    clockwise and counter-clockwise Matplotlib paths produce the same physical
    candidate order.

    Parameters
    ----------
    vertices : numpy.ndarray
        Ordered polygon vertices with shape ``(n_vertices, 2)``.

    Returns
    -------
    float
        Absolute area in squared coordinate units.
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
