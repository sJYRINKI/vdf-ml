"""Select VDF cells with manual, Hessian, or island-contour geometry.

All spatial coordinates and manual widths use Earth radii except the
Hessian-aligned X-point box, whose eigenvector half-widths are configured in
local ion inertial lengths and evaluated in metres.
"""

import numpy as np
from matplotlib.path import Path as MplPath

from src.data.step_02_find_vdf_cells import R_EARTH


def get_point_selection_config(config, point_kind):
    """Return the selection configuration for one critical-point kind.

    Point-selection policy and geometry call this common resolver so X and O
    branches interpret either a full dataset configuration or a nested points
    mapping identically.

    Parameters
    ----------
    config : dict
        Dataset or nested point-selection configuration.
    point_kind : {"x", "o"}
        Magnetic critical-point kind.

    Returns
    -------
    dict
        Selection configuration for the requested point kind.
    """

    points_config = (config or {}).get("points", config or {})
    if point_kind == "x":
        return points_config.get("x_selection", {})
    return points_config.get("o_selection", {})


def get_manual_config_re(config, point_kind):
    """Return manual point-box half-widths in Earth radii.

    Manual, consensus, and union workflows use these three axis widths for
    both geometric selection and saved selection metadata. A nonpositive
    y half-width leaves the out-of-plane coordinate unrestricted.

    Parameters
    ----------
    config : dict
        Dataset or nested point-selection configuration.
    point_kind : {"x", "o"}
        Magnetic critical-point kind.

    Returns
    -------
    dict
        Positive or zero x/y/z half-widths in Earth radii.
    """

    selection_config = get_point_selection_config(
        config=config,
        point_kind=point_kind,
    )
    box_config = selection_config.get("manual_re", {})
    return {
        "x_half_width_re": float(box_config["x_half_width_re"]),
        "y_half_width_re": float(
            box_config.get(
                "y_half_width_re",
                selection_config.get("y_half_width_re", 0.0),
            )
        ),
        "z_half_width_re": float(box_config["z_half_width_re"]),
    }


def get_vdf_cellids_in_manual(
    config,
    point_record,
    vdf_cellids,
    vdf_coords_re,
):
    """Select cells inside an axis-aligned box around a critical point.

    Manual point selection compares VDF-cell centres directly in Earth radii
    and preserves the input cell order. Axes with zero half-width are omitted
    from the mask rather than interpreted as exact-coordinate constraints.

    Parameters
    ----------
    config : dict
        Dataset point-selection configuration.
    point_record : dict
        X- or O-point record containing ``coord_re`` and ``point_kind``.
    vdf_cellids : numpy.ndarray
        VDF-carrying spatial cell identifiers.
    vdf_coords_re : numpy.ndarray
        Corresponding cell centres in Earth radii with shape
        ``(n_cells, 3)``.

    Returns
    -------
    dict
        Stable manual position names mapped to selected cell IDs.
    """

    if len(vdf_cellids) == 0:
        return {}
    point_kind = point_record["point_kind"]
    box_config = get_manual_config_re(
        config=config,
        point_kind=point_kind,
    )
    half_widths_re = np.asarray(
        [
            box_config["x_half_width_re"],
            box_config["y_half_width_re"],
            box_config["z_half_width_re"],
        ],
        dtype=float,
    )
    center_re = np.asarray(point_record["coord_re"], dtype=float)
    offsets_re = vdf_coords_re - center_re
    active_axes = half_widths_re > 0
    if np.any(active_axes):
        selected = np.all(
            np.abs(offsets_re[:, active_axes])
            <= half_widths_re[active_axes],
            axis=1,
        )
    else:
        selected = np.ones(len(vdf_cellids), dtype=bool)
    return {
        f"{point_kind}_box_{index:04d}": int(cid)
        for index, cid in enumerate(vdf_cellids[selected])
    }


def get_vdf_cellids_in_hessian_di_box(
    config,
    point_record,
    vdf_cellids,
    vdf_coords_re,
):
    """Select cells in an X-point Hessian box scaled by local ``d_i``.

    Physical X-point selection projects x-z offsets in metres onto the two
    column eigenvectors of the local flux Hessian. Configured half-widths are
    multiples of the selected-cell proton inertial length; the optional y
    extent remains in Earth radii.

    Parameters
    ----------
    config : dict
        Dataset point-selection configuration.
    point_record : dict
        X-point record containing Hessian eigenvectors and ``di_m``.
    vdf_cellids : numpy.ndarray
        VDF-carrying spatial cell identifiers.
    vdf_coords_re : numpy.ndarray
        Corresponding cell centres in Earth radii with shape
        ``(n_cells, 3)``.

    Returns
    -------
    dict
        Stable Hessian-box position names mapped to selected cell IDs.
    """

    if len(vdf_cellids) == 0 or point_record.get("di_m") is None:
        return {}
    x_selection = get_point_selection_config(
        config=config,
        point_kind="x",
    )
    half_width_di = x_selection.get("half_width_di", {})
    half_widths_m = np.asarray(
        [
            float(half_width_di["eigenvector_0"]),
            float(half_width_di["eigenvector_1"]),
        ],
        dtype=float,
    ) * float(point_record["di_m"])
    y_half_width_re = float(x_selection.get("y_half_width_re", 0.0))
    center_re = np.asarray(point_record["coord_re"], dtype=float)
    offsets_re = vdf_coords_re - center_re
    offsets_xz_m = offsets_re[:, [0, 2]] * R_EARTH
    eigvecs = np.asarray(point_record["eigvecs"], dtype=float)
    projections_m = offsets_xz_m @ eigvecs
    selected = (
        (np.abs(projections_m[:, 0]) <= half_widths_m[0])
        & (np.abs(projections_m[:, 1]) <= half_widths_m[1])
        & (
            (y_half_width_re <= 0)
            | (np.abs(offsets_re[:, 1]) <= y_half_width_re)
        )
    )
    return {
        f"x_di_{index:04d}": int(cid)
        for index, cid in enumerate(vdf_cellids[selected])
    }


def get_hessian_di_box_vertices_re(config, point_record):
    """Construct the exact x-z polygon used by physical X-point selection.

    Dataset-frame plotting uses this pure geometry projection after stage 3
    has attached the local ion inertial length and flux-Hessian eigensystem.
    The four local corners use the same eigenvector-column convention and
    configured ``d_i`` half-widths as
    :func:`get_vdf_cellids_in_hessian_di_box`, then return to global x-z
    coordinates for Matplotlib.

    Parameters
    ----------
    config : dict
        Dataset point-selection configuration containing
        ``points.x_selection.half_width_di``.
    point_record : dict
        Accepted X-point record with ``coord_re``, ``eigvecs``, and ``di_m``.

    Returns
    -------
    numpy.ndarray
        Counterclockwise or clockwise polygon corners with shape ``(4, 2)``
        in Earth radii. Columns are x and z.

    Notes
    -----
    Hessian eigenvectors are columns. Local coordinates therefore return to
    the simulation frame through ``local_corners @ eigvecs.T``.
    """

    x_selection = get_point_selection_config(
        config=config,
        point_kind="x",
    )
    half_width_di = x_selection["half_width_di"]
    half_widths_m = np.asarray(
        [
            float(half_width_di["eigenvector_0"]),
            float(half_width_di["eigenvector_1"]),
        ],
        dtype=float,
    ) * float(point_record["di_m"])
    local_corners_m = np.asarray(
        [
            [-half_widths_m[0], -half_widths_m[1]],
            [half_widths_m[0], -half_widths_m[1]],
            [half_widths_m[0], half_widths_m[1]],
            [-half_widths_m[0], half_widths_m[1]],
        ],
        dtype=float,
    )
    eigvecs = np.asarray(point_record["eigvecs"], dtype=float)
    center_xz_re = np.asarray(point_record["coord_re"], dtype=float)[[0, 2]]
    return center_xz_re + local_corners_m @ eigvecs.T / R_EARTH


def get_vdf_cellids_in_flux_contour(
    config,
    point_record,
    vdf_cellids,
    vdf_coords_re,
):
    """Select cells inside an O-point closed flux contour.

    Physical O-point selection tests each VDF-cell x-z centre against the
    island core contour prepared during topology detection. An optional
    Earth-radius y half-width restricts the otherwise planar contour
    selection.

    Parameters
    ----------
    config : dict
        Dataset point-selection configuration.
    point_record : dict
        O-point record containing ``contour_vertices_re``.
    vdf_cellids : numpy.ndarray
        VDF-carrying spatial cell identifiers.
    vdf_coords_re : numpy.ndarray
        Corresponding cell centres in Earth radii with shape
        ``(n_cells, 3)``.

    Returns
    -------
    dict
        Stable island position names mapped to selected cell IDs.
    """

    if len(vdf_cellids) == 0:
        return {}
    contour_vertices_re = point_record.get("contour_vertices_re")
    if contour_vertices_re is None:
        return {}
    o_selection = get_point_selection_config(
        config=config,
        point_kind="o",
    )
    y_half_width_re = float(o_selection.get("y_half_width_re", 0.0))
    center_re = np.asarray(point_record["coord_re"], dtype=float)
    offsets_re = vdf_coords_re - center_re
    contour_path = MplPath(np.asarray(contour_vertices_re, dtype=float))
    selected = (
        contour_path.contains_points(vdf_coords_re[:, [0, 2]])
        & (
            (y_half_width_re <= 0)
            | (np.abs(offsets_re[:, 1]) <= y_half_width_re)
        )
    )
    return {
        f"o_island_{index:04d}": int(cid)
        for index, cid in enumerate(vdf_cellids[selected])
    }
