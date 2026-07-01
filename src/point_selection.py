import numpy as np
from matplotlib.path import Path as MplPath

from src.vdf_helpers import R_EARTH


def is_point_record(labeled_coord):
    """
    Check whether a labeled coordinate is a detected X/O point record.

    Parameters
    ----------
    labeled_coord : object
        Labeled coordinate entry.

    Returns
    -------
    bool
        Whether the entry is a point record.
    """

    return isinstance(labeled_coord, dict) and labeled_coord.get("is_point_record")


def unpack_labeled_coord(labeled_coord):
    """
    Return class name, label, and coordinate from a labeled entry.

    Parameters
    ----------
    labeled_coord : tuple or dict
        Static coordinate tuple or detected point record.

    Returns
    -------
    tuple
        ``(class_name, label, coord_re)``.
    """

    if is_point_record(labeled_coord):
        return (
            labeled_coord["class_name"],
            int(labeled_coord["label"]),
            labeled_coord["coord_re"],
        )

    class_name, label, coord_re = labeled_coord
    return class_name, int(label), coord_re


def get_point_cellids_by_position(config, point_record, vdf_cellids, vdf_coords_re):
    """
    Select VDF cell IDs for a detected X/O point.

    Parameters
    ----------
    config : dict
        Dataset configuration.
    point_record : dict
        Detected point record.
    vdf_cellids : numpy.ndarray
        Spatial cell IDs with VDF data.
    vdf_coords_re : numpy.ndarray
        VDF cell coordinates in Earth radii with shape ``(n_cells, 3)``.

    Returns
    -------
    dict
        Mapping from neighbor position name to VDF cell ID.
    """

    return get_point_selection_result(
        config=config,
        point_record=point_record,
        vdf_cellids=vdf_cellids,
        vdf_coords_re=vdf_coords_re,
    )["cellids_by_position"]


def get_point_selection_result(config, point_record, vdf_cellids, vdf_coords_re):
    """
    Select VDF cell IDs and rejected candidates for a detected X/O point.

    Parameters
    ----------
    config : dict
        Dataset configuration.
    point_record : dict
        Detected X/O point record.
    vdf_cellids : numpy.ndarray
        Spatial cell IDs with VDF data.
    vdf_coords_re : numpy.ndarray
        VDF cell coordinates in Earth radii with shape ``(n_cells, 3)``.

    Returns
    -------
    dict
        Selection result with selected ``cellids_by_position``,
        ``rejected_cellids``, and optional ``metadata_by_position``.
    """

    point_kind = point_record["point_kind"]
    selection_method = get_point_selection_method(
        config=config,
        point_kind=point_kind,
    )

    if selection_method == "manual":
        return create_point_selection_result(
            cellids_by_position=get_vdf_cellids_in_manual(
                config=config,
                point_record=point_record,
                vdf_cellids=vdf_cellids,
                vdf_coords_re=vdf_coords_re,
            )
        )

    physical_cellids_by_position = get_physical_point_cellids_by_position(
        config=config,
        point_record=point_record,
        vdf_cellids=vdf_cellids,
        vdf_coords_re=vdf_coords_re,
    )

    if selection_method == "physical":
        return create_point_selection_result(
            cellids_by_position=physical_cellids_by_position
        )

    manual_cellids_by_position = get_vdf_cellids_in_manual(
        config=config,
        point_record=point_record,
        vdf_cellids=vdf_cellids,
        vdf_coords_re=vdf_coords_re,
    )

    if selection_method == "consensus":
        return create_consensus_point_selection_result(
            physical_cellids_by_position=physical_cellids_by_position,
            manual_cellids_by_position=manual_cellids_by_position,
        )

    if selection_method == "union_physical_priority":
        return create_union_physical_priority_point_selection_result(
            physical_cellids_by_position=physical_cellids_by_position,
            manual_cellids_by_position=manual_cellids_by_position,
        )

    raise ValueError(f"Unsupported point selection method: {selection_method}")


def get_physical_point_cellids_by_position(
    config,
    point_record,
    vdf_cellids,
    vdf_coords_re,
):
    """
    Select VDF cell IDs with the physical method for one X/O point.

    Parameters
    ----------
    config : dict
        Dataset configuration.
    point_record : dict
        Detected X/O point record.
    vdf_cellids : numpy.ndarray
        Spatial cell IDs with VDF data.
    vdf_coords_re : numpy.ndarray
        VDF cell coordinates in Earth radii with shape ``(n_cells, 3)``.

    Returns
    -------
    dict
        Mapping from physical selection position name to VDF cell ID.
    """

    point_kind = point_record["point_kind"]

    if point_kind == "x":
        return get_vdf_cellids_in_hessian_di_box(
            config=config,
            point_record=point_record,
            vdf_cellids=vdf_cellids,
            vdf_coords_re=vdf_coords_re,
        )

    if point_kind == "o":
        return get_vdf_cellids_in_flux_contour(
            config=config,
            point_record=point_record,
            vdf_cellids=vdf_cellids,
            vdf_coords_re=vdf_coords_re,
        )

    raise ValueError(f"Unknown point kind: {point_kind}")


def create_point_selection_result(
    cellids_by_position,
    rejected_cellids=None,
    metadata_by_position=None,
):
    """
    Create a normalized point-selection result dictionary.

    Parameters
    ----------
    cellids_by_position : dict
        Mapping from selection position name to VDF cell ID.
    rejected_cellids : set of int, optional
        VDF cell IDs rejected from point and background classes.
    metadata_by_position : dict, optional
        Additional metadata for selected positions.

    Returns
    -------
    dict
        Point-selection result.
    """

    return {
        "cellids_by_position": {
            position: int(cid)
            for position, cid in cellids_by_position.items()
        },
        "rejected_cellids": {
            int(cid)
            for cid in (rejected_cellids or set())
        },
        "metadata_by_position": metadata_by_position or {},
    }


def create_consensus_point_selection_result(
    physical_cellids_by_position,
    manual_cellids_by_position,
):
    """
    Keep only cells selected by both physical and manual methods.

    Parameters
    ----------
    physical_cellids_by_position : dict
        Physical selection positions and VDF cell IDs.
    manual_cellids_by_position : dict
        Manual selection positions and VDF cell IDs.

    Returns
    -------
    dict
        Consensus point-selection result. Disagreement cells are rejected from
        all classes so they cannot become background samples.
    """

    physical_positions_by_cellid = invert_cellids_by_position(
        physical_cellids_by_position
    )
    manual_positions_by_cellid = invert_cellids_by_position(
        manual_cellids_by_position
    )
    physical_cellids = set(physical_positions_by_cellid)
    manual_cellids = set(manual_positions_by_cellid)
    consensus_cellids = physical_cellids & manual_cellids
    rejected_cellids = (physical_cellids | manual_cellids) - consensus_cellids

    cellids_by_position = {}
    metadata_by_position = {}
    for position, cid in physical_cellids_by_position.items():
        cid = int(cid)
        if cid not in consensus_cellids:
            continue

        cellids_by_position[position] = cid
        metadata_by_position[position] = create_combined_selection_metadata(
            selection_agreement="both",
            physical_selected=True,
            manual_selected=True,
            plot_selection_method="physical",
        )

    return create_point_selection_result(
        cellids_by_position=cellids_by_position,
        rejected_cellids=rejected_cellids,
        metadata_by_position=metadata_by_position,
    )


def create_union_physical_priority_point_selection_result(
    physical_cellids_by_position,
    manual_cellids_by_position,
):
    """
    Keep all physical/manual cells and prefer physical entries on overlap.

    Parameters
    ----------
    physical_cellids_by_position : dict
        Physical selection positions and VDF cell IDs.
    manual_cellids_by_position : dict
        Manual selection positions and VDF cell IDs.

    Returns
    -------
    dict
        Union point-selection result with physical metadata used for cells
        selected by both methods.
    """

    physical_positions_by_cellid = invert_cellids_by_position(
        physical_cellids_by_position
    )
    manual_positions_by_cellid = invert_cellids_by_position(
        manual_cellids_by_position
    )
    physical_cellids = set(physical_positions_by_cellid)
    manual_cellids = set(manual_positions_by_cellid)

    cellids_by_position = {}
    metadata_by_position = {}
    for position, cid in physical_cellids_by_position.items():
        cid = int(cid)
        manual_selected = cid in manual_cellids
        cellids_by_position[position] = cid
        metadata_by_position[position] = create_combined_selection_metadata(
            selection_agreement="both" if manual_selected else "physical_only",
            physical_selected=True,
            manual_selected=manual_selected,
            plot_selection_method="physical",
        )

    for position, cid in manual_cellids_by_position.items():
        cid = int(cid)
        if cid in physical_cellids:
            continue

        cellids_by_position[position] = cid
        metadata_by_position[position] = create_combined_selection_metadata(
            selection_agreement="manual_only",
            physical_selected=False,
            manual_selected=True,
            plot_selection_method="manual",
        )

    return create_point_selection_result(
        cellids_by_position=cellids_by_position,
        metadata_by_position=metadata_by_position,
    )


def invert_cellids_by_position(cellids_by_position):
    """
    Return the first selection position found for each VDF cell ID.

    Parameters
    ----------
    cellids_by_position : dict
        Mapping from selection position name to VDF cell ID.

    Returns
    -------
    dict
        Mapping from VDF cell ID to selection position name.
    """

    positions_by_cellid = {}
    for position, cid in cellids_by_position.items():
        positions_by_cellid.setdefault(int(cid), position)

    return positions_by_cellid


def create_combined_selection_metadata(
    selection_agreement,
    physical_selected,
    manual_selected,
    plot_selection_method,
):
    """
    Create metadata describing physical/manual selection agreement.

    Parameters
    ----------
    selection_agreement : str
        Agreement category, for example ``"both"`` or ``"manual_only"``.
    physical_selected : bool
        Whether the physical method selected the cell.
    manual_selected : bool
        Whether the manual method selected the cell.
    plot_selection_method : {"physical", "manual"}
        Selection geometry that should be emphasized in diagnostic plots.

    Returns
    -------
    dict
        Metadata fields for one selected VDF sample.
    """

    return {
        "selection_agreement": selection_agreement,
        "physical_selected": bool(physical_selected),
        "manual_selected": bool(manual_selected),
        "plot_selection_method": plot_selection_method,
    }


def get_point_selection_method(config, point_kind):
    """
    Return the configured selection method for one point kind.

    Parameters
    ----------
    config : dict
        Dataset configuration or ``points`` configuration.
    point_kind : {"x", "o"}
        Point kind whose selection method is read.

    Returns
    -------
    str
        Selection method.
    """

    selection_config = get_point_selection_config(
        config=config,
        point_kind=point_kind,
    )
    selection_method = selection_config.get("method", "physical")
    valid_methods = {
        "physical",
        "manual",
        "consensus",
        "union_physical_priority",
    }

    if selection_method not in valid_methods:
        raise ValueError(
            f"Unsupported {point_kind}-point selection method: "
            f"{selection_method}. Expected one of {sorted(valid_methods)}."
        )

    return selection_method


def get_point_selection_config(config, point_kind):
    """
    Return the point-selection config for one point kind.

    Parameters
    ----------
    config : dict
        Dataset configuration or ``points`` configuration.
    point_kind : {"x", "o"}
        Point kind whose selection config is read.

    Returns
    -------
    dict
        Selection configuration for the point kind.
    """

    points_config = (config or {}).get("points", config or {})

    if point_kind == "x":
        return points_config.get("x_selection", {})

    if point_kind == "o":
        return points_config.get("o_selection", {})

    raise ValueError(f"Unknown point kind: {point_kind}")


def get_manual_config_re(config, point_kind):
    """
    Return manual point-box half-widths in Earth radii.

    Parameters
    ----------
    config : dict
        Dataset configuration or ``points`` configuration.
    point_kind : {"x", "o"}
        Point kind whose manual box is read.

    Returns
    -------
    dict
        Manual box half-widths with keys ``x_half_width_re``,
        ``y_half_width_re``, and ``z_half_width_re``.
    """

    selection_config = get_point_selection_config(
        config=config,
        point_kind=point_kind,
    )
    box_config = selection_config.get("manual_re", {})
    required_keys = ("x_half_width_re", "z_half_width_re")
    missing_keys = [
        key
        for key in required_keys
        if key not in box_config
    ]

    if missing_keys:
        raise ValueError(
            f"points.{point_kind}_selection.manual_re is missing "
            f"required keys: {missing_keys}"
        )

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


def get_vdf_cellids_in_manual(config, point_record, vdf_cellids, vdf_coords_re):
    """
    Select VDF cells inside a fixed axis-aligned box around a point.

    Parameters
    ----------
    config : dict
        Dataset configuration.
    point_record : dict
        X-point or O-point record.
    vdf_cellids : numpy.ndarray
        Spatial cell IDs with VDF data.
    vdf_coords_re : numpy.ndarray
        VDF cell coordinates in Earth radii with shape ``(n_cells, 3)``.

    Returns
    -------
    dict
        Mapping from manual position name to VDF cell ID.
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
            np.abs(offsets_re[:, active_axes]) <= half_widths_re[active_axes],
            axis=1,
        )
    else:
        selected = np.ones(len(vdf_cellids), dtype=bool)

    return {
        f"{point_kind}_box_{index:04d}": int(cid)
        for index, cid in enumerate(vdf_cellids[selected])
    }


def get_vdf_cellids_in_hessian_di_box(config, point_record, vdf_cellids, vdf_coords_re):
    """
    Select VDF cells in a Hessian-aligned X-point box scaled by local ``d_i``.

    Parameters
    ----------
    config : dict
        Dataset configuration.
    point_record : dict
        X-point record containing Hessian eigenvectors and local ``d_i``.
    vdf_cellids : numpy.ndarray
        Spatial cell IDs with VDF data.
    vdf_coords_re : numpy.ndarray
        VDF cell coordinates in Earth radii with shape ``(n_cells, 3)``.

    Returns
    -------
    dict
        Mapping from X-point selection position name to VDF cell ID.
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


def get_vdf_cellids_in_flux_contour(config, point_record, vdf_cellids, vdf_coords_re):
    """
    Select VDF cells inside an O-point closed flux contour.

    Parameters
    ----------
    config : dict
        Dataset configuration.
    point_record : dict
        O-point record containing ``contour_vertices_re``.
    vdf_cellids : numpy.ndarray
        Spatial cell IDs with VDF data.
    vdf_coords_re : numpy.ndarray
        VDF cell coordinates in Earth radii with shape ``(n_cells, 3)``.

    Returns
    -------
    dict
        Mapping from island selection position name to VDF cell ID.
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


def create_point_sample_metadata(config, point_record):
    """
    Create metadata fields for a point-selected sample.

    Parameters
    ----------
    config : dict
        Dataset configuration.
    point_record : dict
        Detected X/O point record.

    Returns
    -------
    dict
        Metadata fields describing the source point and selection method.
    """

    point_kind = point_record["point_kind"]
    coord_re = point_record["coord_re"]
    selection_method = get_point_selection_method(
        config=config,
        point_kind=point_kind,
    )
    metadata = {
        "point_kind": point_kind,
        "selection_method": selection_method,
        "source_point_x_re": float(coord_re[0]),
        "source_point_y_re": float(coord_re[1]),
        "source_point_z_re": float(coord_re[2]),
        "source_point_flux": float(point_record["flux"]),
    }
    if "region_name" in point_record:
        metadata["region_name"] = point_record["region_name"]

    if selection_method in {"manual", "consensus", "union_physical_priority"}:
        box_config = get_manual_config_re(
            config=config,
            point_kind=point_kind,
        )
        metadata.update(
            {
                "selection_box_x_half_width_re": box_config["x_half_width_re"],
                "selection_box_y_half_width_re": box_config["y_half_width_re"],
                "selection_box_z_half_width_re": box_config["z_half_width_re"],
            }
        )

    if point_kind == "x":
        eigvecs = np.asarray(point_record["eigvecs"], dtype=float)
        metadata.update(
            {
                "rho": optional_float(point_record.get("rho")),
                "di_re": optional_float(point_record.get("di_re")),
                "hessian_e0_x": float(eigvecs[0, 0]),
                "hessian_e0_z": float(eigvecs[1, 0]),
                "hessian_e1_x": float(eigvecs[0, 1]),
                "hessian_e1_z": float(eigvecs[1, 1]),
            }
        )

    if point_kind == "o":
        metadata.update(
            {
                "boundary_flux": optional_float(point_record.get("boundary_flux")),
                "search_flux": optional_float(point_record.get("search_flux")),
                "core_fraction": optional_float(point_record.get("core_fraction")),
            }
        )

    return metadata


def optional_float(value):
    """
    Return a float value or ``nan`` for missing optional metadata.

    Parameters
    ----------
    value : object
        Value to convert.

    Returns
    -------
    float
        Converted value, or ``nan`` when the value is missing.
    """

    if value is None:
        return float("nan")

    return float(value)
