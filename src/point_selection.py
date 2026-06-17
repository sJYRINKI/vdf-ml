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

    if len(vdf_cellids) == 0:
        return {}

    x_selection = config.get("points", {}).get("x_selection", {})
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
        & (np.abs(offsets_re[:, 1]) <= y_half_width_re)
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

    o_selection = config.get("points", {}).get("o_selection", {})
    y_half_width_re = float(o_selection.get("y_half_width_re", 0.0))
    center_re = np.asarray(point_record["coord_re"], dtype=float)
    offsets_re = vdf_coords_re - center_re

    contour_path = MplPath(np.asarray(contour_vertices_re, dtype=float))
    selected = (
        contour_path.contains_points(vdf_coords_re[:, [0, 2]])
        & (np.abs(offsets_re[:, 1]) <= y_half_width_re)
    )

    return {
        f"o_island_{index:04d}": int(cid)
        for index, cid in enumerate(vdf_cellids[selected])
    }


def create_point_sample_metadata(point_record):
    """
    Create metadata fields for a point-selected sample.

    Parameters
    ----------
    point_record : dict
        Detected X/O point record.

    Returns
    -------
    dict
        Metadata fields describing the source point and selection method.
    """

    coord_re = point_record["coord_re"]
    metadata = {
        "point_kind": point_record["point_kind"],
        "source_point_x_re": float(coord_re[0]),
        "source_point_y_re": float(coord_re[1]),
        "source_point_z_re": float(coord_re[2]),
        "source_point_flux": float(point_record["flux"]),
    }

    if point_record["point_kind"] == "x":
        eigvecs = np.asarray(point_record["eigvecs"], dtype=float)
        metadata.update(
            {
                "rho": float(point_record["rho"]),
                "di_re": float(point_record["di_re"]),
                "hessian_e0_x": float(eigvecs[0, 0]),
                "hessian_e0_z": float(eigvecs[1, 0]),
                "hessian_e1_x": float(eigvecs[0, 1]),
                "hessian_e1_z": float(eigvecs[1, 1]),
            }
        )

    if point_record["point_kind"] == "o":
        metadata.update(
            {
                "boundary_flux": float(point_record["boundary_flux"]),
                "search_flux": float(point_record["search_flux"]),
                "core_fraction": float(point_record["core_fraction"]),
            }
        )

    return metadata


