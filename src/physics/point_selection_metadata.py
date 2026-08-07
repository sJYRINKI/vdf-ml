"""Create saved metadata describing X/O sample-selection geometry.

Stage-3 planning calls this after a critical point's selection method is
known. The returned scalar fields preserve source-point coordinates, flux,
manual box dimensions, and kind-specific Hessian or island information
without becoming representation or model inputs.
"""

import numpy as np

from src.physics.point_selection_geometry import get_manual_config_re


def create_point_sample_metadata(
    config,
    point_record,
    selection_method,
):
    """Create source-point and selection fields for one planned sample.

    Every VDF cell selected around one critical point receives this shared
    source geometry before position-specific agreement metadata is added.
    Coordinates and box widths use Earth radii; Hessian eigenvectors are
    dimensionless and flux values retain source-file units.

    Parameters
    ----------
    config : dict
        Dataset point-selection configuration.
    point_record : dict
        Detected X/O point with coordinates in Earth radii, interpolated flux,
        and kind-specific physical geometry.
    selection_method : str
        Resolved selection method used for this point kind.

    Returns
    -------
    dict
        Source coordinates/flux and physical or manual selection metadata.
    """

    point_kind = point_record["point_kind"]
    coord_re = point_record["coord_re"]
    physical_selection = selection_method in {
        "physical",
        "consensus",
        "union",
    }
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

    if selection_method in {
        "manual",
        "consensus",
        "union",
    }:
        box_config = get_manual_config_re(
            config=config,
            point_kind=point_kind,
        )
        metadata.update(
            {
                "selection_box_x_half_width_re": (
                    box_config["x_half_width_re"]
                ),
                "selection_box_y_half_width_re": (
                    box_config["y_half_width_re"]
                ),
                "selection_box_z_half_width_re": (
                    box_config["z_half_width_re"]
                ),
            }
        )

    if point_kind == "x":
        eigvecs = np.asarray(point_record["eigvecs"], dtype=float)
        metadata.update(
            {
                "rho": optional_float(
                    point_record.get("rho")
                    if physical_selection
                    else None
                ),
                "di_re": optional_float(
                    point_record.get("di_re")
                    if physical_selection
                    else None
                ),
                "hessian_e0_x": float(eigvecs[0, 0]),
                "hessian_e0_z": float(eigvecs[1, 0]),
                "hessian_e1_x": float(eigvecs[0, 1]),
                "hessian_e1_z": float(eigvecs[1, 1]),
            }
        )
    if point_kind == "o":
        metadata.update(
            {
                "boundary_flux": optional_float(
                    point_record.get("boundary_flux")
                    if physical_selection
                    else None
                ),
                "search_flux": optional_float(
                    point_record.get("search_flux")
                    if physical_selection
                    else None
                ),
                "core_fraction": optional_float(
                    point_record.get("core_fraction")
                    if physical_selection
                    else None
                ),
            }
        )
    return metadata


def optional_float(value):
    """Convert optional numeric metadata to float or ``NaN``.

    Point geometry uses ``NaN`` to preserve a numeric metadata column when a
    kind-specific quantity, such as an island boundary or ion inertial length,
    is absent for one row.

    Parameters
    ----------
    value : object
        Optional value supplied by point geometry.

    Returns
    -------
    float
        Converted value, or ``NaN`` when absent.
    """

    if value is None:
        return float("nan")
    return float(value)
