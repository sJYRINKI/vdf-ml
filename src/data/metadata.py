"""Create sample-aligned spatial and point-topology metadata.

All coordinates and X/O displacement components use Earth radii. Metadata
rows preserve the first-axis order of ``X.npy`` and optional
``X_hermite.npy``.
"""

import numpy as np
import pandas as pd

from src.data.metadata_columns import (
    METADATA_COLUMNS,
    TOPOLOGY_TARGET_GROUPS,
)


POINT_REFERENCE_METADATA_COLUMNS = (
    "distance_to_x_point_re",
    "distance_to_o_point_re",
    "vdf_to_x_point_dx_re",
    "vdf_to_x_point_dy_re",
    "vdf_to_x_point_dz_re",
    "vdf_to_o_point_dx_re",
    "vdf_to_o_point_dy_re",
    "vdf_to_o_point_dz_re",
)

SPATIAL_METADATA_COLUMNS = (
    "vdf_x_re",
    "vdf_y_re",
    "vdf_z_re",
    *POINT_REFERENCE_METADATA_COLUMNS,
)

OMITTED_DATASET_METADATA_COLUMNS = (
    "simulation_time",
    "neighbor_position",
    "region_name",
    "selection_box_y_half_width_re",
    "rho",
    "core_fraction",
    "selection_agreement",
    "x_point_available",
    "o_point_available",
)


def create_extraction_metadata(metadata):
    """Arrange extracted rows in the fixed current metadata order.

    Dataset creation calls this after every staged array row has been written.
    It derives the X/O topology-availability masks used by CNN loss masking,
    then places identity, class, coordinate, availability, and six topology
    target columns first without changing row order.

    Parameters
    ----------
    metadata : list of dict
        Extraction rows already aligned with staged array rows.

    Returns
    -------
    pandas.DataFrame
        Metadata with topology availability fields and fixed columns first;
        additional extraction fields retain their first-seen order.
    """

    metadata_frame = pd.DataFrame(metadata)
    source_columns = list(metadata_frame.columns)
    for point_kind, target_columns in TOPOLOGY_TARGET_GROUPS.items():
        finite_targets = np.column_stack(
            [
                np.isfinite(
                    pd.to_numeric(
                        metadata_frame[column],
                        errors="coerce",
                    ).to_numpy(dtype=np.float64, na_value=np.nan)
                )
                for column in target_columns
            ]
        )
        availability_column = f"{point_kind}_topology_available"
        if availability_column not in metadata_frame:
            metadata_frame[availability_column] = np.all(
                finite_targets,
                axis=1,
            )
    ordered_columns = [
        column for column in METADATA_COLUMNS if column in metadata_frame
    ]
    ordered_columns.extend(
        column
        for column in source_columns
        if column not in ordered_columns
    )
    ordered_columns.extend(
        column
        for column in metadata_frame.columns
        if column not in ordered_columns
    )
    return metadata_frame.loc[:, ordered_columns]


def create_point_reference_metadata_arrays(
    vdf_coords_re,
    x_point_coords_re=None,
    o_point_coords_re=None,
    point_kinds=None,
    source_point_coords_re=None,
):
    """
    Calculate X/O-point distances and vectors for VDF-cell centers.

    Without point kinds and source coordinates, every VDF uses its closest
    detected X point and closest detected O point. When per-sample provenance
    is provided, X-selected samples use their source X point and O-selected
    samples use their source O point. The opposite point kind remains the
    closest detected point. Vectors point from each VDF center to its reference
    point and are expressed in Earth radii.

    The nearest-point calculation loops over point coordinates and keeps only
    one best distance/vector per VDF. It does not allocate an
    ``n_samples x n_points x 3`` array.

    Parameters
    ----------
    vdf_coords_re : array-like of float
        VDF-cell-center coordinates with shape ``(n_samples, 3)``.
    x_point_coords_re : array-like of float, optional
        Raw detected X-point coordinates with shape ``(n_x_points, 3)``.
    o_point_coords_re : array-like of float, optional
        Raw detected O-point coordinates with shape ``(n_o_points, 3)``.
    point_kinds : array-like of str, optional
        Per-sample point kind: ``"x"``, ``"o"``, or an empty value for a
        non-point sample.
    source_point_coords_re : array-like of float, optional
        Per-sample source-point coordinates with shape ``(n_samples, 3)``.
        Coordinates must be finite for samples whose point kind is X or O.

    Returns
    -------
    dict[str, numpy.ndarray]
        Column names mapped to one-dimensional arrays in sample order.
        Missing point kinds contain ``NaN`` distances and vector components.
    """

    vdf_coords_re = np.asarray(vdf_coords_re, dtype=float)

    x_point_coords_re = _normalize_point_coords_re(
        x_point_coords_re,
        argument_name="x_point_coords_re",
    )
    o_point_coords_re = _normalize_point_coords_re(
        o_point_coords_re,
        argument_name="o_point_coords_re",
    )

    normalized_point_kinds = None
    normalized_source_coords_re = None
    if point_kinds is not None:
        point_kinds = np.asarray(point_kinds, dtype=object)
        normalized_point_kinds = np.asarray(
            [
                "" if value is None else str(value).strip().lower()
                for value in point_kinds
            ],
            dtype=object,
        )
        if source_point_coords_re is not None:
            normalized_source_coords_re = np.asarray(
                source_point_coords_re,
                dtype=float,
            )

    distances_by_kind = {}
    vectors_by_kind = {}
    for point_kind, point_coords_re in (
        ("x", x_point_coords_re),
        ("o", o_point_coords_re),
    ):
        distances, vectors = _find_nearest_point_geometry(
            vdf_coords_re=vdf_coords_re,
            point_coords_re=point_coords_re,
        )

        if normalized_point_kinds is not None:
            source_mask = normalized_point_kinds == point_kind
            if np.any(source_mask):
                source_vectors = (
                    normalized_source_coords_re[source_mask]
                    - vdf_coords_re[source_mask]
                )
                vectors[source_mask] = source_vectors
                distances[source_mask] = np.linalg.norm(
                    source_vectors,
                    axis=1,
                )

        distances_by_kind[point_kind] = distances
        vectors_by_kind[point_kind] = vectors

    return {
        "distance_to_x_point_re": distances_by_kind["x"],
        "distance_to_o_point_re": distances_by_kind["o"],
        "vdf_to_x_point_dx_re": vectors_by_kind["x"][:, 0],
        "vdf_to_x_point_dy_re": vectors_by_kind["x"][:, 1],
        "vdf_to_x_point_dz_re": vectors_by_kind["x"][:, 2],
        "vdf_to_o_point_dx_re": vectors_by_kind["o"][:, 0],
        "vdf_to_o_point_dy_re": vectors_by_kind["o"][:, 1],
        "vdf_to_o_point_dz_re": vectors_by_kind["o"][:, 2],
    }


def _normalize_point_coords_re(point_coords_re, argument_name):
    """
    Return point coordinates as a two-dimensional array.

    The nearest-reference calculation accepts no points, one point, or a
    point matrix. This conversion gives those branches one stable
    ``(n_points, 3)`` Earth-radius representation without altering values.

    Parameters
    ----------
    point_coords_re : array-like of float or None
        Point coordinates in Earth radii.
    argument_name : str
        Descriptive argument name retained for call-site readability.

    Returns
    -------
    numpy.ndarray
        Point coordinates with shape ``(n_points, 3)``.
    """

    if point_coords_re is None:
        return np.empty((0, 3), dtype=float)

    point_coords_re = np.asarray(point_coords_re, dtype=float)
    if point_coords_re.size == 0:
        return np.empty((0, 3), dtype=float)
    if point_coords_re.shape == (3,):
        point_coords_re = point_coords_re.reshape(1, 3)
    return point_coords_re


def _find_nearest_point_geometry(vdf_coords_re, point_coords_re):
    """
    Return closest-point distances and VDF-to-point vectors.

    Metadata construction calls this independently for X and O references.
    Iterating over points keeps memory proportional to the sample count rather
    than allocating a three-dimensional sample-by-point displacement array.

    Parameters
    ----------
    vdf_coords_re : numpy.ndarray
        VDF-cell-center coordinates in Earth radii with shape
        ``(n_samples, 3)``.
    point_coords_re : numpy.ndarray
        Point coordinates in Earth radii with shape ``(n_points, 3)``.

    Returns
    -------
    distances : numpy.ndarray
        Closest-point distances in Earth radii.
    vectors : numpy.ndarray
        Vectors from VDF centers to closest points in Earth radii.
    """

    n_samples = len(vdf_coords_re)
    best_distance_squared = np.full(n_samples, np.inf, dtype=float)
    best_vectors = np.full((n_samples, 3), np.nan, dtype=float)

    for point_coord_re in point_coords_re:
        vectors = point_coord_re - vdf_coords_re
        distance_squared = np.einsum("ij,ij->i", vectors, vectors)
        closer = distance_squared < best_distance_squared
        best_distance_squared[closer] = distance_squared[closer]
        best_vectors[closer] = vectors[closer]

    distances = np.full(n_samples, np.nan, dtype=float)
    has_reference = np.isfinite(best_distance_squared)
    distances[has_reference] = np.sqrt(best_distance_squared[has_reference])
    return distances, best_vectors


def create_vdf_spatial_metadata(
    vdf_coord_re,
    point_kind=None,
    source_point_coord_re=None,
):
    """
    Create VDF-cell coordinates and optional source-point geometry.

    One metadata row uses this focused calculation when precomputed nearest
    references are unavailable. Non-point samples receive ``NaN`` topology
    fields; point-selected samples receive only their same-kind source vector.

    Parameters
    ----------
    vdf_coord_re : array-like of float
        VDF-cell-center coordinate in Earth radii.
    point_kind : {"x", "o"}, optional
        Source point kind for a point-selected VDF.
    source_point_coord_re : array-like of float, optional
        Source X- or O-point coordinate in Earth radii.

    Returns
    -------
    dict
        VDF-cell-center fields and sparse X/O distance/vector fields.
    """

    vdf_coord_re = np.asarray(vdf_coord_re, dtype=float)
    metadata = {
        "vdf_x_re": float(vdf_coord_re[0]),
        "vdf_y_re": float(vdf_coord_re[1]),
        "vdf_z_re": float(vdf_coord_re[2]),
        **{
            column: float("nan")
            for column in POINT_REFERENCE_METADATA_COLUMNS
        },
    }

    point_kind = "" if point_kind is None else str(point_kind).strip().lower()
    if not point_kind:
        return metadata

    source_point_coord_re = np.asarray(source_point_coord_re, dtype=float)
    vector_re = source_point_coord_re - vdf_coord_re
    metadata[f"distance_to_{point_kind}_point_re"] = float(
        np.linalg.norm(vector_re)
    )
    for component_name, component in zip(("dx", "dy", "dz"), vector_re):
        metadata[
            f"vdf_to_{point_kind}_point_{component_name}_re"
        ] = float(component)
    return metadata


def add_point_reference_metadata(
    sample_specs,
    raw_x_point_records=None,
    raw_o_point_records=None,
):
    """Attach source-or-nearest X/O geometry to planned sample records.

    Sample planning calls this after contradictory class assignments and
    background selection have been resolved. The function mutates each input
    record in place so the following extraction stage can write aligned
    topology fields without recomputing point geometry.

    Parameters
    ----------
    sample_specs : list of dict
        Final sample specifications for one timestep, each with a selected
        VDF-cell centre in Earth radii.
    raw_x_point_records : iterable of dict, optional
        Raw X-point records returned by topology detection.
    raw_o_point_records : iterable of dict, optional
        Raw O-point records returned by topology detection.

    Returns
    -------
    list of dict
        The input records with distances and VDF-to-point vectors added in
        Earth radii.

    Notes
    -----
    X-selected samples use their source X point, and O-selected samples use
    their source O point. Other references use the nearest raw point of that
    kind. Vectors point from the selected VDF-cell centre to the reference.
    """

    if not sample_specs:
        return sample_specs

    raw_x_point_records = list(raw_x_point_records or [])
    raw_o_point_records = list(raw_o_point_records or [])
    vdf_coords_re = np.asarray(
        [sample_spec["vdf_coord_re"] for sample_spec in sample_specs],
        dtype=float,
    )
    point_kinds = np.empty(len(sample_specs), dtype=object)
    source_point_coords_re = np.full(vdf_coords_re.shape, np.nan, dtype=float)
    for index, sample_spec in enumerate(sample_specs):
        point_kind = sample_spec.get("point_kind")
        point_kind = "" if point_kind is None else str(point_kind).strip().lower()
        point_kinds[index] = point_kind
        if point_kind in {"x", "o"}:
            source_point_coords_re[index] = (
                sample_spec["source_point_x_re"],
                sample_spec["source_point_y_re"],
                sample_spec["source_point_z_re"],
            )

    reference_metadata = create_point_reference_metadata_arrays(
        vdf_coords_re=vdf_coords_re,
        x_point_coords_re=[
            point_record["coord_re"]
            for point_record in raw_x_point_records
        ],
        o_point_coords_re=[
            point_record["coord_re"]
            for point_record in raw_o_point_records
        ],
        point_kinds=point_kinds,
        source_point_coords_re=source_point_coords_re,
    )
    for column, values in reference_metadata.items():
        for sample_spec, value in zip(sample_specs, values):
            sample_spec[column] = float(value)
    return sample_specs


def create_sample_metadata_row(sample_spec, cid, file_location):
    """Create one saved metadata row for an extracted physical VDF.

    Stage 4 calls this immediately after extracting a planned sample. It
    converts planning-only fields into the persistent row aligned with the
    same ``X.npy`` index while retaining source-file location for plotting.

    Parameters
    ----------
    sample_spec : dict
        Planned sample containing class identity, selected VDF-cell centre,
        and optional X/O source geometry.
    cid : int
        Spatial cell identifier actually used for VDF extraction.
    file_location : str or pathlib.Path
        Source VLSV path retained for plotting and traceability.

    Returns
    -------
    dict
        Current metadata fields for one array row, including selected
        coordinates and X/O vectors in Earth radii.
    """

    metadata_row = {
        "timestep": int(sample_spec["timestep"]),
        "cid": int(cid),
        "class_id": int(sample_spec["label"]),
        "class_name": sample_spec["class_name"],
        "file_location": str(file_location),
    }
    point_kind = sample_spec.get("point_kind")
    if point_kind is not None:
        point_kind = str(point_kind).strip().lower()
    source_point_coord_re = None
    if point_kind in {"x", "o"}:
        source_point_coord_re = (
            sample_spec["source_point_x_re"],
            sample_spec["source_point_y_re"],
            sample_spec["source_point_z_re"],
        )
    if all(
        column in sample_spec
        for column in POINT_REFERENCE_METADATA_COLUMNS
    ):
        vdf_coord_re = sample_spec["vdf_coord_re"]
        spatial_metadata = {
            "vdf_x_re": float(vdf_coord_re[0]),
            "vdf_y_re": float(vdf_coord_re[1]),
            "vdf_z_re": float(vdf_coord_re[2]),
            **{
                column: sample_spec[column]
                for column in POINT_REFERENCE_METADATA_COLUMNS
            },
        }
    else:
        spatial_metadata = create_vdf_spatial_metadata(
            vdf_coord_re=sample_spec["vdf_coord_re"],
            point_kind=point_kind,
            source_point_coord_re=source_point_coord_re,
        )
    internal_keys = {
        "file_location",
        "simulation_time",
        "cid",
        "label",
        "class_name",
        "coord_re",
        "vdf_coord_re",
        "neighbor_position",
        "timestep",
        "velocity_grid",
        *OMITTED_DATASET_METADATA_COLUMNS,
        *POINT_REFERENCE_METADATA_COLUMNS,
    }
    for key, value in sample_spec.items():
        if key not in internal_keys:
            metadata_row[key] = value
    metadata_row.update(spatial_metadata)
    return metadata_row
