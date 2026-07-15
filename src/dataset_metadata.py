import csv
import os
import stat
import tempfile
from pathlib import Path

import numpy as np


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
    if vdf_coords_re.ndim != 2 or vdf_coords_re.shape[1:] != (3,):
        raise ValueError("vdf_coords_re must have shape (n_samples, 3)")
    if not np.all(np.isfinite(vdf_coords_re)):
        raise ValueError("vdf_coords_re must contain finite coordinates")

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
        if point_kinds.shape != (len(vdf_coords_re),):
            raise ValueError("point_kinds must have shape (n_samples,)")
        normalized_point_kinds = np.asarray(
            [
                "" if value is None else str(value).strip().lower()
                for value in point_kinds
            ],
            dtype=object,
        )
        invalid_point_kinds = ~np.isin(
            normalized_point_kinds,
            ("", "x", "o"),
        )
        if np.any(invalid_point_kinds):
            invalid_values = sorted(set(normalized_point_kinds[invalid_point_kinds]))
            raise ValueError(f"Unknown point kinds: {invalid_values}")

        if source_point_coords_re is None:
            if np.any(normalized_point_kinds != ""):
                raise ValueError(
                    "source_point_coords_re is required for X/O samples"
                )
        else:
            normalized_source_coords_re = np.asarray(
                source_point_coords_re,
                dtype=float,
            )
            if normalized_source_coords_re.shape != vdf_coords_re.shape:
                raise ValueError(
                    "source_point_coords_re must have shape (n_samples, 3)"
                )
            point_mask = normalized_point_kinds != ""
            if not np.all(np.isfinite(normalized_source_coords_re[point_mask])):
                raise ValueError(
                    "Source coordinates must be finite for X/O samples"
                )
    elif source_point_coords_re is not None:
        raise ValueError(
            "point_kinds is required when source_point_coords_re is provided"
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
    Return point coordinates as a finite two-dimensional array.

    Parameters
    ----------
    point_coords_re : array-like of float or None
        Point coordinates in Earth radii.
    argument_name : str
        Argument name used in validation errors.

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
    if point_coords_re.ndim != 2 or point_coords_re.shape[1:] != (3,):
        raise ValueError(f"{argument_name} must have shape (n_points, 3)")
    if not np.all(np.isfinite(point_coords_re)):
        raise ValueError(f"{argument_name} must contain finite coordinates")

    return point_coords_re


def _find_nearest_point_geometry(vdf_coords_re, point_coords_re):
    """
    Return closest-point distances and VDF-to-point vectors.

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
    if vdf_coord_re.shape != (3,) or not np.all(np.isfinite(vdf_coord_re)):
        raise ValueError("vdf_coord_re must contain three finite coordinates")

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
    if point_kind not in {"", "x", "o"}:
        raise ValueError(f"Unknown point_kind: {point_kind!r}")
    if not point_kind:
        return metadata
    if source_point_coord_re is None:
        raise ValueError(
            f"Source coordinates are required for {point_kind}-points"
        )

    source_point_coord_re = np.asarray(source_point_coord_re, dtype=float)
    if (
        source_point_coord_re.shape != (3,)
        or not np.all(np.isfinite(source_point_coord_re))
    ):
        raise ValueError(
            "source_point_coord_re must contain three finite coordinates"
        )
    vector_re = source_point_coord_re - vdf_coord_re
    metadata[f"distance_to_{point_kind}_point_re"] = float(
        np.linalg.norm(vector_re)
    )
    for component_name, component in zip(("dx", "dy", "dz"), vector_re):
        metadata[
            f"vdf_to_{point_kind}_point_{component_name}_re"
        ] = float(component)
    return metadata


def backfill_dataset_spatial_metadata(dataset_dir, config, n_jobs=4):
    """
    Backfill VDF coordinates and hybrid X/O reference geometry.

    Topology detection is rerun once per timestep from the configured bulk and
    flux files. All raw detected points are used for closest-point lookups,
    including points that did not produce a saved VDF sample. Existing X/O
    rows retain their source point for the same-kind reference. Required
    metadata values and their order are preserved. Unused metadata
    columns are removed, derived spatial columns are recalculated, and the
    legacy ``reconnection`` class name is migrated to ``x_point``. The original
    file is replaced only after all source data has been processed and the
    temporary CSV has been checked.

    Parameters
    ----------
    dataset_dir : str or pathlib.Path
        Dataset directory containing ``metadata.csv``.
    config : dict
        Dataset creation config containing bulk and flux path templates and
        the point-topology selection settings used for the dataset.
    n_jobs : int, optional
        Number of parallel timestep processes used for topology and VDF-cell
        coordinate reads.

    Returns
    -------
    pathlib.Path
        Updated metadata path.
    """

    metadata_path = Path(dataset_dir) / "metadata.csv"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Dataset metadata does not exist: {metadata_path}")

    with open(metadata_path, newline="") as metadata_file:
        reader = csv.DictReader(metadata_file)
        if reader.fieldnames is None:
            raise ValueError(f"Dataset metadata has no header: {metadata_path}")
        original_fieldnames = list(reader.fieldnames)
        if len(original_fieldnames) != len(set(original_fieldnames)):
            raise ValueError(
                f"Dataset metadata has duplicate columns: {metadata_path}"
            )
        fieldnames = [
            column
            for column in original_fieldnames
            if column not in OMITTED_DATASET_METADATA_COLUMNS
        ]
        rows = list(reader)
    for row in rows:
        for column in OMITTED_DATASET_METADATA_COLUMNS:
            row.pop(column, None)

    if not isinstance(config, dict):
        raise TypeError("config must be a dataset creation config dictionary")
    required_config_keys = {
        "file_template_bulk",
        "file_template_flux",
        "points",
    }
    missing_config_keys = required_config_keys - set(config)
    if missing_config_keys:
        raise ValueError(
            f"Dataset config is missing keys: {sorted(missing_config_keys)}"
        )
    if not isinstance(config["points"], dict):
        raise TypeError("config.points must be a dictionary")

    from src.timesteps import create_timestep_path

    required_columns = {"timestep", "cid", "class_name", "file_location"}
    missing_columns = required_columns - set(fieldnames)
    if missing_columns:
        raise ValueError(
            f"Dataset metadata is missing columns: {sorted(missing_columns)}"
        )

    source_columns = (
        "source_point_x_re",
        "source_point_y_re",
        "source_point_z_re",
    )
    present_source_columns = set(source_columns) & set(fieldnames)
    if present_source_columns and present_source_columns != set(source_columns):
        raise ValueError(
            "Point source metadata must contain all of "
            "source_point_x_re, source_point_y_re, and source_point_z_re"
        )
    if "point_kind" not in fieldnames and present_source_columns:
        raise ValueError("Point metadata requires a point_kind column")
    has_source_columns = set(source_columns).issubset(fieldnames)
    row_keys = []
    row_indices_by_timestep = {}
    cellids_by_timestep = {}
    for row_number, row in enumerate(rows, start=2):
        class_name = str(row.get("class_name", "")).strip()
        if class_name == "reconnection":
            row["class_name"] = "x_point"
            class_name = "x_point"

        file_location = str(row["file_location"]).strip()
        if file_location == "":
            raise ValueError(f"Metadata row {row_number} has no file_location")
        try:
            timestep = int(row["timestep"])
            cid = int(row["cid"])
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"Metadata row {row_number} has invalid timestep or cid"
            ) from error

        configured_bulk_path = create_timestep_path(
            path_template=config["file_template_bulk"],
            timestep=timestep,
        )
        if Path(file_location) != Path(configured_bulk_path):
            raise ValueError(
                f"Metadata row {row_number} bulk file does not match the "
                f"configured template: {file_location!r} != "
                f"{str(configured_bulk_path)!r}"
            )

        point_kind = str(row.get("point_kind", "")).strip().lower()
        if point_kind not in {"", "x", "o"}:
            raise ValueError(
                f"Metadata row {row_number} has invalid point_kind "
                f"{point_kind!r}"
            )
        expected_point_kind = {
            "x_point": "x",
            "o_point": "o",
        }.get(class_name)
        if expected_point_kind is not None and point_kind != expected_point_kind:
            raise ValueError(
                f"Metadata row {row_number} class {class_name!r} requires "
                f"point_kind={expected_point_kind!r} and source coordinates"
            )
        if point_kind and expected_point_kind is None:
            raise ValueError(
                f"Metadata row {row_number} has point_kind={point_kind!r} "
                f"but class {class_name!r} is not an X/O-point class"
            )
        if point_kind and not has_source_columns:
            raise ValueError(
                "X/O-point rows require source_point_x_re, "
                "source_point_y_re, and source_point_z_re"
            )
        source_point_coord_re = None
        if point_kind:
            try:
                source_point_coord_re = tuple(
                    float(row[column]) for column in source_columns
                )
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    f"Metadata row {row_number} has invalid source coordinates"
                ) from error
            if not np.all(np.isfinite(source_point_coord_re)):
                raise ValueError(
                    f"Metadata row {row_number} has invalid source coordinates"
                )

        row_keys.append((timestep, cid, point_kind, source_point_coord_re))
        row_index = len(row_keys) - 1
        row_indices_by_timestep.setdefault(timestep, []).append(row_index)
        cellids_by_timestep.setdefault(timestep, set()).add(cid)

    missing_files = []
    for timestep in sorted(cellids_by_timestep):
        for path_template in (
            config["file_template_bulk"],
            config["file_template_flux"],
        ):
            file_path = create_timestep_path(
                path_template=path_template,
                timestep=timestep,
            )
            if not file_path.is_file():
                missing_files.append(str(file_path))
    if missing_files:
        raise FileNotFoundError(
            f"Configured bulk/flux files do not exist: {missing_files[:5]}"
        )

    n_jobs = int(n_jobs)
    if n_jobs == 0:
        raise ValueError("n_jobs must be non-zero")

    from joblib import Parallel, delayed

    print(
        f"Recomputing topology for {len(cellids_by_timestep)} timesteps "
        f"with n_jobs={n_jobs}"
    )
    timestep_results = Parallel(n_jobs=n_jobs, prefer="processes")(
        delayed(_read_timestep_spatial_data)(
            timestep=timestep,
            cellids=cellids,
            config=config,
        )
        for timestep, cellids in cellids_by_timestep.items()
    )

    for (
        timestep,
        cellids,
        coords_re,
        x_point_coords_re,
        o_point_coords_re,
    ) in timestep_results:
        coord_by_cellid = {
            int(cid): coord_re
            for cid, coord_re in zip(cellids, coords_re)
        }
        row_indices = row_indices_by_timestep[timestep]
        vdf_coords_re = np.asarray(
            [coord_by_cellid[row_keys[index][1]] for index in row_indices],
            dtype=float,
        )
        point_kinds = np.asarray(
            [row_keys[index][2] for index in row_indices],
            dtype=object,
        )
        source_point_coords_re = np.full(vdf_coords_re.shape, np.nan, dtype=float)
        for local_index, row_index in enumerate(row_indices):
            source_point_coord_re = row_keys[row_index][3]
            if source_point_coord_re is not None:
                source_point_coords_re[local_index] = source_point_coord_re

        reference_metadata = create_point_reference_metadata_arrays(
            vdf_coords_re=vdf_coords_re,
            x_point_coords_re=x_point_coords_re,
            o_point_coords_re=o_point_coords_re,
            point_kinds=point_kinds,
            source_point_coords_re=source_point_coords_re,
        )

        for local_index, row_index in enumerate(row_indices):
            row = rows[row_index]
            coord_re = vdf_coords_re[local_index]
            row["vdf_x_re"] = repr(float(coord_re[0]))
            row["vdf_y_re"] = repr(float(coord_re[1]))
            row["vdf_z_re"] = repr(float(coord_re[2]))
            for column, values in reference_metadata.items():
                row[column] = _format_metadata_value(values[local_index])

    output_fieldnames = fieldnames + [
        column
        for column in SPATIAL_METADATA_COLUMNS
        if column not in fieldnames
    ]

    original_mode = stat.S_IMODE(metadata_path.stat().st_mode)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            newline="",
            dir=metadata_path.parent,
            prefix=f".{metadata_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temp_file:
            temp_path = Path(temp_file.name)
            writer = csv.DictWriter(
                temp_file,
                fieldnames=output_fieldnames,
                extrasaction="raise",
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(rows)
            temp_file.flush()
            os.fsync(temp_file.fileno())

        with open(temp_path, newline="") as temp_file:
            reader = csv.DictReader(temp_file)
            written_fieldnames = list(reader.fieldnames or [])
            written_rows = list(reader)
        if written_fieldnames != output_fieldnames or written_rows != rows:
            raise RuntimeError("Temporary metadata validation failed")

        os.chmod(temp_path, original_mode)
        os.replace(temp_path, metadata_path)
    except Exception:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
        raise

    return metadata_path


def _read_timestep_spatial_data(timestep, cellids, config):
    """
    Read VDF centers and rerun raw point detection for one timestep.

    Parameters
    ----------
    timestep : int
        Dataset timestep.
    cellids : collection of int
        Spatial cell IDs to read.
    config : dict
        Dataset creation config.

    Returns
    -------
    timestep : int
        Dataset timestep.
    cellids : numpy.ndarray
        Sorted spatial cell IDs.
    coords_re : numpy.ndarray
        Corresponding cell-center coordinates in Earth radii.
    x_point_coords_re : list of list of float
        All raw detected X-point coordinates.
    o_point_coords_re : list of list of float
        All raw detected O-point coordinates.
    """

    import analysator as pt

    from src.point_topology import find_point_records
    from src.timesteps import create_timestep_path
    from src.vdf_helpers import R_EARTH

    bulk_file_path = create_timestep_path(
        path_template=config["file_template_bulk"],
        timestep=timestep,
    )
    flux_file_path = create_timestep_path(
        path_template=config["file_template_flux"],
        timestep=timestep,
    )
    cellids = np.asarray(sorted(cellids), dtype=int)
    reader = pt.vlsvfile.VlsvReader(str(bulk_file_path))
    try:
        coords = np.asarray(reader.get_cell_coordinates(cellids), dtype=float)
        if coords.ndim == 1 and len(cellids) == 1:
            coords = coords.reshape(1, 3)
        if coords.shape != (len(cellids), 3):
            raise ValueError("Vector coordinate lookup returned invalid shape")
    except Exception:
        coords = np.asarray(
            [reader.get_cell_coordinates(int(cid)) for cid in cellids],
            dtype=float,
        )

    coords_re = coords / R_EARTH
    if coords_re.shape != (len(cellids), 3) or not np.all(
        np.isfinite(coords_re)
    ):
        raise ValueError(f"Invalid cell coordinates read from {bulk_file_path}")

    x_point_records, o_point_records = find_point_records(
        reader=reader,
        flux_file_location=flux_file_path,
        points_config=config["points"],
    )

    return (
        int(timestep),
        cellids,
        coords_re,
        [point_record["coord_re"] for point_record in x_point_records],
        [point_record["coord_re"] for point_record in o_point_records],
    )


def _format_metadata_value(value):
    """
    Format one derived metadata value for CSV output.

    Parameters
    ----------
    value : float
        Derived metadata value.

    Returns
    -------
    str
        CSV representation. Missing numeric values become empty strings.
    """

    value = float(value)
    return "" if np.isnan(value) else repr(value)
