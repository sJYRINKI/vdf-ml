import csv
import os
import stat
import tempfile
from pathlib import Path

import numpy as np


SPATIAL_METADATA_COLUMNS = (
    "vdf_x_re",
    "vdf_y_re",
    "vdf_z_re",
    "distance_to_x_point_re",
    "distance_to_o_point_re",
)


def create_vdf_spatial_metadata(
    vdf_coord_re,
    point_kind=None,
    source_point_coord_re=None,
):
    """
    Create VDF-cell coordinates and optional point-distance metadata.

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
        VDF-cell-center fields and sparse X/O distance fields.
    """

    vdf_coord_re = np.asarray(vdf_coord_re, dtype=float)
    if vdf_coord_re.shape != (3,) or not np.all(np.isfinite(vdf_coord_re)):
        raise ValueError("vdf_coord_re must contain three finite coordinates")

    metadata = {
        "vdf_x_re": float(vdf_coord_re[0]),
        "vdf_y_re": float(vdf_coord_re[1]),
        "vdf_z_re": float(vdf_coord_re[2]),
        "distance_to_x_point_re": float("nan"),
        "distance_to_o_point_re": float("nan"),
    }

    point_kind = "" if point_kind is None else str(point_kind).strip().lower()
    if point_kind not in {"", "x", "o"}:
        raise ValueError(f"Unknown point_kind: {point_kind!r}")
    if point_kind == "":
        return metadata
    if source_point_coord_re is None:
        raise ValueError(f"Source coordinates are required for {point_kind}-points")

    source_point_coord_re = np.asarray(source_point_coord_re, dtype=float)
    if (
        source_point_coord_re.shape != (3,)
        or not np.all(np.isfinite(source_point_coord_re))
    ):
        raise ValueError(
            "source_point_coord_re must contain three finite coordinates"
        )

    metadata[f"distance_to_{point_kind}_point_re"] = float(
        np.linalg.norm(vdf_coord_re - source_point_coord_re)
    )
    return metadata


def backfill_dataset_spatial_metadata(dataset_dir, n_jobs=4):
    """
    Add VDF-cell coordinates and X/O-point distances to metadata.csv.

    Existing metadata values and column order are preserved. The five derived
    columns are appended when missing, and the original file is replaced only
    after all source coordinates have been read and calculated successfully.

    Parameters
    ----------
    dataset_dir : str or pathlib.Path
        Dataset directory containing ``metadata.csv``.
    n_jobs : int, optional
        Number of parallel processes used to read source VLSV files.

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
        fieldnames = list(reader.fieldnames)
        if len(fieldnames) != len(set(fieldnames)):
            raise ValueError(f"Dataset metadata has duplicate columns: {metadata_path}")
        rows = list(reader)

    required_columns = {
        "cid",
        "file_location",
        "x_re",
        "y_re",
        "z_re",
    }
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
    if "point_kind" not in fieldnames and any(
        column in fieldnames for column in source_columns
    ):
        raise ValueError("Point metadata requires a point_kind column")
    has_source_columns = set(source_columns).issubset(fieldnames)
    row_keys = []
    cellids_by_file = {}
    for row_number, row in enumerate(rows, start=2):
        file_location = str(row["file_location"]).strip()
        if file_location == "":
            raise ValueError(f"Metadata row {row_number} has no file_location")
        try:
            cid = int(row["cid"])
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"Metadata row {row_number} has invalid cid {row['cid']!r}"
            ) from error

        point_kind = str(row.get("point_kind", "")).strip().lower()
        if point_kind not in {"", "x", "o"}:
            raise ValueError(
                f"Metadata row {row_number} has invalid point_kind "
                f"{point_kind!r}"
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

        row_keys.append(
            (file_location, cid, point_kind, source_point_coord_re)
        )
        cellids_by_file.setdefault(file_location, set()).add(cid)

    missing_files = [
        file_location
        for file_location in cellids_by_file
        if not Path(file_location).is_file()
    ]
    if missing_files:
        raise FileNotFoundError(
            f"Source VLSV files do not exist: {missing_files[:5]}"
        )

    n_jobs = int(n_jobs)
    if n_jobs == 0:
        raise ValueError("n_jobs must be non-zero")

    from joblib import Parallel, delayed

    print(f"Reading {len(cellids_by_file)} VLSV files with n_jobs={n_jobs}")
    file_results = Parallel(n_jobs=n_jobs, prefer="processes")(
        delayed(_read_file_vdf_coords_re)(file_location, cellids)
        for file_location, cellids in cellids_by_file.items()
    )
    coord_by_file_and_cellid = {}
    for file_location, cellids, coords_re in file_results:
        for cid, coord_re in zip(cellids, coords_re):
            coord_by_file_and_cellid[(file_location, int(cid))] = tuple(
                float(value) for value in coord_re
            )

    for row, row_key in zip(rows, row_keys):
        file_location, cid, point_kind, source_point_coord_re = row_key

        spatial_metadata = create_vdf_spatial_metadata(
            vdf_coord_re=coord_by_file_and_cellid[(file_location, cid)],
            point_kind=point_kind,
            source_point_coord_re=source_point_coord_re,
        )
        row.update(
            {
                column: "" if np.isnan(value) else repr(float(value))
                for column, value in spatial_metadata.items()
            }
        )

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


def _read_file_vdf_coords_re(file_location, cellids):
    """
    Read selected VDF-cell centers from one source file.

    Parameters
    ----------
    file_location : str
        Source VLSV filepath.
    cellids : collection of int
        Spatial cell IDs to read.

    Returns
    -------
    file_location : str
        Source VLSV filepath used as part of the metadata lookup key.
    cellids : numpy.ndarray
        Sorted spatial cell IDs.
    coords_re : numpy.ndarray
        Corresponding cell-center coordinates in Earth radii.
    """

    import analysator as pt

    from src.vdf_helpers import R_EARTH

    file_path = Path(file_location)
    cellids = np.asarray(sorted(cellids), dtype=int)
    reader = pt.vlsvfile.VlsvReader(str(file_path))
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
        raise ValueError(f"Invalid cell coordinates read from {file_path}")

    return file_location, cellids, coords_re
