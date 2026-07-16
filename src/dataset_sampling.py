import time

import numpy as np
import analysator as pt

from src.dataset_metadata import (
    OMITTED_DATASET_METADATA_COLUMNS,
    POINT_REFERENCE_METADATA_COLUMNS,
    create_point_reference_metadata_arrays,
    create_vdf_spatial_metadata,
)
from src.timesteps import create_timestep_path
from src.vdf_extract import VdfExtractor, create_velocity_grid_descriptor
from src.point_labels import create_point_label_data, iter_labeled_coords
from src.point_selection import (
    create_point_sample_metadata,
    get_point_selection_result,
    is_point_record,
    unpack_labeled_coord,
)
from src.vdf_helpers import (
    R_EARTH,
    create_region_mask_re,
    get_nearest_vdf_cellid,
    get_region_axis_bounds_re,
    get_vdf_cells_with_coords_re,
    iter_enabled_regions_re,
)


def create_timestep_sample_specs_for_timestep(config, timestep):
    """
    Create sample specs for one timestep using one shared VLSV reader.

    Parameters
    ----------
    config : dict
        Dataset configuration.
    timestep : int
        Timestep to process.

    Returns
    -------
    tuple
        Pair of ``(timestep, sample_specs)``.
    """

    file_location = create_timestep_path(
        path_template=config["file_template_bulk"],
        timestep=timestep,
    )
    planning_start = time.perf_counter()

    reader_start = time.perf_counter()
    reader = pt.vlsvfile.VlsvReader(str(file_location))
    reader_elapsed = time.perf_counter() - reader_start

    velocity_grid = create_velocity_grid_descriptor(reader=reader)
    population = velocity_grid["population"]

    vdf_cells_start = time.perf_counter()
    vdf_cellids, vdf_coords_re = get_vdf_cells_with_coords_re(
        reader,
        pop=population,
    )
    vdf_cells_elapsed = time.perf_counter() - vdf_cells_start

    point_start = time.perf_counter()
    labeled_coords = list(iter_labeled_coords(config))
    point_label_data = create_point_label_data(
        config=config,
        timestep=timestep,
        reader=reader,
    )
    labeled_coords.extend(point_label_data["point_labeled_coords"])
    point_elapsed = time.perf_counter() - point_start

    specs_start = time.perf_counter()
    sample_specs = create_timestep_sample_specs(
        config=config,
        timestep=timestep,
        labeled_coords=labeled_coords,
        rejected_cellids=point_label_data["rejected_cellids"],
        reader=reader,
        vdf_cellids=vdf_cellids,
        vdf_coords_re=vdf_coords_re,
        raw_x_point_records=point_label_data["raw_x_point_records"],
        raw_o_point_records=point_label_data["raw_o_point_records"],
    )
    if sample_specs:
        sample_specs[0]["velocity_grid"] = velocity_grid

    specs_elapsed = time.perf_counter() - specs_start
    planning_elapsed = time.perf_counter() - planning_start

    print(
        f"Timestep {int(timestep)} planning: "
        f"reader={reader_elapsed:.2f}s, "
        f"vdf_cells={vdf_cells_elapsed:.2f}s, "
        f"points={point_elapsed:.2f}s, "
        f"samples={specs_elapsed:.2f}s, "
        f"total={planning_elapsed:.2f}s, "
        f"n={len(sample_specs)}"
    )

    return int(timestep), sample_specs


def create_timestep_sample_specs(
    config,
    timestep,
    labeled_coords,
    rejected_cellids=None,
    reader=None,
    vdf_cellids=None,
    vdf_coords_re=None,
    raw_x_point_records=None,
    raw_o_point_records=None,
):
    """
    Create VDF sample specifications for one timestep.

    Point classes configured under ``points`` are expanded using magnetic
    topology. Each detected X/O point can use either the physical selection
    method or a fixed manual search box from the point-selection config.
    Background classes are then filled from VDF cells in the configured
    sampling regions that were not already assigned to a point or static class.

    Parameters
    ----------
    config : dict
        Dataset configuration containing file templates, labels, and point
        selection settings.
    timestep : int
        Timestep to process.
    labeled_coords : iterable of tuple
        Tuples of ``(class_name, label, coord_re)``.
    rejected_cellids : set of int, optional
        VDF cell IDs rejected earlier in point-label detection.
    reader : analysator.vlsvfile.VlsvReader, optional
        Existing reader for the timestep bulk file.
    vdf_cellids : numpy.ndarray, optional
        Spatial cell IDs with VDF data.
    vdf_coords_re : numpy.ndarray, optional
        VDF cell coordinates in Earth radii with shape ``(n_cells, 3)``.
    raw_x_point_records : iterable of dict, optional
        All X-point records returned by topology detection before VDF-cell
        conflict removal. Accepted X-point labels are used as a compatibility
        fallback when omitted.
    raw_o_point_records : iterable of dict, optional
        All O-point records returned by topology detection before VDF-cell
        conflict removal. Accepted O-point labels are used as a compatibility
        fallback when omitted.

    Returns
    -------
    list of dict
        Sample specification dictionaries.
    """

    file_location = create_timestep_path(
        path_template=config["file_template_bulk"],
        timestep=timestep
    )

    if reader is None:
        reader = pt.vlsvfile.VlsvReader(str(file_location))

    labeled_coords = list(labeled_coords)
    if raw_x_point_records is None:
        raw_x_point_records = [
            labeled_coord
            for labeled_coord in labeled_coords
            if is_point_record(labeled_coord)
            and str(labeled_coord.get("point_kind", "")).lower() == "x"
        ]
    else:
        raw_x_point_records = list(raw_x_point_records)
    if raw_o_point_records is None:
        raw_o_point_records = [
            labeled_coord
            for labeled_coord in labeled_coords
            if is_point_record(labeled_coord)
            and str(labeled_coord.get("point_kind", "")).lower() == "o"
        ]
    else:
        raw_o_point_records = list(raw_o_point_records)
    rejected_cellids = {int(cid) for cid in (rejected_cellids or set())}
    if vdf_cellids is None or vdf_coords_re is None:
        vdf_cellids, vdf_coords_re = get_vdf_cells_with_coords_re(reader)
    vdf_coord_by_cellid = {
        int(cid): tuple(float(value) for value in coord_re)
        for cid, coord_re in zip(vdf_cellids, vdf_coords_re)
    }

    sample_specs = []
    seen_class_cellids = set()

    for labeled_coord in labeled_coords:
        class_name, label, coord_re = unpack_labeled_coord(labeled_coord)
        sample_metadata = {}
        metadata_by_position = {}

        if is_point_record(labeled_coord):
            selection_result = get_point_selection_result(
                config=config,
                point_record=labeled_coord,
                vdf_cellids=vdf_cellids,
                vdf_coords_re=vdf_coords_re,
            )
            cellids_by_position = selection_result["cellids_by_position"]
            rejected_cellids.update(selection_result["rejected_cellids"])
            metadata_by_position = selection_result["metadata_by_position"]
            sample_metadata = create_point_sample_metadata(
                config=config,
                point_record=labeled_coord,
            )
        else:
            cid = get_nearest_vdf_cellid(
                coord_re=coord_re,
                vdf_cellids=vdf_cellids,
                vdf_coords_re=vdf_coords_re,
            )
            cellids_by_position = {"closest": cid}

        for neighbor_position, cid in cellids_by_position.items():
            class_cellid = (class_name, int(cid))

            if class_cellid in seen_class_cellids:
                continue

            seen_class_cellids.add(class_cellid)
            sample_spec = {
                "file_location": file_location,
                "cid": int(cid),
                "label": int(label),
                "class_name": class_name,
                "coord_re": coord_re,
                "vdf_coord_re": vdf_coord_by_cellid[int(cid)],
                "timestep": int(timestep),
            }
            sample_spec.update(
                {
                    key: value
                    for key, value in sample_metadata.items()
                    if key not in OMITTED_DATASET_METADATA_COLUMNS
                }
            )
            sample_spec.update(
                {
                    key: value
                    for key, value in metadata_by_position.get(
                        neighbor_position,
                        {},
                    ).items()
                    if key not in OMITTED_DATASET_METADATA_COLUMNS
                }
            )
            sample_specs.append(sample_spec)

    conflicting_cellids = find_conflicting_cellids(sample_specs)
    rejected_cellids.update(conflicting_cellids)
    sample_specs = remove_conflicting_cellids(
        sample_specs=sample_specs,
        conflicting_cellids=rejected_cellids,
    )

    sample_specs.extend(
        create_background_region_sample_specs(
            config=config,
            reader=reader,
            file_location=file_location,
            simulation_time=None,
            existing_sample_specs=sample_specs,
            timestep=timestep,
            rejected_cellids=rejected_cellids,
            vdf_cellids=vdf_cellids,
            vdf_coords_re=vdf_coords_re,
            vdf_coord_by_cellid=vdf_coord_by_cellid,
        )
    )

    sample_specs = remove_conflicting_cellids(
        sample_specs=sample_specs,
        conflicting_cellids=find_conflicting_cellids(sample_specs),
    )
    add_point_reference_metadata(
        sample_specs=sample_specs,
        raw_x_point_records=raw_x_point_records,
        raw_o_point_records=raw_o_point_records,
    )
    return sample_specs


def add_point_reference_metadata(
    sample_specs,
    raw_x_point_records=None,
    raw_o_point_records=None,
):
    """
    Add source-or-nearest X/O geometry to final timestep sample specs.

    X-selected samples use their source X point and O-selected samples use
    their source O point. All other references use the closest raw detected
    point of that kind. Vectors point from the VDF-cell center to the point.

    Parameters
    ----------
    sample_specs : list of dict
        Final sample specifications for one timestep.
    raw_x_point_records : iterable of dict, optional
        Raw X-point records returned by topology detection.
    raw_o_point_records : iterable of dict, optional
        Raw O-point records returned by topology detection.

    Returns
    -------
    list of dict
        The input sample specifications with point-reference fields added.
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


def create_background_region_sample_specs(
    config,
    reader,
    file_location,
    simulation_time,
    existing_sample_specs,
    timestep,
    rejected_cellids=None,
    cell_has_vdf_func=None,
    vdf_cellids=None,
    vdf_coords_re=None,
    vdf_coord_by_cellid=None,
):
    """
    Create background samples from configured spatial regions.

    Each region can define ``background_class_name``. Cells already assigned to
    static coordinates or X/O point boxes are excluded, so region backgrounds
    are filled only by VDF cells that are not point-selected samples.

    Parameters
    ----------
    config : dict
        Dataset configuration containing ``labels`` and ``points`` sections.
    reader : analysator.vlsvfile.VlsvReader
        Reader for the timestep VLSV file.
    file_location : pathlib.Path
        Path to the timestep VLSV file.
    simulation_time : float or None
        Retained for compatibility. Simulation time is no longer stored in
        dataset metadata.
    existing_sample_specs : list of dict
        Sample specifications already created for configured point and static
        coordinate classes.
    timestep : int
        Timestep to process.
    rejected_cellids : set of int, optional
        Cell IDs rejected because they had conflicting class labels before the
        background region samples were created.
    cell_has_vdf_func : callable, optional
        Function taking a cell ID and returning whether it has VDF data.
    vdf_cellids : numpy.ndarray, optional
        Spatial cell IDs with VDF data.
    vdf_coords_re : numpy.ndarray, optional
        VDF cell coordinates in Earth radii with shape ``(n_cells, 3)``.
    vdf_coord_by_cellid : dict, optional
        Mapping from VDF cell IDs to cell-center coordinates in Earth radii.

    Returns
    -------
    list of dict
        Sample specifications for background VDF cells in configured regions.
    """

    labels = config["labels"]
    excluded_cellids = {
        int(sample_spec["cid"])
        for sample_spec in existing_sample_specs
    }
    excluded_cellids.update(int(cid) for cid in (rejected_cellids or set()))
    seen_background_class_cellids = set()
    sample_specs = []
    if vdf_coord_by_cellid is None:
        if vdf_cellids is not None and vdf_coords_re is not None:
            vdf_coord_by_cellid = {
                int(cid): tuple(float(value) for value in coord_re)
                for cid, coord_re in zip(vdf_cellids, vdf_coords_re)
            }
        else:
            vdf_coord_by_cellid = {}

    points_config = config.get("points", {})
    for region_name, region_re in iter_enabled_regions_re(
            points_config=points_config,
            names_key="background_region_names",
    ):
        class_name = get_region_background_class_name(
            points_config=points_config,
            region_name=region_name,
            region_re=region_re,
        )
        if class_name not in labels:
            raise ValueError(
                f"Region {region_name!r} uses background class "
                f"{class_name!r}, but labels has no such class"
            )
        label = int(labels[class_name])

        for cid in get_vdf_cellids_in_region_re(
            reader=reader,
            region_re=region_re,
            cell_has_vdf_func=cell_has_vdf_func,
            vdf_cellids=vdf_cellids,
            vdf_coords_re=vdf_coords_re,
        ):
            cid = int(cid)
            background_class_cellid = (class_name, cid)
            if (
                    cid in excluded_cellids
                    or background_class_cellid in seen_background_class_cellids
            ):
                continue

            seen_background_class_cellids.add(background_class_cellid)
            coord_re = vdf_coord_by_cellid.get(cid)
            if coord_re is None:
                coord_re = (
                    np.asarray(reader.get_cell_coordinates(cid), dtype=float)
                    / R_EARTH
                )
            sample_specs.append(
                {
                    "file_location": file_location,
                    "cid": cid,
                    "label": label,
                    "class_name": class_name,
                    "coord_re": coord_re,
                    "vdf_coord_re": coord_re,
                    "timestep": int(timestep),
                }
            )

    return sample_specs


def create_other_region_sample_specs(*args, **kwargs):
    """
    Create background samples from configured regions.

    This compatibility wrapper keeps older direct callers working after the
    region background classes were generalized beyond ``other``.
    """

    return create_background_region_sample_specs(*args, **kwargs)


def get_region_background_class_name(points_config, region_name, region_re):
    """
    Return the background class name for one spatial region.

    Parameters
    ----------
    points_config : dict
        Point sampling configuration.
    region_name : str
        Name of the configured region.
    region_re : dict
        Region bounds and optional background class.

    Returns
    -------
    str
        Background class name for cells not selected as point samples.
    """

    class_name = region_re.get("background_class_name")
    if class_name is not None:
        return str(class_name)

    class_names = (points_config or {}).get("background_class_names", {})
    if region_name in class_names:
        return str(class_names[region_name])

    if region_name == "tail":
        return str((points_config or {}).get("background_class_name", "other"))

    return str(region_name)


def get_vdf_cellids_in_point_region(
    reader,
    config,
    cell_has_vdf_func=None,
    vdf_cellids=None,
    vdf_coords_re=None,
):
    """
    Return VDF cell IDs inside configured background regions.

    Parameters
    ----------
    reader : analysator.vlsvfile.VlsvReader
        Reader for the timestep VLSV file.
    config : dict
        Dataset configuration containing the ``points`` region bounds.
    cell_has_vdf_func : callable, optional
        Function taking a cell ID and returning whether it has VDF data.
    vdf_cellids : numpy.ndarray, optional
        Spatial cell IDs with VDF data.
    vdf_coords_re : numpy.ndarray, optional
        VDF cell coordinates in Earth radii with shape ``(n_cells, 3)``.

    Returns
    -------
    list of int
        Cell IDs with VDF data inside the configured background regions.
    """

    selected_cellids = []
    seen_cellids = set()
    points_config = config.get("points", {})

    for _, region_re in iter_enabled_regions_re(
            points_config=points_config,
            names_key="background_region_names",
    ):
        for cid in get_vdf_cellids_in_region_re(
                reader=reader,
                region_re=region_re,
                cell_has_vdf_func=cell_has_vdf_func,
                vdf_cellids=vdf_cellids,
                vdf_coords_re=vdf_coords_re,
        ):
            if cid in seen_cellids:
                continue

            seen_cellids.add(cid)
            selected_cellids.append(cid)

    return selected_cellids


def get_vdf_cellids_in_region_re(
    reader,
    region_re,
    cell_has_vdf_func=None,
    vdf_cellids=None,
    vdf_coords_re=None,
):
    """
    Return VDF cell IDs inside one configured spatial region.

    Parameters
    ----------
    reader : analysator.vlsvfile.VlsvReader
        Reader for the timestep VLSV file.
    region_re : dict
        Region bounds in Earth radii.
    cell_has_vdf_func : callable, optional
        Function taking a cell ID and returning whether it has VDF data.
    vdf_cellids : numpy.ndarray, optional
        Spatial cell IDs with VDF data.
    vdf_coords_re : numpy.ndarray, optional
        VDF cell coordinates in Earth radii with shape ``(n_cells, 3)``.

    Returns
    -------
    list of int
        Cell IDs with VDF data inside the configured spatial region.
    """

    if vdf_cellids is None or vdf_coords_re is None:
        vdf_cellids, vdf_coords_re = get_vdf_cells_with_coords_re(reader)

    if len(vdf_cellids) == 0:
        return []

    selected = create_region_mask_re(vdf_coords_re, region_re)

    selected_cellids = vdf_cellids[selected]
    if cell_has_vdf_func is not None:
        selected_cellids = [
            int(cid)
            for cid in selected_cellids
            if cell_has_vdf_func(int(cid))
        ]

    return [int(cid) for cid in selected_cellids]


def get_region_x_bounds_re(region_re):
    """
    Return sorted x bounds in Earth radii from a region config.

    Parameters
    ----------
    region_re : dict
        Region configuration with either ``x_between`` or ``x_min``/``x_max``.

    Returns
    -------
    tuple of float
        Lower and upper x bounds in Earth radii.
    """

    x_min_re, x_max_re = get_region_axis_bounds_re(region_re, "x")
    if x_min_re is None or x_max_re is None:
        raise ValueError("Region config must define x bounds")

    return x_min_re, x_max_re


def find_conflicting_cellids(sample_specs):
    """
    Find VDF cell IDs assigned to more than one class.

    Parameters
    ----------
    sample_specs : list of dict
        Sample specification dictionaries.

    Returns
    -------
    set of int
        Cell IDs that have conflicting class labels.
    """

    class_names_by_cellid = {}

    for sample_spec in sample_specs:
        cid = int(sample_spec["cid"])
        class_name = sample_spec["class_name"]
        class_names_by_cellid.setdefault(cid, set()).add(class_name)

    return {
        cid
        for cid, class_names in class_names_by_cellid.items()
        if len(class_names) > 1
    }


def remove_conflicting_cellids(sample_specs, conflicting_cellids):
    """
    Remove samples whose VDF cell ID has conflicting class labels.

    Parameters
    ----------
    sample_specs : list of dict
        Sample specification dictionaries.
    conflicting_cellids : set of int
        Cell IDs that should be rejected from every class.

    Returns
    -------
    list of dict
        Sample specifications without conflicting cell IDs.
    """

    if not conflicting_cellids:
        return sample_specs

    conflicting_cellids = {int(cid) for cid in conflicting_cellids}

    return [
        sample_spec
        for sample_spec in sample_specs
        if int(sample_spec["cid"]) not in conflicting_cellids
    ]


def count_timestep_samples(config, timestep, labeled_coords, rejected_cellids=None):
    """
    Count VDF samples for one timestep after neighborhood expansion.

    Prefer creating sample specs once and reusing them for extraction in new
    code paths. This helper is kept for compatibility with direct callers.

    Parameters
    ----------
    config : dict
        Dataset configuration containing the VLSV file template.
    timestep : int
        Timestep to process.
    labeled_coords : iterable of tuple
        Tuples of ``(class_name, label, coord_re)``.
    rejected_cellids : set of int, optional
        VDF cell IDs rejected earlier in point-label detection.

    Returns
    -------
    int
        Number of samples for the timestep.
    """

    return len(
        create_timestep_sample_specs(
            config=config,
            timestep=timestep,
            labeled_coords=labeled_coords,
            rejected_cellids=rejected_cellids,
        )
    )


def iter_timestep_sample_specs(sample_specs):
    """
    Extract labeled VDF samples from precomputed timestep sample specs.

    Parameters
    ----------
    sample_specs : list of dict
        Sample specifications for one timestep.

    Yields
    ------
    dict
        Sample dictionary containing ``vdf``, ``label``, and ``metadata``.
    """

    if not sample_specs:
        return

    file_location = sample_specs[0]["file_location"]
    timestep = int(sample_specs[0].get("timestep", 0))
    velocity_grid = sample_specs[0].get("velocity_grid")
    population = (
        str(velocity_grid["population"])
        if velocity_grid is not None
        else None
    )
    extraction_start = time.perf_counter()

    print_memory_usage(f"timestep {timestep} before reader")
    reader = pt.vlsvfile.VlsvReader(str(file_location))
    extractor = VdfExtractor(reader=reader, pop=population)
    print_memory_usage(f"timestep {timestep} after reader")

    print(f"Timestep {timestep}: extracting {len(sample_specs)} samples")

    for sample_spec in sample_specs:
        coord_re = sample_spec["coord_re"]
        cid = int(sample_spec["cid"])

        vdf = extractor.extract(cid=cid).astype(np.float32, copy=False)

        yield {
            "vdf": vdf,
            "label": sample_spec["label"],
            "metadata": create_sample_metadata_row(
                sample_spec=sample_spec,
                cid=cid,
                coord_re=coord_re,
                file_location=file_location,
            ),
        }

    extraction_elapsed = time.perf_counter() - extraction_start
    print_memory_usage(f"timestep {timestep} extraction complete")
    print(f"Timestep {timestep} extraction: {extraction_elapsed:.2f} s")


def process_timestep_sample_specs(sample_specs):
    """
    Extract labeled VDF samples from precomputed timestep sample specs.

    Parameters
    ----------
    sample_specs : list of dict
        Sample specifications for one timestep.

    Returns
    -------
    list of dict
        Sample dictionaries.
    """

    return list(iter_timestep_sample_specs(sample_specs))


def create_sample_metadata_row(sample_spec, cid, coord_re, file_location):
    """
    Create one metadata row for an extracted sample.

    Parameters
    ----------
    sample_spec : dict
        Sample specification used for extraction. Must contain the VDF-cell
        center as ``vdf_coord_re``. X/O-point samples must also contain
        ``point_kind`` and the three ``source_point_*_re`` coordinates.
    cid : int
        Spatial cell ID used for extraction.
    coord_re : array-like of float
        Coordinate in Earth radii, given as ``[x, y, z]``.
    file_location : pathlib.Path
        Source VLSV file path.

    Returns
    -------
    dict
        Metadata fields for the extracted sample, including the VDF-cell
        center and source-or-nearest X/O distances and vectors in Earth radii.
    """

    metadata_row = {
        "timestep": int(sample_spec["timestep"]),
        "cid": int(cid),
        "label": sample_spec["label"],
        "class_name": sample_spec["class_name"],
        "x_re": float(coord_re[0]),
        "y_re": float(coord_re[1]),
        "z_re": float(coord_re[2]),
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


def process_timestep(config, timestep, labeled_coords, rejected_cellids=None):
    """
    Extract labeled VDF samples for one timestep.

    This compatibility wrapper creates the timestep sample specs and then
    extracts them with a single shared VLSV reader.

    Parameters
    ----------
    config : dict
        Dataset configuration containing the VLSV file template.
    timestep : int
        Timestep to process.
    labeled_coords : iterable of tuple
        Tuples of ``(class_name, label, coord_re)``.
    rejected_cellids : set of int, optional
        VDF cell IDs rejected earlier in point-label detection.

    Returns
    -------
    list of dict
        Sample dictionaries.
    """

    sample_specs = create_timestep_sample_specs(
        config=config,
        timestep=timestep,
        labeled_coords=labeled_coords,
        rejected_cellids=rejected_cellids,
    )

    return process_timestep_sample_specs(sample_specs)


def write_timestep_samples(X, y, metadata, timestep_samples, sample_index):
    """
    Write samples from one timestep into dataset arrays and metadata.

    Parameters
    ----------
    X : numpy.ndarray
        Output array for VDF samples.
    y : numpy.ndarray
        Output array for integer labels.
    metadata : list of dict
        Metadata rows accumulated for the dataset.
    timestep_samples : iterable of dict
        Sample dictionaries.
    sample_index : int
        Index where writing should start.

    Returns
    -------
    int
        Next sample index after the written samples.
    """

    for sample in timestep_samples:
        X[sample_index] = sample["vdf"]
        y[sample_index] = sample["label"]

        metadata_row = {"sample_index": sample_index}
        metadata_row.update(sample["metadata"])
        metadata.append(metadata_row)

        sample_index += 1

    return sample_index


def iter_chunks(items, chunk_size):
    """
    Yield consecutive chunks from a sequence.

    Parameters
    ----------
    items : sequence
        Items to split into chunks.
    chunk_size : int
        Maximum number of items per chunk.

    Yields
    ------
    sequence
        Slices of ``items`` with at most ``chunk_size`` entries.
    """

    for start in range(0, len(items), chunk_size):
        yield items[start:start + chunk_size]


def get_point_class_names(config):
    """
    Return configured point class names.

    Parameters
    ----------
    config : dict
        Dataset configuration.

    Returns
    -------
    set of str
        X-point and O-point class names.
    """

    points_config = config.get("points", {})
    class_names = {
        points_config.get("x_class_name"),
        points_config.get("o_class_name"),
    }

    return {class_name for class_name in class_names if class_name is not None}


def print_memory_usage(label):
    """
    Print current process resident memory usage.

    Parameters
    ----------
    label : str
        Text describing where memory is measured.
    """

    print(f"Memory {label}: {get_memory_usage_mb():.1f} MB")


def get_memory_usage_mb():
    """
    Return current process resident memory usage.

    Returns
    -------
    float
        Resident set size in megabytes.
    """

    with open("/proc/self/status", "r") as status_file:
        for line in status_file:
            if line.startswith("VmRSS:"):
                value_kb = float(line.split()[1])
                return value_kb / 1024.0

    return 0.0
