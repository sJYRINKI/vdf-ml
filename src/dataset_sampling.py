import time

import numpy as np
import analysator as pt

from src.timesteps import create_timestep_path
from src.vdf_extract import VdfExtractor
from src.point_labels import create_point_labeled_coords, iter_labeled_coords
from src.point_selection import (
    create_point_sample_metadata,
    get_point_cellids_by_position,
    is_point_record,
    unpack_labeled_coord,
)
from src.vdf_helpers import (
    R_EARTH,
    get_nearest_vdf_cellid,
    get_vdf_cells_with_coords_re,
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

    vdf_cells_start = time.perf_counter()
    vdf_cellids, vdf_coords_re = get_vdf_cells_with_coords_re(reader)
    vdf_cells_elapsed = time.perf_counter() - vdf_cells_start

    point_start = time.perf_counter()
    labeled_coords = list(iter_labeled_coords(config))
    point_labeled_coords, rejected_cellids = create_point_labeled_coords(
        config=config,
        timestep=timestep,
        reader=reader,
    )
    labeled_coords.extend(point_labeled_coords)
    point_elapsed = time.perf_counter() - point_start

    specs_start = time.perf_counter()
    sample_specs = create_timestep_sample_specs(
        config=config,
        timestep=timestep,
        labeled_coords=labeled_coords,
        rejected_cellids=rejected_cellids,
        reader=reader,
        vdf_cellids=vdf_cellids,
        vdf_coords_re=vdf_coords_re,
    )

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
):
    """
    Create VDF sample specifications for one timestep.

    Point classes configured under ``points`` are expanded using magnetic
    topology. X points use Hessian-aligned boxes scaled by local ion inertial
    length, and O points use the closed island contour built from X-point flux.
    The ``other`` class is then filled from all VDF cells in ``points.region_re``
    that were not already assigned to the X-point or O-point classes.

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

    simulation_time = reader.read_parameter("time")
    labeled_coords = list(labeled_coords)
    rejected_cellids = {int(cid) for cid in (rejected_cellids or set())}
    if vdf_cellids is None or vdf_coords_re is None:
        vdf_cellids, vdf_coords_re = get_vdf_cells_with_coords_re(reader)

    sample_specs = []
    seen_class_cellids = set()

    for labeled_coord in labeled_coords:
        class_name, label, coord_re = unpack_labeled_coord(labeled_coord)
        sample_metadata = {}

        if is_point_record(labeled_coord):
            cellids_by_position = get_point_cellids_by_position(
                config=config,
                point_record=labeled_coord,
                vdf_cellids=vdf_cellids,
                vdf_coords_re=vdf_coords_re,
            )
            sample_metadata = create_point_sample_metadata(labeled_coord)
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
                "simulation_time": simulation_time,
                "cid": int(cid),
                "label": int(label),
                "class_name": class_name,
                "coord_re": coord_re,
                "neighbor_position": neighbor_position,
                "timestep": int(timestep),
            }
            sample_spec.update(sample_metadata)
            sample_specs.append(sample_spec)

    conflicting_cellids = find_conflicting_cellids(sample_specs)
    rejected_cellids.update(conflicting_cellids)
    sample_specs = remove_conflicting_cellids(
        sample_specs=sample_specs,
        conflicting_cellids=rejected_cellids,
    )

    sample_specs.extend(
        create_other_region_sample_specs(
            config=config,
            reader=reader,
            file_location=file_location,
            simulation_time=simulation_time,
            existing_sample_specs=sample_specs,
            timestep=timestep,
            rejected_cellids=rejected_cellids,
            vdf_cellids=vdf_cellids,
            vdf_coords_re=vdf_coords_re,
        )
    )

    return remove_conflicting_cellids(
        sample_specs=sample_specs,
        conflicting_cellids=find_conflicting_cellids(sample_specs),
    )


def create_other_region_sample_specs(
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
):
    """
    Create ``other`` samples from VDF cells in the point region.

    The ``other`` class is always derived from ``points.region_re``. Cells
    already assigned to X/O point boxes are excluded, so the region is filled
    only by VDF cells that are not reconnection or O-point samples.

    Parameters
    ----------
    config : dict
        Dataset configuration containing ``labels`` and ``points`` sections.
    reader : analysator.vlsvfile.VlsvReader
        Reader for the timestep VLSV file.
    file_location : pathlib.Path
        Path to the timestep VLSV file.
    simulation_time : float
        Simulation time read from the VLSV file.
    existing_sample_specs : list of dict
        Sample specifications already created for configured point and static
        coordinate classes.
    timestep : int
        Timestep to process.
    rejected_cellids : set of int, optional
        Cell IDs rejected because they had conflicting class labels before the
        ``other`` region samples were created.
    cell_has_vdf_func : callable, optional
        Function taking a cell ID and returning whether it has VDF data.
    vdf_cellids : numpy.ndarray, optional
        Spatial cell IDs with VDF data.
    vdf_coords_re : numpy.ndarray, optional
        VDF cell coordinates in Earth radii with shape ``(n_cells, 3)``.

    Returns
    -------
    list of dict
        Sample specifications for ``other`` VDF cells in the point region.
    """

    labels = config["labels"]
    class_name = "other"
    label = int(labels[class_name])

    excluded_cellids = {
        int(sample_spec["cid"])
        for sample_spec in existing_sample_specs
    }
    excluded_cellids.update(int(cid) for cid in (rejected_cellids or set()))
    seen_other_cellids = set()
    sample_specs = []
    coord_by_cellid = {}
    if vdf_cellids is not None and vdf_coords_re is not None:
        coord_by_cellid = {
            int(cid): np.asarray(coord_re, dtype=float)
            for cid, coord_re in zip(vdf_cellids, vdf_coords_re)
        }

    for cid in get_vdf_cellids_in_point_region(
        reader=reader,
        config=config,
        cell_has_vdf_func=cell_has_vdf_func,
        vdf_cellids=vdf_cellids,
        vdf_coords_re=vdf_coords_re,
    ):
        cid = int(cid)
        if cid in excluded_cellids or cid in seen_other_cellids:
            continue

        seen_other_cellids.add(cid)
        coord_re = coord_by_cellid.get(cid)
        if coord_re is None:
            coord_re = np.asarray(reader.get_cell_coordinates(cid), dtype=float) / R_EARTH
        sample_specs.append(
            {
                "file_location": file_location,
                "simulation_time": simulation_time,
                "cid": cid,
                "label": label,
                "class_name": class_name,
                "coord_re": coord_re,
                "neighbor_position": "point_region",
                "timestep": int(timestep),
            }
        )

    return sample_specs


def get_vdf_cellids_in_point_region(
    reader,
    config,
    cell_has_vdf_func=None,
    vdf_cellids=None,
    vdf_coords_re=None,
):
    """
    Return VDF cell IDs inside ``points.region_re``.

    Parameters
    ----------
    reader : analysator.vlsvfile.VlsvReader
        Reader for the timestep VLSV file.
    config : dict
        Dataset configuration containing the ``points.region_re`` bounds.
    cell_has_vdf_func : callable, optional
        Function taking a cell ID and returning whether it has VDF data.
    vdf_cellids : numpy.ndarray, optional
        Spatial cell IDs with VDF data.
    vdf_coords_re : numpy.ndarray, optional
        VDF cell coordinates in Earth radii with shape ``(n_cells, 3)``.

    Returns
    -------
    list of int
        Cell IDs with VDF data inside the configured point region.
    """

    if vdf_cellids is None or vdf_coords_re is None:
        vdf_cellids, vdf_coords_re = get_vdf_cells_with_coords_re(reader)

    if len(vdf_cellids) == 0:
        return []

    region_re = config.get("points", {}).get("region_re", {})
    x_min_re, x_max_re = get_region_x_bounds_re(region_re)
    z_abs_max_re = float(region_re["z_abs_max"])

    y_min_re = float(reader.read_parameter("ymin")) / R_EARTH
    y_max_re = float(reader.read_parameter("ymax")) / R_EARTH

    selected = (
        (vdf_coords_re[:, 0] >= x_min_re)
        & (vdf_coords_re[:, 0] <= x_max_re)
        & (vdf_coords_re[:, 1] >= y_min_re)
        & (vdf_coords_re[:, 1] <= y_max_re)
        & (vdf_coords_re[:, 2] >= -z_abs_max_re)
        & (vdf_coords_re[:, 2] <= z_abs_max_re)
    )

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

    if region_re.get("x_between") is not None:
        left_x, right_x = region_re["x_between"]
        return min(left_x, right_x), max(left_x, right_x)

    return float(region_re["x_min"]), float(region_re["x_max"])


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
    extraction_start = time.perf_counter()

    print_memory_usage(f"timestep {timestep} before reader")
    reader = pt.vlsvfile.VlsvReader(str(file_location))
    extractor = VdfExtractor(reader=reader)
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
        Sample specification used for extraction.
    cid : int
        Spatial cell ID used for extraction.
    coord_re : array-like of float
        Coordinate in Earth radii, given as ``[x, y, z]``.
    file_location : pathlib.Path
        Source VLSV file path.

    Returns
    -------
    dict
        Metadata fields for the extracted sample.
    """

    metadata_row = {
        "timestep": int(sample_spec["timestep"]),
        "simulation_time": sample_spec["simulation_time"],
        "cid": int(cid),
        "label": sample_spec["label"],
        "class_name": sample_spec["class_name"],
        "neighbor_position": sample_spec["neighbor_position"],
        "x_re": float(coord_re[0]),
        "y_re": float(coord_re[1]),
        "z_re": float(coord_re[2]),
        "file_location": str(file_location),
    }
    internal_keys = {
        "file_location",
        "simulation_time",
        "cid",
        "label",
        "class_name",
        "coord_re",
        "neighbor_position",
        "timestep",
    }

    for key, value in sample_spec.items():
        if key not in internal_keys:
            metadata_row[key] = value

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
