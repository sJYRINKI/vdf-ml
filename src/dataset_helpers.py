import numpy as np
import analysator as pt

from src.timesteps import create_timestep_path
from src.vdf_extract import extract_vdf
from src.vdf_helpers import (
    R_EARTH,
    cell_has_vdf,
    get_cellid_with_vdf,
    get_spatial_index_range,
    get_vdf_cellids_in_box,
)

def create_timestep_sample_specs(config, timestep, labeled_coords, rejected_cellids=None):
    """
    Create VDF sample specifications for one timestep.

    Point classes configured under ``points`` are expanded to spatial boxes.
    The ``other`` class is then filled from all VDF cells in ``points.region_re``
    that were not already assigned to the X-point or O-point classes.

    Parameters
    ----------
    config : dict
        Dataset configuration containing file templates, labels, point classes,
        and VDF box settings.
    timestep : int
        Timestep to process.
    labeled_coords : iterable of tuple
        Tuples of ``(class_name, label, coord_re)``.
    rejected_cellids : set of int, optional
        VDF cell IDs rejected earlier in point-label detection.

    Returns
    -------
    list of dict
        Sample specification dictionaries.
    """

    file_location = create_timestep_path(
        path_template=config["file_template_bulk"],
        timestep=timestep
    )

    reader = pt.vlsvfile.VlsvReader(str(file_location))
    simulation_time = reader.read_parameter("time")
    point_class_names = get_point_class_names(config)
    box_config = config["vdf_box"]
    labeled_coords = list(labeled_coords)
    rejected_cellids = {int(cid) for cid in (rejected_cellids or set())}

    sample_specs = []
    seen_class_cellids = set()

    for class_name, label, coord_re in labeled_coords:
        if class_name in point_class_names:
            cellids_by_position = get_vdf_cellids_in_box(
                reader=reader,
                coord_re=coord_re,
                box_config=box_config,
            )
        else:
            cid = get_cellid_with_vdf(
                reader=reader,
                coord_re=coord_re,
            )
            cellids_by_position = {"closest": cid}

        for neighbor_position, cid in cellids_by_position.items():
            class_cellid = (class_name, int(cid))

            if class_cellid in seen_class_cellids:
                continue

            seen_class_cellids.add(class_cellid)
            sample_specs.append(
                {
                    "file_location": file_location,
                    "simulation_time": simulation_time,
                    "cid": int(cid),
                    "label": int(label),
                    "class_name": class_name,
                    "coord_re": coord_re,
                    "neighbor_position": neighbor_position,
                }
            )

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
            rejected_cellids=rejected_cellids,
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
    rejected_cellids=None,
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
    rejected_cellids : set of int, optional
        Cell IDs rejected because they had conflicting class labels before the
        ``other`` region samples were created.

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

    for cid in get_vdf_cellids_in_point_region(reader, config):
        cid = int(cid)
        if cid in excluded_cellids or cid in seen_other_cellids:
            continue

        seen_other_cellids.add(cid)
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
            }
        )

    return sample_specs


def get_vdf_cellids_in_point_region(reader, config):
    """
    Return VDF cell IDs inside ``points.region_re``.

    Parameters
    ----------
    reader : analysator.vlsvfile.VlsvReader
        Reader for the timestep VLSV file.
    config : dict
        Dataset configuration containing the ``points.region_re`` bounds.

    Returns
    -------
    list of int
        Cell IDs with VDF data inside the configured point region.
    """

    region_re = config.get("points", {}).get("region_re", {})
    x_min_re, x_max_re = get_region_x_bounds_re(region_re)
    z_abs_max_re = float(region_re["z_abs_max"])

    x_min = x_min_re * R_EARTH
    x_max = x_max_re * R_EARTH
    y_min = float(reader.read_parameter("ymin"))
    y_max = float(reader.read_parameter("ymax"))
    z_min = -z_abs_max_re * R_EARTH
    z_max = z_abs_max_re * R_EARTH

    x_index_min, x_index_max = get_spatial_index_range(
        reader=reader,
        axis_name="x",
        axis_index=0,
        min_value=x_min,
        max_value=x_max,
    )
    y_index_min, y_index_max = get_spatial_index_range(
        reader=reader,
        axis_name="y",
        axis_index=1,
        min_value=y_min,
        max_value=y_max,
    )
    z_index_min, z_index_max = get_spatial_index_range(
        reader=reader,
        axis_name="z",
        axis_index=2,
        min_value=z_min,
        max_value=z_max,
    )

    cellids = []
    seen_cellids = set()

    for i in range(x_index_min, x_index_max + 1):
        for j in range(y_index_min, y_index_max + 1):
            for k in range(z_index_min, z_index_max + 1):
                try:
                    cid = int(reader.get_cellid_at_fsgrid_index(i, j, k))
                except Exception:
                    continue

                if cid <= 0 or cid in seen_cellids:
                    continue

                cell_coord = np.asarray(reader.get_cell_coordinates(cid), dtype=float)
                if not (x_min <= cell_coord[0] <= x_max):
                    continue
                if not (y_min <= cell_coord[1] <= y_max):
                    continue
                if not (z_min <= cell_coord[2] <= z_max):
                    continue
                if not cell_has_vdf(reader, cid):
                    continue

                seen_cellids.add(cid)
                cellids.append(cid)

    return cellids


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


def process_timestep(config, timestep, labeled_coords, rejected_cellids=None):
    """
    Extract labeled VDF samples for one timestep.

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

    print(f"Timestep: {timestep}")

    samples = []

    for sample_spec in sample_specs:
        coord_re = sample_spec["coord_re"]
        cid = sample_spec["cid"]
        file_location = sample_spec["file_location"]

        print(
            f"Class: {sample_spec['class_name']}, "
            f"label: {sample_spec['label']}, "
            f"position: {sample_spec['neighbor_position']}"
        )
        print(f"Coordinate RE: {coord_re}")
        print(f"Cell ID: {cid}")

        vdf = extract_vdf(
            file_location=file_location,
            cid=int(cid)
        ).astype(np.float32, copy=False)

        samples.append(
            {
                "vdf": vdf,
                "label": sample_spec["label"],
                "metadata": {
                    "timestep": int(timestep),
                    "simulation_time": sample_spec["simulation_time"],
                    "cid": int(cid),
                    "label": sample_spec["label"],
                    "class_name": sample_spec["class_name"],
                    "neighbor_position": sample_spec["neighbor_position"],
                    "x_re": float(coord_re[0]),
                    "y_re": float(coord_re[1]),
                    "z_re": float(coord_re[2]),
                    "file_location": str(file_location),
                },
            }
        )

        print(f"VDF shape: {vdf.shape}")
        print(f"VDF dtype: {vdf.dtype}")
        print(f"VDF min: {vdf.min()}")
        print(f"VDF max: {vdf.max()}")

    return samples


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
        print(f"Sample index: {sample_index}")

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
