import numpy as np
import analysator as pt

from src.timesteps import create_timestep_path
from src.vdf_extract import extract_vdf
from src.vdf_helpers import get_neighbor_vdf_cellids

def create_timestep_sample_specs(config, timestep, labeled_coords):
    """
    Create VDF sample specifications for one timestep.

    Parameters
    ----------
    config : dict
        Dataset configuration containing the VLSV file template.
    timestep : int
        Timestep to process.
    labeled_coords : iterable of tuple
        Tuples of ``(class_name, label, coord_re)``

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
    neighborhood_config = config.get("vdf_neighborhood", {})

    sample_specs = []
    seen_class_cellids = set()

    for class_name, label, coord_re in labeled_coords:
        cellids_by_position = get_neighbor_vdf_cellids(
            reader=reader,
            coord_re=coord_re,
            neighborhood_config=neighborhood_config,
        )

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

    return remove_shared_flux_point_cellids(
        sample_specs=sample_specs,
        config=config,
    )


def remove_shared_flux_point_cellids(sample_specs, config):
    """
    Remove VDF cells shared by configured X-point and O-point classes.

    Parameters
    ----------
    sample_specs : list of dict
        Sample specification dictionaries.
    config : dict
        Dataset configuration.

    Returns
    -------
    list of dict
        Sample specifications with shared flux-point cells removed.
    """

    flux_points_config = config.get("flux_points", {})
    x_class_name = flux_points_config.get("x_class_name")
    o_class_name = flux_points_config.get("o_class_name")

    if x_class_name is None or o_class_name is None:
        return sample_specs

    x_cellids = {
        sample_spec["cid"]
        for sample_spec in sample_specs
        if sample_spec["class_name"] == x_class_name
    }
    o_cellids = {
        sample_spec["cid"]
        for sample_spec in sample_specs
        if sample_spec["class_name"] == o_class_name
    }
    shared_cellids = x_cellids & o_cellids

    if not shared_cellids:
        return sample_specs

    flux_point_classes = {x_class_name, o_class_name}

    return [
        sample_spec
        for sample_spec in sample_specs
        if not (
            sample_spec["class_name"] in flux_point_classes
            and sample_spec["cid"] in shared_cellids
        )
    ]


def count_timestep_samples(config, timestep, labeled_coords):
    """
    Count VDF samples for one timestep after neighborhood expansion.

    Parameters
    ----------
    config : dict
        Dataset configuration containing the VLSV file template.
    timestep : int
        Timestep to process.
    labeled_coords : iterable of tuple
        Tuples of ``(class_name, label, coord_re)``

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
        )
    )


def process_timestep(config, timestep, labeled_coords):
    """
    Extract labeled VDF samples for one timestep.

    Parameters
    ----------
    config : dict
        Dataset configuration containing the VLSV file template.
    timestep : int
        Timestep to process.
    labeled_coords : iterable of tuple
        Tuples of ``(class_name, label, coord_re)``

    Returns
    -------
    list of dict
        Sample dictionaries.
    """

    sample_specs = create_timestep_sample_specs(
        config=config,
        timestep=timestep,
        labeled_coords=labeled_coords,
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
