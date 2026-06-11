import numpy as np
import analysator as pt

from src.timesteps import create_timestep_path
from src.vdf_extract import extract_vdf
from src.vdf_helpers import get_cellid_with_vdf


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

    file_location = create_timestep_path(
        path_template=config["file_template_bulk"],
        timestep=timestep
    )

    print(f"Timestep: {timestep}")
    print(f"File: {file_location}")

    reader = pt.vlsvfile.VlsvReader(str(file_location))
    simulation_time = reader.read_parameter("time")

    print(simulation_time)

    samples = []

    for class_name, label, coord_re in labeled_coords:
        print(f"Class: {class_name}, label: {label}")
        print(f"Coordinate RE: {coord_re}")

        cid = get_cellid_with_vdf(
            reader=reader,
            coord_re=coord_re,
        )

        print(f"Cell ID: {cid}")

        vdf = extract_vdf(
            file_location=file_location,
            cid=int(cid)
        ).astype(np.float32, copy=False)

        samples.append(
            {
                "vdf": vdf,
                "label": int(label),
                "metadata": {
                    "timestep": int(timestep),
                    "simulation_time": simulation_time,
                    "cid": int(cid),
                    "label": int(label),
                    "class_name": class_name,
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