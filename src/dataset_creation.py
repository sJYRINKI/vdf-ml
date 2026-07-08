import os
from pathlib import Path
import tempfile
import time

import numpy as np
from joblib import Parallel, delayed

from src.dataset_sampling import (
    create_timestep_sample_specs_for_timestep,
    iter_chunks,
    iter_timestep_sample_specs,
    print_memory_usage,
    write_timestep_samples,
)
from src.dataset_io import (
    create_dataset_output_dir,
    create_memmap_dataset,
    flush_and_release_memmaps,
    release_memmap_pages,
    save_metadata,
)
from src.timesteps import create_timestep_list


POINT_SELECTION_METHOD_ALIASES = {
    "manual": "manual",
    "physical": "physical",
    "consensus": "consensus",
    "union": "union_physical_priority",
    "union_physical_priority": "union_physical_priority",
}

POINT_SELECTION_METHOD_SUFFIXES = {
    "manual": "manual",
    "physical": "physical",
    "consensus": "consensus",
    "union_physical_priority": "union",
}


def apply_dataset_creation_overrides(
    config,
    x_selection_method=None,
    o_selection_method=None,
):
    """
    Apply command-line dataset creation overrides to config.

    Parameters
    ----------
    config : dict
        Dataset creation config modified in place.
    x_selection_method : str, optional
        X-point selection method override.
    o_selection_method : str, optional
        O-point selection method override.

    Returns
    -------
    str or None
        Dataset name suffix for the applied overrides.
    """

    points_config = config.setdefault("points", {})
    suffix_parts = []

    if x_selection_method is not None:
        method = normalize_point_selection_method(x_selection_method)
        points_config.setdefault("x_selection", {})["method"] = method
        suffix_parts.append(f"x_{POINT_SELECTION_METHOD_SUFFIXES[method]}")

    if o_selection_method is not None:
        method = normalize_point_selection_method(o_selection_method)
        points_config.setdefault("o_selection", {})["method"] = method
        suffix_parts.append(f"o_{POINT_SELECTION_METHOD_SUFFIXES[method]}")

    if not suffix_parts:
        return None

    return "_".join(suffix_parts)


def normalize_point_selection_method(selection_method):
    """
    Normalize a point-selection method name.

    Parameters
    ----------
    selection_method : str
        User-facing point-selection method name.

    Returns
    -------
    str
        Internal point-selection method name.
    """

    selection_method = str(selection_method).lower()
    if selection_method not in POINT_SELECTION_METHOD_ALIASES:
        raise ValueError(
            f"Unsupported point-selection method: {selection_method}. "
            f"Expected one of {sorted(POINT_SELECTION_METHOD_ALIASES)}."
        )

    return POINT_SELECTION_METHOD_ALIASES[selection_method]


def create_dataset(
    config,
    start_timestep,
    n_timesteps,
    dataset_kind,
    output_suffix=None,
):
    """
    Create and save a labeled VDF dataset.

    Parameters
    ----------
    config : dict
        Dataset creation config.
    start_timestep : int
        First timestep to include.
    n_timesteps : int
        Number of consecutive timesteps to process.
    dataset_kind : {"train", "test"}
        Output dataset split name.
    output_suffix : str, optional
        Extra suffix appended to the dataset directory name.
    """

    total_start = time.perf_counter()
    timesteps = create_timestep_list(
        start_timestep=start_timestep,
        n_timesteps=n_timesteps,
    )

    output_dirs = config["output_dirs"]
    output_dir = output_dirs[dataset_kind]

    outdir = create_dataset_output_dir(
        output_dir=output_dir,
        start_timestep=start_timestep,
        n_timestep=n_timesteps,
        name_suffix=output_suffix,
    )

    print(f"Dataset kind: {dataset_kind}")
    print(f"Output directory: {outdir}")

    creation_config = config.get("creation", {})
    default_n_jobs = int(creation_config.get("n_jobs", 1))
    planning_n_jobs = int(
        creation_config.get("planning_n_jobs", default_n_jobs)
    )
    extraction_n_jobs = int(
        creation_config.get("extraction_n_jobs", 1)
    )

    planning_worker_count = get_worker_count(
        planning_n_jobs,
        "creation.planning_n_jobs",
    )
    extraction_worker_count = get_worker_count(
        extraction_n_jobs,
        "creation.extraction_n_jobs",
    )

    print(
        f"Planning jobs: {planning_n_jobs} "
        f"({planning_worker_count} workers)"
    )
    print(
        f"Extraction jobs: {extraction_n_jobs} "
        f"({extraction_worker_count} workers)"
    )

    sample_specs_by_timestep, planning_elapsed = plan_dataset_sample_specs(
        config=config,
        timesteps=timesteps,
        planning_n_jobs=planning_n_jobs,
    )
    sample_counts_by_timestep = count_sample_specs_by_timestep(
        sample_specs_by_timestep
    )
    n_samples = count_sample_counts(sample_counts_by_timestep)

    print(f"Samples: {n_samples}")
    print(f"Timing planning: {planning_elapsed:.2f} s")
    print_memory_usage("after planning")

    if n_samples == 0:
        raise ValueError("No samples were found for the requested timesteps")

    extraction_start = time.perf_counter()
    metadata = []
    sample_index = 0

    first_timestep_index, first_timestep = find_first_nonempty_timestep(
        sample_counts_by_timestep=sample_counts_by_timestep,
        timesteps=timesteps,
    )
    first_sample_specs = sample_specs_by_timestep[first_timestep]
    first_sample, first_sample_iter = extract_first_sample_from_specs(
        first_sample_specs
    )

    X, y = create_memmap_dataset(
        outdir=outdir,
        n_samples=n_samples,
        sample_shape=first_sample["vdf"].shape,
        dtype=np.float32,
    )
    print_memory_usage("after memmap creation")

    sample_index = write_timestep_samples(
        X=X,
        y=y,
        metadata=metadata,
        timestep_samples=[first_sample],
        sample_index=sample_index,
    )
    sample_index = write_timestep_samples(
        X=X,
        y=y,
        metadata=metadata,
        timestep_samples=first_sample_iter,
        sample_index=sample_index,
    )
    sample_specs_by_timestep.pop(first_timestep, None)
    flush_and_release_memmaps(X, y)
    print_memory_usage("after first timestep memmap release")

    remaining_timesteps = timesteps[first_timestep_index + 1:]
    sample_index = write_remaining_timesteps(
        sample_specs_by_timestep=sample_specs_by_timestep,
        X=X,
        y=y,
        metadata=metadata,
        sample_index=sample_index,
        timesteps=remaining_timesteps,
        extraction_n_jobs=extraction_n_jobs,
        extraction_worker_count=extraction_worker_count,
    )

    if sample_index != n_samples:
        raise RuntimeError(
            f"Expected to write {n_samples} samples, wrote {sample_index}"
        )

    extraction_elapsed = time.perf_counter() - extraction_start
    save_start = time.perf_counter()

    print_memory_usage("before flush")
    flush_and_release_memmaps(X, y)
    print_memory_usage("after flush and memmap release")

    save_metadata(
        outdir=outdir,
        metadata=metadata,
    )

    save_elapsed = time.perf_counter() - save_start
    total_elapsed = time.perf_counter() - total_start

    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"Samples written: {sample_index}")
    print(f"Timing extraction/write: {extraction_elapsed:.2f} s")
    print(f"Timing save/flush: {save_elapsed:.2f} s")
    print(f"Timing total: {total_elapsed:.2f} s")

    print(f"Saved X: {outdir / 'X.npy'}")
    print(f"Saved y: {outdir / 'y.npy'}")
    print(f"Saved metadata: {outdir / 'metadata.csv'}")


def get_worker_count(n_jobs, config_name):
    """
    Return number of workers used for chunk sizing.

    Parameters
    ----------
    n_jobs : int
        Joblib worker count from config. Negative values use all CPUs.
    config_name : str
        Name of the config value, used in error messages.

    Returns
    -------
    int
        Positive worker count.
    """

    if n_jobs == 0:
        raise ValueError(f"{config_name} must be non-zero")

    if n_jobs < 0:
        return os.cpu_count() or 1

    return max(1, n_jobs)


def plan_dataset_sample_specs(config, timesteps, planning_n_jobs):
    """
    Create sample specs for all requested timesteps.

    Parameters
    ----------
    config : dict
        Dataset creation config.
    timesteps : list of int
        Timesteps to process.
    planning_n_jobs : int
        Number of joblib workers used for planning.

    Returns
    -------
    sample_specs_by_timestep : dict
        Mapping from timestep to sample specification dictionaries.
    elapsed : float
        Planning wall-clock time in seconds.
    """

    print_memory_usage("before planning")
    planning_start = time.perf_counter()

    if planning_n_jobs == 1:
        sample_spec_results = [
            create_timestep_sample_specs_for_timestep(
                config=config,
                timestep=timestep,
            )
            for timestep in timesteps
        ]
    else:
        sample_spec_results = Parallel(n_jobs=planning_n_jobs)(
            delayed(create_timestep_sample_specs_for_timestep)(
                config=config,
                timestep=timestep,
            )
            for timestep in timesteps
        )

    return dict(sample_spec_results), time.perf_counter() - planning_start


def count_sample_specs_by_timestep(sample_specs_by_timestep):
    """
    Count sample specs for each planned timestep.

    Parameters
    ----------
    sample_specs_by_timestep : dict
        Mapping from timestep to sample specification dictionaries.

    Returns
    -------
    dict
        Mapping from timestep to number of samples.
    """

    return {
        int(timestep): len(sample_specs)
        for timestep, sample_specs in sample_specs_by_timestep.items()
    }


def count_sample_counts(sample_counts_by_timestep):
    """
    Count samples across timesteps.

    Parameters
    ----------
    sample_counts_by_timestep : dict
        Mapping from timestep to sample count.

    Returns
    -------
    int
        Total number of samples.
    """

    return sum(sample_counts_by_timestep.values())


def find_first_nonempty_timestep(sample_counts_by_timestep, timesteps):
    """
    Find the first timestep with at least one sample.

    Parameters
    ----------
    sample_counts_by_timestep : dict
        Mapping from timestep to sample count.
    timesteps : list of int
        Timesteps in dataset order.

    Returns
    -------
    timestep_index : int
        Index of the first timestep containing samples.
    timestep : int
        First timestep containing samples.
    """

    for timestep_index, timestep in enumerate(timesteps):
        if sample_counts_by_timestep[int(timestep)] > 0:
            return timestep_index, int(timestep)

    raise ValueError("No samples were found for the requested timesteps")


def extract_first_sample_from_specs(sample_specs):
    """
    Extract the first sample and keep its iterator alive.

    Parameters
    ----------
    sample_specs : list of dict
        Sample specifications for one timestep.

    Returns
    -------
    first_sample : dict
        First extracted sample.
    sample_iter : iterator
        Iterator for the remaining samples from the same timestep.
    """

    print_memory_usage("before first extraction")
    sample_iter = iter_timestep_sample_specs(sample_specs)

    try:
        first_sample = next(sample_iter)
    except StopIteration as error:
        raise ValueError("No samples were found for the first timestep") from error

    print_memory_usage("after first extraction")

    return first_sample, sample_iter


def write_remaining_timesteps(
    sample_specs_by_timestep,
    X,
    y,
    metadata,
    sample_index,
    timesteps,
    extraction_n_jobs,
    extraction_worker_count,
):
    """
    Extract and write all remaining timesteps.

    Parameters
    ----------
    sample_specs_by_timestep : dict
        Mapping from timestep to planned sample specifications.
    X : numpy.ndarray
        Output array for VDF samples.
    y : numpy.ndarray
        Output array for integer labels.
    metadata : list of dict
        Metadata rows accumulated for the dataset.
    sample_index : int
        Next sample index to write.
    timesteps : list of int
        Timesteps left to write.
    extraction_n_jobs : int
        Number of extraction workers.
    extraction_worker_count : int
        Positive worker count used for chunk sizing.

    Returns
    -------
    int
        Next sample index after writing.
    """

    if extraction_n_jobs == 1:
        return write_timesteps_serial(
            sample_specs_by_timestep=sample_specs_by_timestep,
            X=X,
            y=y,
            metadata=metadata,
            sample_index=sample_index,
            timesteps=timesteps,
        )

    return write_timesteps_parallel(
        sample_specs_by_timestep=sample_specs_by_timestep,
        X=X,
        y=y,
        metadata=metadata,
        sample_index=sample_index,
        timesteps=timesteps,
        extraction_n_jobs=extraction_n_jobs,
        extraction_worker_count=extraction_worker_count,
    )


def write_timesteps_serial(
    sample_specs_by_timestep,
    X,
    y,
    metadata,
    sample_index,
    timesteps,
):
    """
    Stream timestep samples into output arrays serially.

    Parameters
    ----------
    sample_specs_by_timestep : dict
        Mapping from timestep to planned sample specifications.
    X : numpy.ndarray
        Output array for VDF samples.
    y : numpy.ndarray
        Output array for integer labels.
    metadata : list of dict
        Metadata rows accumulated for the dataset.
    sample_index : int
        Next sample index to write.
    timesteps : list of int
        Timesteps to write.

    Returns
    -------
    int
        Next sample index after writing.
    """

    for timestep in timesteps:
        sample_specs = sample_specs_by_timestep.pop(int(timestep))
        sample_index = write_timestep_samples(
            X=X,
            y=y,
            metadata=metadata,
            timestep_samples=iter_timestep_sample_specs(sample_specs),
            sample_index=sample_index,
        )
        flush_and_release_memmaps(X, y)
        print_memory_usage(f"after timestep {int(timestep)} memmap release")

    return sample_index


def write_timesteps_parallel(
    sample_specs_by_timestep,
    X,
    y,
    metadata,
    sample_index,
    timesteps,
    extraction_n_jobs,
    extraction_worker_count,
):
    """
    Extract timestep samples in parallel and write them in timestep order.

    Parameters
    ----------
    sample_specs_by_timestep : dict
        Mapping from timestep to planned sample specifications.
    X : numpy.ndarray
        Output array for VDF samples.
    y : numpy.ndarray
        Output array for integer labels.
    metadata : list of dict
        Metadata rows accumulated for the dataset.
    sample_index : int
        Next sample index to write.
    timesteps : list of int
        Timesteps to write.
    extraction_n_jobs : int
        Number of extraction workers.
    extraction_worker_count : int
        Positive worker count used for chunk sizing.

    Returns
    -------
    int
        Next sample index after writing.
    """

    with tempfile.TemporaryDirectory(
        prefix="extraction_",
        dir=Path(X.filename).parent,
    ) as temp_dir:
        temp_dir = Path(temp_dir)

        for timestep_chunk in iter_chunks(timesteps, extraction_worker_count):
            chunk_start = int(timestep_chunk[0])
            chunk_end = int(timestep_chunk[-1])
            print_memory_usage(
                f"before extraction chunk {chunk_start}-{chunk_end}"
            )
            chunk_specs_by_timestep = {
                int(timestep): sample_specs_by_timestep.pop(int(timestep))
                for timestep in timestep_chunk
            }
            chunk_results = extract_timestep_chunk_parallel(
                sample_specs_by_timestep=chunk_specs_by_timestep,
                timestep_chunk=timestep_chunk,
                extraction_n_jobs=extraction_n_jobs,
                temp_dir=temp_dir,
                sample_shape=X.shape[1:],
                dtype=X.dtype,
            )
            print_memory_usage(
                f"after extraction chunk {chunk_start}-{chunk_end}"
            )

            for extracted_timestep in chunk_results:
                sample_index = write_extracted_timestep(
                    X=X,
                    y=y,
                    metadata=metadata,
                    extracted_timestep=extracted_timestep,
                    sample_index=sample_index,
                )

            print_memory_usage(f"after writing chunk {chunk_start}-{chunk_end}")
            flush_and_release_memmaps(X, y)
            print_memory_usage(
                f"after memmap release chunk {chunk_start}-{chunk_end}"
            )

    return sample_index


def extract_timestep_samples_to_temp(sample_specs, temp_dir, sample_shape, dtype):
    """
    Extract planned timestep samples into temporary memmap files.

    Parameters
    ----------
    sample_specs : list of dict
        Sample specifications for one timestep.
    temp_dir : pathlib.Path
        Directory for temporary extraction arrays.
    sample_shape : tuple of int
        Shape of one VDF sample.
    dtype : data-type
        VDF array dtype.

    Returns
    -------
    dict
        Descriptor containing temporary array paths and metadata rows.
    """

    if not sample_specs:
        return {
            "n_samples": 0,
            "metadata": [],
            "X_path": None,
            "y_path": None,
        }

    timestep = int(sample_specs[0]["timestep"])
    temp_dir = Path(temp_dir)
    X_path = temp_dir / f"timestep_{timestep}_X.npy"
    y_path = temp_dir / f"timestep_{timestep}_y.npy"

    X_temp = np.lib.format.open_memmap(
        X_path,
        mode="w+",
        dtype=dtype,
        shape=(len(sample_specs), *sample_shape),
    )
    y_temp = np.lib.format.open_memmap(
        y_path,
        mode="w+",
        dtype=np.int64,
        shape=(len(sample_specs),),
    )

    metadata = []
    sample_index = write_timestep_samples(
        X=X_temp,
        y=y_temp,
        metadata=metadata,
        timestep_samples=iter_timestep_sample_specs(sample_specs),
        sample_index=0,
    )

    if sample_index != len(sample_specs):
        raise RuntimeError(
            f"Expected to extract {len(sample_specs)} samples for timestep "
            f"{timestep}, extracted {sample_index}"
        )

    flush_and_release_memmaps(X_temp, y_temp)

    return {
        "n_samples": sample_index,
        "metadata": metadata,
        "X_path": str(X_path),
        "y_path": str(y_path),
    }


def write_extracted_timestep(X, y, metadata, extracted_timestep, sample_index):
    """
    Copy one extracted temporary timestep into the final dataset arrays.

    Parameters
    ----------
    X : numpy.ndarray
        Final output array for VDF samples.
    y : numpy.ndarray
        Final output array for integer labels.
    metadata : list of dict
        Metadata rows accumulated for the final dataset.
    extracted_timestep : dict
        Descriptor returned by ``extract_timestep_samples_to_temp``.
    sample_index : int
        Index where writing should start.

    Returns
    -------
    int
        Next sample index after writing this timestep.
    """

    n_samples = int(extracted_timestep["n_samples"])
    if n_samples == 0:
        return sample_index

    timestep_metadata = extracted_timestep["metadata"]
    if len(timestep_metadata) != n_samples:
        raise RuntimeError(
            f"Expected {n_samples} metadata rows, got {len(timestep_metadata)}"
        )

    X_temp = np.load(extracted_timestep["X_path"], mmap_mode="r")
    y_temp = np.load(extracted_timestep["y_path"], mmap_mode="r")
    write_end = sample_index + n_samples

    X[sample_index:write_end] = X_temp[:n_samples]
    y[sample_index:write_end] = y_temp[:n_samples]

    for metadata_row in timestep_metadata:
        output_row = dict(metadata_row)
        output_row["sample_index"] = sample_index + int(
            metadata_row["sample_index"]
        )
        metadata.append(output_row)

    release_memmap_pages(X_temp)
    release_memmap_pages(y_temp)

    return write_end


def extract_timestep_chunk_parallel(
    sample_specs_by_timestep,
    timestep_chunk,
    extraction_n_jobs,
    temp_dir,
    sample_shape,
    dtype,
):
    """
    Extract a timestep chunk with joblib.

    Parameters
    ----------
    sample_specs_by_timestep : dict
        Mapping from timestep to planned sample specifications.
    timestep_chunk : list of int
        Timesteps in the chunk.
    extraction_n_jobs : int
        Number of extraction workers.
    temp_dir : pathlib.Path
        Directory for temporary extraction arrays.
    sample_shape : tuple of int
        Shape of one VDF sample.
    dtype : data-type
        VDF array dtype.

    Returns
    -------
    iterable
        Extracted timestep descriptors in timestep order.
    """

    parallel = create_extraction_parallel(extraction_n_jobs)

    return parallel(
        delayed(extract_timestep_samples_to_temp)(
            sample_specs=sample_specs_by_timestep[int(timestep)],
            temp_dir=temp_dir,
            sample_shape=sample_shape,
            dtype=dtype,
        )
        for timestep in timestep_chunk
    )


def create_extraction_parallel(extraction_n_jobs):
    """
    Create a joblib ``Parallel`` object for extraction.

    Parameters
    ----------
    extraction_n_jobs : int
        Number of extraction workers.

    Returns
    -------
    joblib.Parallel
        Parallel executor.
    """

    try:
        return Parallel(
            n_jobs=extraction_n_jobs,
            return_as="generator",
        )
    except TypeError:
        return Parallel(n_jobs=extraction_n_jobs)
