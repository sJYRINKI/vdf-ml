import os
import time

import numpy as np
from joblib import Parallel, delayed

from src.dataset_sampling import (
    create_timestep_sample_specs_for_timestep,
    iter_chunks,
    iter_timestep_sample_specs,
    print_memory_usage,
    process_timestep_sample_specs,
    write_timestep_samples,
)
from src.dataset_io import (
    create_dataset_output_dir,
    create_memmap_dataset,
    flush_and_release_memmaps,
    save_metadata,
)
from src.timesteps import create_timestep_list


def create_dataset(config, start_timestep, n_timesteps, dataset_kind):
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

    sample_counts_by_timestep, planning_elapsed = plan_dataset_sample_counts(
        config=config,
        timesteps=timesteps,
        planning_n_jobs=planning_n_jobs,
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
    _, first_sample_specs = create_timestep_sample_specs_for_timestep(
        config=config,
        timestep=first_timestep,
    )
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
    flush_and_release_memmaps(X, y)
    print_memory_usage("after first timestep memmap release")

    remaining_timesteps = timesteps[first_timestep_index + 1:]
    sample_index = write_remaining_timesteps(
        config=config,
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


def plan_dataset_sample_counts(config, timesteps, planning_n_jobs):
    """
    Count sample specs for all requested timesteps.

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
    sample_counts_by_timestep : dict
        Mapping from timestep to number of samples.
    elapsed : float
        Planning wall-clock time in seconds.
    """

    print_memory_usage("before planning")
    planning_start = time.perf_counter()

    if planning_n_jobs == 1:
        sample_count_results = [
            count_timestep_sample_specs_for_timestep(
                config=config,
                timestep=timestep,
            )
            for timestep in timesteps
        ]
    else:
        sample_count_results = Parallel(n_jobs=planning_n_jobs)(
            delayed(count_timestep_sample_specs_for_timestep)(
                config=config,
                timestep=timestep,
            )
            for timestep in timesteps
        )

    return dict(sample_count_results), time.perf_counter() - planning_start


def count_timestep_sample_specs_for_timestep(config, timestep):
    """
    Count sample specs for one timestep without retaining them.

    Parameters
    ----------
    config : dict
        Dataset creation config.
    timestep : int
        Timestep to count.

    Returns
    -------
    tuple
        Pair of ``(timestep, n_samples)``.
    """

    timestep, sample_specs = create_timestep_sample_specs_for_timestep(
        config=config,
        timestep=timestep,
    )

    return int(timestep), len(sample_specs)


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
    config,
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
    config : dict
        Dataset creation config.
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
            config=config,
            X=X,
            y=y,
            metadata=metadata,
            sample_index=sample_index,
            timesteps=timesteps,
        )

    return write_timesteps_parallel(
        config=config,
        X=X,
        y=y,
        metadata=metadata,
        sample_index=sample_index,
        timesteps=timesteps,
        extraction_n_jobs=extraction_n_jobs,
        extraction_worker_count=extraction_worker_count,
    )


def write_timesteps_serial(
    config,
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
    config : dict
        Dataset creation config.
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
        _, sample_specs = create_timestep_sample_specs_for_timestep(
            config=config,
            timestep=timestep,
        )
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
    config,
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
    config : dict
        Dataset creation config.
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

    for timestep_chunk in iter_chunks(timesteps, extraction_worker_count):
        chunk_start = int(timestep_chunk[0])
        chunk_end = int(timestep_chunk[-1])
        print_memory_usage(
            f"before extraction chunk {chunk_start}-{chunk_end}"
        )
        chunk_results = extract_timestep_chunk_parallel(
            config=config,
            timestep_chunk=timestep_chunk,
            extraction_n_jobs=extraction_n_jobs,
        )
        print_memory_usage(
            f"after extraction chunk {chunk_start}-{chunk_end}"
        )

        for timestep_samples in chunk_results:
            sample_index = write_timestep_samples(
                X=X,
                y=y,
                metadata=metadata,
                timestep_samples=timestep_samples,
                sample_index=sample_index,
            )

        print_memory_usage(f"after writing chunk {chunk_start}-{chunk_end}")
        flush_and_release_memmaps(X, y)
        print_memory_usage(
            f"after memmap release chunk {chunk_start}-{chunk_end}"
        )

    return sample_index


def extract_timestep_samples(config, timestep):
    """
    Create sample specs for one timestep and extract its samples.

    Parameters
    ----------
    config : dict
        Dataset creation config.
    timestep : int
        Timestep to extract.

    Returns
    -------
    list of dict
        Extracted sample dictionaries.
    """

    _, sample_specs = create_timestep_sample_specs_for_timestep(
        config=config,
        timestep=timestep,
    )

    return process_timestep_sample_specs(sample_specs)


def extract_timestep_chunk_parallel(
    config,
    timestep_chunk,
    extraction_n_jobs,
):
    """
    Extract a timestep chunk with joblib.

    Parameters
    ----------
    config : dict
        Dataset creation config.
    timestep_chunk : list of int
        Timesteps in the chunk.
    extraction_n_jobs : int
        Number of extraction workers.

    Returns
    -------
    iterable
        Extracted sample lists in timestep order.
    """

    parallel = create_extraction_parallel(extraction_n_jobs)

    return parallel(
        delayed(extract_timestep_samples)(
            config=config,
            timestep=timestep,
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
