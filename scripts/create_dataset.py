#python scripts/create_dataset.py --config configs/create_dataset.yaml --start-timestep 3408 --n-timesteps 100 --dataset-kind train

import argparse
import os
import sys
import time
from pathlib import Path
import numpy as np
from joblib import Parallel, delayed

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.config import load_config
from src.timesteps import create_timestep_list
from src.dataset_helpers import (
    create_timestep_sample_specs_for_timestep,
    iter_chunks,
    process_timestep_sample_specs,
    write_timestep_samples,
)
from src.dataset_io import (
    create_dataset_output_dir,
    create_memmap_dataset,
    save_metadata,
)

def main(config_path, start_timestep, n_timesteps, dataset_kind):
    """
    Create and save a labeled VDF dataset.

    Parameters
    ----------
    config_path : str
        Path to dataset creation YAML config.
    start_timestep : int
        First timestep to include.
    n_timesteps : int
        Number of consecutive timesteps to process.
    dataset_kind : {"train", "test"}
        Output dataset split name.
    """

    total_start = time.perf_counter()
    config = load_config(config_path)

    timesteps = create_timestep_list(
        start_timestep=start_timestep,
        n_timesteps=n_timesteps
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
    n_jobs = int(creation_config.get("n_jobs", 1))
    if n_jobs == 0:
        raise ValueError("creation.n_jobs must be non-zero")

    if n_jobs < 0:
        worker_count = os.cpu_count() or 1
    else:
        worker_count = max(1, n_jobs)

    planning_start = time.perf_counter()

    if n_jobs == 1:
        sample_specs_by_timestep = dict(
            create_timestep_sample_specs_for_timestep(
                config=config,
                timestep=timestep,
            )
            for timestep in timesteps
        )
    else:
        sample_spec_results = Parallel(n_jobs=n_jobs)(
            delayed(create_timestep_sample_specs_for_timestep)(
                config=config,
                timestep=timestep,
            )
            for timestep in timesteps
        )
        sample_specs_by_timestep = dict(sample_spec_results)

    planning_elapsed = time.perf_counter() - planning_start
    n_samples = sum(
        len(sample_specs)
        for sample_specs in sample_specs_by_timestep.values()
    )

    print(f"Samples: {n_samples}")
    print(f"Timing planning: {planning_elapsed:.2f} s")

    if n_samples == 0:
        raise ValueError("No samples were found for the requested timesteps")

    extraction_start = time.perf_counter()

    metadata = []
    sample_index = 0

    first_samples = []
    first_sample_timestep_index = None

    for timestep_index, timestep in enumerate(timesteps):
        first_samples = process_timestep_sample_specs(
            sample_specs_by_timestep[timestep]
        )

        if first_samples:
            first_sample_timestep_index = timestep_index
            break

    X, y = create_memmap_dataset(
        outdir=outdir,
        n_samples=n_samples,
        sample_shape=first_samples[0]["vdf"].shape,
        dtype=np.float32,
    )

    sample_index = write_timestep_samples(
        X=X,
        y=y,
        metadata=metadata,
        timestep_samples=first_samples,
        sample_index=sample_index,
    )

    remaining_timesteps = timesteps[first_sample_timestep_index + 1:]

    if n_jobs == 1:
        for timestep in remaining_timesteps:
            timestep_samples = process_timestep_sample_specs(
                sample_specs_by_timestep[timestep]
            )
            sample_index = write_timestep_samples(
                X=X,
                y=y,
                metadata=metadata,
                timestep_samples=timestep_samples,
                sample_index=sample_index,
            )
    else:
        for timestep_chunk in iter_chunks(remaining_timesteps, worker_count):
            chunk_results = Parallel(n_jobs=n_jobs)(
                delayed(process_timestep_sample_specs)(
                    sample_specs_by_timestep[timestep]
                )
                for timestep in timestep_chunk
            )

            for timestep_samples in chunk_results:
                sample_index = write_timestep_samples(
                    X=X,
                    y=y,
                    metadata=metadata,
                    timestep_samples=timestep_samples,
                    sample_index=sample_index,
                )

    extraction_elapsed = time.perf_counter() - extraction_start
    save_start = time.perf_counter()

    X.flush()
    y.flush()

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

if __name__=="__main__":
    parser = argparse.ArgumentParser(
        description="Create VDF dataset from Vlasiator VLSV files"
    )

    parser.add_argument(
        "--config",
        required=True,
        help="Path to YAML config file"
    )

    parser.add_argument(
        "--start-timestep",
        type=int,
        required=True,
        help="First timestep"
    )

    parser.add_argument(
        "--n-timesteps",
        type=int,
        required=True,
        help="Number of timesteps to process starting from start timestep"
    )

    parser.add_argument(
        "--dataset-kind",
        choices=["train", "test"],
        required=True,
        help="Save the complete dataset under the configured train or test directory."
    )

    args = parser.parse_args()

    main(
        config_path=args.config,
        start_timestep=args.start_timestep,
        n_timesteps=args.n_timesteps,
        dataset_kind=args.dataset_kind,
    )
