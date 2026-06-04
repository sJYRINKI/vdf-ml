#python scripts/make_dataset.py --config configs/create_dataset.yaml --start-timestep 3408 --n-timesteps 100 --dataset-kind train

import argparse
import sys
from pathlib import Path
import numpy as np
import analysator as pt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.config import load_config
from src.timesteps import create_timestep_list, create_timestep_path
from src.vdf_helpers import get_cellid_with_vdf
from src.vdf_extract import extract_vdf
from src.dataset_items import iter_labeled_coords
from src.dataset_io import create_dataset_output_dir, save_dataset

def main(config_path, start_timestep, n_timesteps, dataset_kind):
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

    X = []
    y = []
    metadata = []

    sample_index = 0

    for timestep in timesteps:
        file_location = create_timestep_path(
            path_template=config["file_template"],
            timestep=timestep
        )

        print(f"Timestep: {timestep}")
        print(f"File: {file_location}")

        reader = pt.vlsvfile.VlsvReader(str(file_location))

        simulation_time = reader.read_parameter("time")

        print(simulation_time)

        for class_name, label, coord_re in iter_labeled_coords(config):
            print(f"Sample index: {sample_index}")
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
            )

            X.append(vdf)
            y.append(label)

            metadata.append(
                {
                    "sample_index": sample_index,
                    "timestep": int(timestep),
                    "simulation_time": simulation_time,
                    "cid": int(cid),
                    "label": int(label),
                    "class_name": class_name,
                    "x_re": float(coord_re[0]),
                    "y_re": float(coord_re[1]),
                    "z_re": float(coord_re[2]),
                    "file_location": str(file_location),
                }
            )

            print(f"VDF shape: {vdf.shape}")
            print(f"VDF dtype: {vdf.dtype}")
            print(f"VDF min: {vdf.min()}")
            print(f"VDF max: {vdf.max()}")

            sample_index += 1

    X = np.stack(X).astype(np.float32)
    y = np.array(y, dtype=np.int64)

    save_dataset(
        outdir=outdir,
        X=X,
        y=y,
        metadata=metadata,
    )

    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"y: {y}")
    



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