import argparse
import sys
from pathlib import Path

PRPJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PRPJECT_ROOT))

from src.config import load_config
from src.timesteps import create_timestep_list, create_file_location

def main(config_path, start_timestep, n_timesteps):
    config = load_config(config_path)

    timesteps = create_timestep_list(
        start_timestep=start_timestep,
        n_timesteps=n_timesteps
    )

    print("success")
    print(config["population"])

    file_template = config["file_template"]

    for timestep in timesteps:
        file_location = create_file_location(
            file_template=file_template,
            timestep=timestep
        )

        print(f"{timestep}->{file_location}")

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

    args = parser.parse_args()

    main(
        config_path=args.config,
        start_timestep=args.start_timestep,
        n_timesteps=args.n_timesteps
    )