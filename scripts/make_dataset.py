import argparse
import sys
from pathlib import Path
import analysator as pt

PRPJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PRPJECT_ROOT))

from src.config import load_config
from src.timesteps import create_timestep_list, create_file_location
from src.vdf_helpers import get_cellid_with_vdf
from src.vdf_extract import extract_vdf

def main(config_path, start_timestep, n_timesteps):
    config = load_config(config_path)

    timesteps = create_timestep_list(
        start_timestep=start_timestep,
        n_timesteps=n_timesteps
    )

    first_timestep = timesteps[0]

    file_location = create_file_location(
        file_template=config["file_template"],
        timestep=first_timestep
    )

    test_coord_re = config["test_coord_re"]

    print(f"First timestep: {first_timestep}")
    print(f"First file: {file_location}")
    print(f"Test coordinate RE: {test_coord_re}")

    reader = pt.vlsvfile.VlsvReader(str(file_location))

    cid = get_cellid_with_vdf(
        reader=reader,
        coord_re=test_coord_re,
    )

    print(f"Cell ID: {cid}")

    vdf = extract_vdf(
        file=file_location,
        cid=int(cid)
    )

    print(f"VDF shape: {vdf.shape}")
    print(f"VDF dtype: {vdf.dtype}")
    print(f"VDF min: {vdf.min()}")
    print(f"VDF max: {vdf.max()}")

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