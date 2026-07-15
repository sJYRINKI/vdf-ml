#python scripts/create_dataset.py --config configs/create_dataset.yaml --start-timestep 3408 --n-timesteps 100 --dataset-kind train

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.config import load_config
from src.dataset_creation import apply_dataset_creation_overrides, create_dataset


def main(
    config_path,
    start_timestep,
    n_timesteps,
    dataset_kind,
    x_selection_method=None,
    o_selection_method=None,
):
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
    x_selection_method : str, optional
        X-point selection method override.
    o_selection_method : str, optional
        O-point selection method override.
    """

    config = load_config(config_path)
    output_suffix = apply_dataset_creation_overrides(
        config=config,
        x_selection_method=x_selection_method,
        o_selection_method=o_selection_method,
    )
    create_dataset(
        config=config,
        start_timestep=start_timestep,
        n_timesteps=n_timesteps,
        dataset_kind=dataset_kind,
        output_suffix=output_suffix,
    )


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
    parser.add_argument(
        "--x-selection-method",
        choices=[
            "manual",
            "physical",
            "union",
            "union_physical_priority",
            "consensus",
        ],
        default=None,
        help="Optional X-point selection method override.",
    )
    parser.add_argument(
        "--o-selection-method",
        choices=[
            "manual",
            "physical",
            "union",
            "union_physical_priority",
            "consensus",
        ],
        default=None,
        help="Optional O-point selection method override.",
    )

    args = parser.parse_args()

    main(
        config_path=args.config,
        start_timestep=args.start_timestep,
        n_timesteps=args.n_timesteps,
        dataset_kind=args.dataset_kind,
        x_selection_method=args.x_selection_method,
        o_selection_method=args.o_selection_method,
    )
