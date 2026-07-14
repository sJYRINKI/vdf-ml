# python scripts/backfill_dataset_metadata.py --dataset-dir data/train/timesteps_2900_1400_x_physical_o_union

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.dataset_metadata import backfill_dataset_spatial_metadata


def main(dataset_dir, n_jobs=4):
    """
    Add VDF-cell coordinates and point distances to dataset metadata.

    Parameters
    ----------
    dataset_dir : str
        Dataset directory containing ``metadata.csv``.
    n_jobs : int, optional
        Number of parallel processes used to read source VLSV files.
    """

    metadata_path = backfill_dataset_spatial_metadata(
        dataset_dir=dataset_dir,
        n_jobs=n_jobs,
    )
    print(f"Updated: {metadata_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Add VDF-cell coordinates and X/O-point distances to metadata.csv."
        )
    )
    parser.add_argument(
        "--dataset-dir",
        required=True,
        help="Dataset directory whose metadata.csv is updated in place.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=4,
        help="Parallel VLSV file readers (default: 4; use 1 for serial).",
    )
    args = parser.parse_args()

    main(dataset_dir=args.dataset_dir, n_jobs=args.n_jobs)
