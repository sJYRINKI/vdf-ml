#python scripts/backfill_dataset_velocity_grid.py --dataset-dir data/train/timesteps_2900_1400_x_manual_o_manual

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.dataset_io import backfill_dataset_velocity_grid


def main(
    dataset_dir,
    model_dirs=None,
    overwrite=False,
):
    """
    Backfill dataset and model velocity-grid information.

    Parameters
    ----------
    dataset_dir : str
        Dataset directory containing ``X.npy`` and ``metadata.csv``.
    model_dirs : iterable of str, optional
        Model directories whose ``preprocessing.npz`` files should be updated.
    overwrite : bool, optional
        Whether conflicting existing velocity-grid information may be replaced.
    """

    result = backfill_dataset_velocity_grid(
        dataset_dir=dataset_dir,
        model_dirs=model_dirs,
        overwrite=overwrite,
    )
    print(f"Saved: {result['velocity_grid_path']}")
    for preprocessing_path in result["preprocessing_paths"]:
        print(f"Updated: {preprocessing_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Backfill the training velocity grid for a dataset and models."
        )
    )
    parser.add_argument(
        "--dataset-dir",
        required=True,
        help="Dataset directory containing X.npy and metadata.csv.",
    )
    parser.add_argument(
        "--model-dir",
        action="append",
        default=[],
        help=(
            "Model directory whose preprocessing.npz should be updated. "
            "May be repeated."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace conflicting existing velocity-grid information.",
    )
    args = parser.parse_args()

    main(
        dataset_dir=args.dataset_dir,
        model_dirs=args.model_dir,
        overwrite=args.overwrite,
    )
