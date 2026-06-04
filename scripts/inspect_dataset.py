#python scripts/inspect_dataset.py --dataset-dir data/processed/timesteps_3408_100

import argparse
import sys
from pathlib import Path

PRPJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PRPJECT_ROOT))

from src.dataset_io import load_dataset
from src.dataset_inspect import print_dataset_info, print_vdf_statisctics

def main(dataset_dir, batch_size):
    X, y, metadata = load_dataset(dataset_dir, mmap=True)

    print_dataset_info(X, y, metadata)
    print_vdf_statisctics(X, batch_size=batch_size)

if __name__=="__main__":
    parser = argparse.ArgumentParser(
        description="Load a saved VDF dataset."
    )

    parser.add_argument(
        "--dataset-dir",
        required=True,
        help="Directory containing vdf.npz and metadata.csv"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size used for statistics."
    )

    args = parser.parse_args()

    main(
        dataset_dir=args.dataset_dir, 
        batch_size=args.batch_size,
    )