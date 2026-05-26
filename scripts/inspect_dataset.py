import argparse
import sys
from pathlib import Path

PRPJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PRPJECT_ROOT))

from src.dataset_io import load_dataset

def main(dataset_dir):
    X, y, metadata = load_dataset(dataset_dir)

    print("Dataset loaded")
    print(dataset_dir)
    print(X.shape)

if __name__=="__main__":
    parser = argparse.ArgumentParser(
        description="Load a saved VDF dataset."
    )

    parser.add_argument(
        "--dataset-dir",
        required=True,
        help="Directory containing vdf.npz and metadata.csv"
    )

    args = parser.parse_args()

    main(dataset_dir=args.dataset_dir)