#python scripts/train_autoencoder.py --config configs/train_autoencoder.yaml --dataset-id 3000_1300 --model-id v0.1

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.autoencoder import train_autoencoder
from src.config import load_config


def main(config_path, dataset_id, model_id):
    """
    Train a deterministic VDF slice autoencoder.

    Parameters
    ----------
    config_path : str
        Path to autoencoder training config.
    dataset_id : str
        Dataset identifier.
    model_id : str
        Model identifier.
    """

    config = load_config(config_path)
    train_autoencoder(
        config=config,
        dataset_id=dataset_id,
        model_id=model_id,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a deterministic autoencoder on VDF xz slices."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to autoencoder training config.",
    )
    parser.add_argument(
        "--dataset-id",
        required=True,
        help="Dataset identifier.",
    )
    parser.add_argument(
        "--model-id",
        required=True,
        help="Model identifier.",
    )
    args = parser.parse_args()

    main(
        config_path=args.config,
        dataset_id=args.dataset_id,
        model_id=args.model_id,
    )
