#python scripts/train_pytorch_convolutional_neural_network_classifier.py --config configs/train_pytorch_convolutional_neural_network_classifier.yaml --dataset-id 3408_100 --model-id v1.0

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.config import load_config
from src.pytorch_cnn import train_pytorch_convolutional_neural_network_classifier


def main(config_path, dataset_id, model_id):
    """
    Train a PyTorch CNN VDF classifier.

    Parameters
    ----------
    config_path : str
        Path to training config.
    dataset_id : str
        Dataset identifier.
    model_id : str
        Model identifier.
    """

    config = load_config(config_path)
    train_pytorch_convolutional_neural_network_classifier(
        config=config,
        dataset_id=dataset_id,
        model_id=model_id,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a PyTorch CNN classifier on VDF data."
    )
    parser.add_argument("--config", required=True, help="Path to CNN config.")
    parser.add_argument("--dataset-id", required=True, help="Dataset identifier.")
    parser.add_argument("--model-id", required=True, help="Model identifier.")
    args = parser.parse_args()

    main(
        config_path=args.config,
        dataset_id=args.dataset_id,
        model_id=args.model_id,
    )
