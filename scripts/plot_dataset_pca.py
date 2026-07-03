#python scripts/plot_dataset_pca.py --config configs/plot_dataset_pca.yaml --timestep 3408_100 --pca-id v0.01
import argparse
import os
import sys
from pathlib import Path

os.environ["PTNOLATEX"] = "1"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.config import load_config
from src.dataset_pca import plot_dataset_pca


def main(config_path, timestep, pca_id=None):
    """
    Create PCA plots and metrics for a saved VDF dataset.

    Parameters
    ----------
    config_path : str
        Path to PCA plotting YAML config.
    timestep : str
        Dataset timestep identifier.
    pca_id : str, optional
        PCA output version identifier.
    """

    config = load_config(config_path)
    plot_dataset_pca(
        config=config,
        timestep=timestep,
        pca_id=pca_id,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create PCA plots and metrics for a saved VDF dataset."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to PCA plotting YAML config file.",
    )
    parser.add_argument(
        "--timestep",
        required=True,
        help="Dataset timestep identifier.",
    )
    parser.add_argument(
        "--pca-id",
        default=None,
        help="PCA output version identifier, for example v0.01.",
    )
    args = parser.parse_args()

    main(
        config_path=args.config,
        timestep=args.timestep,
        pca_id=args.pca_id,
    )
