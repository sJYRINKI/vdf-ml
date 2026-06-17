#python scripts/predict_logistic_regression_coordinate.py --config configs/predict_coordinate_logistic_regression.yaml --timestep 4000 --model-id v1.0 --coord-re -12 0 0

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.config import load_config
from src.coordinate_prediction import predict_coordinate
from src.model_io import load_logistic_regression_model


def main(config_path, timestep, model_id, coord_re):
    """
    Predict one VDF from coordinates.

    Parameters
    ----------
    config_path : str
        Path to coordinate prediction config.
    timestep : int
        VLSV timestep to read.
    model_id : str
        Trained model identifier.
    coord_re : array-like of float
        Coordinate in Earth radii, given as ``[x, y, z]``.
    """

    config = load_config(config_path)
    result = predict_coordinate(
        config=config,
        timestep=timestep,
        model_id=model_id,
        coord_re=coord_re,
        load_model=load_logistic_regression_model,
    )

    print(f"Saved plot: {result['output_plot_path']}")
    print(f"Predicted class: {result['predicted_class_name']}")
    print(f"{result['score_name']}: {result['prediction_score']}")
    print(result["model_classes"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Predict one VDF from coordinates using a trained logistic regression classifier."
    )

    parser.add_argument(
        "--config",
        required=True,
        help="Path to coordinate prediction config.",
    )

    parser.add_argument(
        "--timestep",
        required=True,
        help="VLSV timestep to read, for example 3500.",
    )

    parser.add_argument(
        "--model-id",
        required=True,
        help="Trained model identifier, for example v1.0.",
    )

    parser.add_argument(
        "--coord-re",
        type=float,
        nargs=3,
        required=True,
        metavar=("X_RE", "Y_RE", "Z_RE"),
        help="Coordinate in Earth radii.",
    )

    args = parser.parse_args()

    main(
        config_path=args.config,
        timestep=args.timestep,
        model_id=args.model_id,
        coord_re=args.coord_re,
    )
