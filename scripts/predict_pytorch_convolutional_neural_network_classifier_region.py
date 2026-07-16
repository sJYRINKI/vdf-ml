# python scripts/predict_pytorch_convolutional_neural_network_classifier_region.py --config configs/predict_region_pytorch_convolutional_neural_network_classifier.yaml --start-timestep 1300 --n-timesteps 100 --model-id v68.0 --file-source 3d --device cuda --n-jobs 1

import argparse
import sys
from functools import partial
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.config import load_config
from src.coordinate_prediction import predict_region_timesteps
from src.model_io import load_pytorch_convolutional_neural_network_classifier_model


def main(
    config_path,
    start_timestep,
    n_timesteps,
    model_id,
    n_jobs=None,
    file_source=None,
    device="cpu",
):
    """
    Predict VDF cells in a spatial region for multiple timesteps.

    Parameters
    ----------
    config_path : str
        Path to region prediction config.
    start_timestep : int
        First VLSV timestep to read.
    n_timesteps : int
        Number of consecutive timesteps to predict.
    model_id : str
        Trained model identifier.
    n_jobs : int, optional
        Override for ``prediction.n_jobs``. GPU prediction requires one job.
    file_source : str, optional
        Name of file template source to use.
    device : str, optional
        Device used for CNN prediction, such as ``cpu`` or ``cuda``.
    """

    config = load_config(config_path)
    prediction_config = config.setdefault("prediction", {})
    if n_jobs is not None:
        prediction_config["n_jobs"] = int(n_jobs)

    prediction_n_jobs = int(prediction_config.get("n_jobs", 1))
    device_type = str(device).strip().lower().split(":", maxsplit=1)[0]
    if device_type != "cpu" and prediction_n_jobs != 1:
        raise ValueError("GPU prediction requires --n-jobs 1")

    results = predict_region_timesteps(
        config=config,
        start_timestep=start_timestep,
        n_timesteps=n_timesteps,
        model_id=model_id,
        load_model=partial(
            load_pytorch_convolutional_neural_network_classifier_model,
            device=device,
        ),
        file_source=file_source,
    )

    total_cells = sum(result["n_selected_cells"] for result in results)
    print(f"Predicted timesteps: {len(results)}")
    print(f"Selected VDF cells: {total_cells}")
    for result in results:
        print(
            f"{result['timestep']}: "
            f"{result['n_selected_cells']} cells -> {result['output_path']}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Predict VDF cells inside a spatial region using a trained "
            "PyTorch CNN classifier."
        )
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to region prediction config.",
    )
    parser.add_argument(
        "--start-timestep",
        type=int,
        required=True,
        help="First VLSV timestep to read, for example 1300.",
    )
    parser.add_argument(
        "--n-timesteps",
        type=int,
        required=True,
        help="Number of consecutive timesteps to predict.",
    )
    parser.add_argument(
        "--model-id",
        required=True,
        help="Trained model identifier, for example v68.0.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=None,
        help=(
            "Override prediction.n_jobs from the config. GPU prediction "
            "requires --n-jobs 1."
        ),
    )
    parser.add_argument(
        "--file-source",
        default=None,
        help="File source from config file_templates, for example 2d or 3d.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device used for CNN prediction, for example cpu or cuda.",
    )
    args = parser.parse_args()

    main(
        config_path=args.config,
        start_timestep=args.start_timestep,
        n_timesteps=args.n_timesteps,
        model_id=args.model_id,
        n_jobs=args.n_jobs,
        file_source=args.file_source,
        device=args.device,
    )
