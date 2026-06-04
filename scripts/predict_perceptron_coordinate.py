#python scripts/predict_perceptron_coordinate.py --config configs/predict_coordinate_perceptron.yaml --timestep 4000 --model-id v1.0 --coord-re -12 0 0

import argparse
import sys
from pathlib import Path
import pandas as pd
import analysator as pt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.config import load_config
from src.dataset_plot import plot_vdf_xz_slice
from src.model_io import load_perceptron_model
from src.perceptron_features import create_perceptron_features
from src.timesteps import create_path
from src.vdf_extract import extract_vdf
from src.vdf_helpers import get_vdf_plot_parameters_from_file, get_cellid_with_vdf
from src.vdf_helpers import create_coordinate_name

def main(config_path, timestep, model_id, coord_re):
    config = load_config(config_path)

    file_location = create_path(
        path_template=config["file_template"],
        timestep=int(timestep)
    )

    model_dir = create_path(
        path_template=config["model_dir"],
        model_id=model_id,
        timestep=int(timestep)
    )

    output_dir = create_path(
        path_template=config["output_dir"],
        model_id=model_id,
        timestep=int(timestep)
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    labels = config.get("labels")

    label_to_class = {
        int(label): class_name
        for class_name, label in labels.items()
    }

    plot_config = config["plot"]

    vdflim = float(plot_config.get("vdflim", 2000000.0))

    reader = pt.vlsvfile.VlsvReader(file_location)

    cid = get_cellid_with_vdf(
        reader=reader,
        coord_re=coord_re,
    )

    vdf = extract_vdf(
        file_location=file_location,
        cid=cid,
    )

    model, preprocessing = load_perceptron_model(model_dir)

    downsample_factor = int(preprocessing["downsample_factor"])
    log_eps = float(preprocessing["log_eps"])

    features = create_perceptron_features(
        X=vdf[None, ...],
        downsample_factor=downsample_factor,
        log_eps=log_eps,
    )

    predicted_label = int(model.predict(features)[0])
    decision_score = float(model.decision_function(features)[0])

    predicted_class_name = label_to_class[predicted_label]

    extent, dv, threshold = get_vdf_plot_parameters_from_file(
        file_location=file_location,
        cid=cid,
        vdf_shape=vdf.shape,
    )

    metadata_row = {
        "timestep": int(timestep),
        "cid": int(cid),
        "x_re": float(coord_re[0]),
        "y_re": float(coord_re[1]),
        "z_re": float(coord_re[2]),
        "file_location": str(file_location),
    }

    coord_name = create_coordinate_name(coord_re)

    output_plot_path = output_dir / f"prediction_{coord_name}_xz.png"

    plot_vdf_xz_slice(
            vdf=vdf,
            y_label=None,
            metadata_row=metadata_row,
            extent=extent,
            output_path=output_plot_path,
            dv=dv,
            threshold=threshold,
            vdflim=vdflim,
            predicted_class_name=predicted_class_name,
            decision_score=decision_score,
        )

    print(f"Saved plot: {output_plot_path}")
    print(model.classes_)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Predict one VDF from coordinates using a trained perceptron."
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