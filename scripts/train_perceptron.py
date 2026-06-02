#python scripts/train_perceptron.py --config configs/train_perceptron.yaml --timestep 3408_100
import argparse
import sys
from pathlib import Path
import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import joblib

PRPJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PRPJECT_ROOT))

from src.config import load_config
from src.dataset_io import load_dataset
from src.perceptron_features import (
    create_perceptron_features,
    standardize_features,
)
from src.timesteps import create_timestep_path

def main(config_path, timestep):
    config = load_config(config_path)

    dataset_dir = create_timestep_path(
        path_template=config["dataset_dir"],
        timestep=timestep
    )

    output_dir = create_timestep_path(
        path_template=config["output_dir"],
        timestep=timestep
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / "perceptron.joblib"
    preprocessing_path = output_dir / "perceptron_preoprocessing.npz"

    downsample_factor = int(config["model"].get("downsample_factor", 8))
    test_size = float(config["model"].get("test_size", 0.25))
    random_state = int(config["model"].get("random_state", 1234))

    X, y, metadata = load_dataset(dataset_dir)

    print(X.shape)
    print(y.shape)

    features = create_perceptron_features(
        X,
        downsample_factor=downsample_factor
    )

    features, features_mean, features_std = standardize_features(features)

    print(features.shape)
    print(features_mean)
    print(features_std)

    X_train, X_test, y_train, y_test = train_test_split(
        features,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    model = Perceptron(
        max_iter=1000,
        random_state=random_state
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    print("Perceptron results")
    print("\n")
    print(f"Accuracy: {accuracy}")
    print("\n")
    print(classification_report(y_test, y_pred))

    joblib.dump(model, model_path)

    np.savez(
        preprocessing_path,
        features_mean=features_mean,
        features_std=features_std,
        downsample_factor=downsample_factor,
    )

    print(model_path)

if __name__=="__main__":
    parser = argparse.ArgumentParser(
        description="Train a one-neuron perceptron on VDF data."
    )

    parser.add_argument(
        "--config",
        required=True,
        help="Path to perceptron training config"
    )

    parser.add_argument(
        "--timestep",
        required=True,
        help="Dataset timestep identifier."
    )

    args = parser.parse_args()

    main(
        config_path=args.config,
        timestep=args.timestep,
    )
