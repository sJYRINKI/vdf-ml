#python scripts/train_perceptron.py --config configs/train_perceptron.yaml --timestep 3408_100
import argparse
import sys
from pathlib import Path
import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import pandas as pd

PRPJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PRPJECT_ROOT))

from src.config import load_config
from src.dataset_io import load_dataset
from src.perceptron_features import (
    create_perceptron_features,
    standardize_features,
)
from src.timesteps import create_timestep_path
from src.model_evaluation import create_predictions_dataframe
from src.model_split import split_by_timestep

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

    features_config = config["features"]
    model_config = config["model"]

    downsample_factor = int(features_config.get("downsample_factor", 8))
    max_iter = int(model_config.get("downsample_factor", 8))

    X, y, metadata = load_dataset(dataset_dir)

    print(X.shape)
    print(y.shape)
    print(metadata.shape)

    features = create_perceptron_features(
        X,
        downsample_factor=downsample_factor
    )

    features, features_mean, features_std = standardize_features(features)

    print(features.shape)
    print(features_mean)
    print(features_std)

    train_indices, test_indices, train_timesteps, test_timesteps = split_by_timestep(
        metadata=metadata
    )

    X_train_raw = features[train_indices]
    X_test_raw = features[test_indices]

    y_train = y[train_indices]
    y_test = y[test_indices]

    X_train, X_train_mean, X_train_std = standardize_features(X_train_raw)
    X_test, X_test_mean, X_test_std= standardize_features(X_test_raw)

    model = Perceptron(
        max_iter=1000,
        random_state=1234
    )

    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)

    print("Perceptron results")
    print("\n")
    print(f"Train accuracy: {train_accuracy}")
    print("\n")
    print(f"Test accuracy: {test_accuracy}")
    print("\n")
    print(classification_report(y_test, y_test_pred))

    train_predictions = create_predictions_dataframe(
        metadata=metadata,
        indices=train_indices,
        y_true=y_train,
        y_pred=y_train_pred,
        split_name="train",
    )

    test_predictions = create_predictions_dataframe(
        metadata=metadata,
        indices=test_indices,
        y_true=y_test,
        y_pred=y_test_pred,
        split_name="test",
    )

    predictions = pd.concat(
        [train_predictions, test_predictions],
        ignore_index=True,
    )

    model_path = output_dir / "perceptron.joblib"
    preprocessing_path = output_dir / "perceptron_preoprocessing.npz"
    predictions_path = output_dir / "predictions.csv"
    metrics_path = output_dir / "metrics.txt"

    joblib.dump(model, model_path)

    np.savez(
        preprocessing_path,
        features_mean=features_mean,
        features_std=features_std,
        downsample_factor=downsample_factor,
    )

    predictions.to_csv(predictions_path, index=False)

    with open(metrics_path, "w") as f:
        f.write("Perceptron evaluation\n")
        f.write("=" * 70 + "\n")
        f.write(f"Dataset directory: {dataset_dir}\n")
        f.write(f"Feature shape: {features.shape}\n")
        f.write(f"Train samples: {len(train_indices)}\n")
        f.write(f"Test samples: {len(test_indices)}\n")
        f.write(f"Train timesteps: {train_timesteps[0]} ... {train_timesteps[-1]}\n")
        f.write(f"Test timesteps: {test_timesteps[0]} ... {test_timesteps[-1]}\n")
        f.write(f"Train accuracy: {train_accuracy}\n")
        f.write(f"Test accuracy: {test_accuracy}\n\n")
        f.write("Test classification report\n")
        f.write("=" * 70 + "\n")
        f.write(classification_report(y_test, y_test_pred, zero_division=0))
        f.write("\n\nTest confusion matrix\n")
        f.write("=" * 70 + "\n")
        f.write(str(confusion_matrix(y_test, y_test_pred)))
        f.write("\n")

    print(model_path)
    print(metrics_path)
    print(predictions)

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
