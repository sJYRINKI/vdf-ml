#python scripts/train_logistic_regression.py --config configs/train_logistic_regression.yaml --dataset-id 3408_100 --model-id v1.0

import argparse
import sys
from pathlib import Path
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.config import load_config
from src.dataset_io import load_dataset
from src.features import create_features
from src.timesteps import create_path
from src.model_evaluation import create_predictions_dataframe
from src.model_split import split_by_timestep
from src.batches import iter_index_batches, predict_in_batches


def main(config_path, dataset_id, model_id):
    config = load_config(config_path)

    dataset_dir = create_path(
        path_template=config["dataset_dir"],
        dataset_id=dataset_id,
        model_id=model_id,
    )

    output_dir = create_path(
        path_template=config["output_dir"],
        dataset_id=dataset_id,
        model_id=model_id
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    features_config = config["features"]
    model_config = config["model"]

    downsample_factor = int(features_config.get("downsample_factor", 8))
    batch_size = int(features_config.get("batch_size", 64))
    max_iter = int(model_config.get("max_iter", 1000))
    alpha = float(model_config.get("alpha", 0.0001))
    penalty = model_config.get("penalty", "l2")
    learning_rate = model_config.get("learning_rate", "optimal")
    eta0 = float(model_config.get("eta0", 0.0))

    X, y, metadata = load_dataset(dataset_dir, mmap=True)

    print(X.shape)
    print(y.shape)
    print(metadata.shape)

    train_indices, test_indices, train_timesteps, test_timesteps = split_by_timestep(
        metadata=metadata
    )

    y_train = np.asarray(y[train_indices])
    y_test = np.asarray(y[test_indices])
    classes = np.unique(y)

    scaler = StandardScaler()
    logistic_regression = SGDClassifier(
        loss="log_loss",
        alpha=alpha,
        penalty=penalty,
        learning_rate=learning_rate,
        eta0=eta0,
        random_state=1234,
        warm_start=True,
        max_iter=max_iter,
        tol=None,
    )

    for batch_indices in iter_index_batches(train_indices, batch_size):
        features_batch = create_features(
            X=X[batch_indices],
            downsample_factor=downsample_factor,
        )

        scaler.partial_fit(features_batch)

    first_batch = True

    for epoch in range(max_iter):
        print(f"Epoch {epoch + 1}/{max_iter}")
        for batch_indices in iter_index_batches(train_indices, batch_size):
            features_batch = create_features(
                X[batch_indices],
                downsample_factor=downsample_factor,
            )

            features_batch = scaler.transform(features_batch)
            y_batch = np.asarray(y[batch_indices])

            if first_batch:
                logistic_regression.partial_fit(
                    features_batch,
                    y_batch,
                    classes=classes,
                )
                first_batch = False
            else:
                logistic_regression.partial_fit(
                    features_batch,
                    y_batch,
                )

    model = make_pipeline(scaler, logistic_regression)

    y_train_pred = predict_in_batches(
        model_pipeline=model,
        X=X,
        indices=train_indices,
        downsample_factor=downsample_factor,
        batch_size=batch_size,
    )

    y_test_pred = predict_in_batches(
        model_pipeline=model,
        X=X,
        indices=test_indices,
        downsample_factor=downsample_factor,
        batch_size=batch_size,
    )

    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)

    print("Logistic regression results")
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

    model_path = output_dir / "logistic_regression.joblib"
    preprocessing_path = output_dir / "preprocessing.npz"
    predictions_path = output_dir / "predictions.csv"
    metrics_path = output_dir / "metrics.txt"

    joblib.dump(model, model_path)

    np.savez(
        preprocessing_path,
        downsample_factor=downsample_factor,
        dataset_id=dataset_id,
        model_id=model_id,
        log_eps=1e-30,
        batch_size=batch_size,
    )

    predictions.to_csv(predictions_path, index=False)

    with open(metrics_path, "w") as f:
        f.write("Logistic regression evaluation\n")
        f.write("=" * 70 + "\n")
        f.write(f"Dataset ID: {dataset_id}\n")
        f.write(f"Model ID: {model_id}\n")
        f.write(f"Dataset directory: {dataset_dir}\n")
        f.write(f"Feature shape: {X.shape}\n")
        f.write(f"Train samples: {len(train_indices)}\n")
        f.write(f"Test samples: {len(test_indices)}\n")
        f.write(f"Train timesteps: {train_timesteps[0]} ... {train_timesteps[-1]}\n")
        f.write(f"Test timesteps: {test_timesteps[0]} ... {test_timesteps[-1]}\n")
        f.write(f"Max iterations: {max_iter}\n")
        f.write(f"Alpha: {alpha}\n")
        f.write(f"Penalty: {penalty}\n")
        f.write(f"Learning rate: {learning_rate}\n")
        f.write(f"Eta0: {eta0}\n")
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
        description="Train a logistic regression classifier on VDF data."
    )

    parser.add_argument(
        "--config",
        required=True,
        help="Path to logistic regression training config"
    )

    parser.add_argument(
        "--dataset-id",
        required=True,
        help="Dataset identifier."
    )

    parser.add_argument(
        "--model-id",
        required=True,
        help="Model identifier."
    )

    args = parser.parse_args()

    main(
        config_path=args.config,
        dataset_id=args.dataset_id,
        model_id=args.model_id,
    )