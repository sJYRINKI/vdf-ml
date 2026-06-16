#python scripts/train_multilayer_perceptron_classifier.py --config configs/train_multilayer_perceptron_classifier.yaml --dataset-id 3408_100 --model-id v1.0

import argparse
import sys
from pathlib import Path
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.config import load_config
from src.dataset_io import load_dataset
from src.timesteps import create_path
from src.model_evaluation import create_predictions_dataframe
from src.model_split import split_by_timestep
from src.batches import create_features_in_batches


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

    labels_config = config["labels"]
    features_config = config["features"]
    model_config = config["model"]

    class_names_by_label = {
        int(label): class_name
        for class_name, label in labels_config.items()
    }

    class_labels = np.asarray(sorted(class_names_by_label), dtype=int)
    class_names = [class_names_by_label[label] for label in class_labels]

    downsample_factor = int(features_config.get("downsample_factor", 8))
    batch_size = int(features_config.get("batch_size", 64))
    n_jobs = int(features_config.get("n_jobs", 1))
    log_eps = float(features_config.get("log_eps", 1e-30))

    hidden_layer_sizes = tuple(model_config.get("hidden_layer_sizes", [128, 64]))
    activation = model_config.get("activation", "relu")
    solver = model_config.get("solver", "adam")
    alpha = float(model_config.get("alpha", 0.0001))
    model_batch_size = model_config.get("batch_size", "auto")
    if model_batch_size != "auto":
        model_batch_size = int(model_batch_size)
    learning_rate_init = float(model_config.get("learning_rate_init", 0.001))
    max_iter = int(model_config.get("max_iter", 300))
    early_stopping = bool(model_config.get("early_stopping", True))
    validation_fraction = float(model_config.get("validation_fraction", 0.15))
    random_state = int(model_config.get("random_state", 1234))

    X, y, metadata = load_dataset(dataset_dir, mmap=True)

    print(X.shape)
    print(y.shape)
    print(metadata.shape)

    train_indices, test_indices, train_timesteps, test_timesteps = split_by_timestep(
        metadata=metadata
    )

    y_train = np.asarray(y[train_indices], dtype=int)
    y_test = np.asarray(y[test_indices], dtype=int)

    print("Configured classes:")
    for label, class_name in zip(class_labels, class_names):
        print(f"  {label}: {class_name}")
    print("Creating train features")
    X_train_features = create_features_in_batches(
        X=X,
        indices=train_indices,
        downsample_factor=downsample_factor,
        batch_size=batch_size,
        n_jobs=n_jobs,
        log_eps=log_eps,
    )

    print("Creating test features")
    X_test_features = create_features_in_batches(
        X=X,
        indices=test_indices,
        downsample_factor=downsample_factor,
        batch_size=batch_size,
        n_jobs=n_jobs,
        log_eps=log_eps,
    )

    print(f"Train features: {X_train_features.shape}")
    print(f"Test features: {X_test_features.shape}")

    scaler = StandardScaler()
    multilayer_perceptron_classifier = MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
        solver=solver,
        alpha=alpha,
        batch_size=model_batch_size,
        learning_rate_init=learning_rate_init,
        max_iter=max_iter,
        early_stopping=early_stopping,
        validation_fraction=validation_fraction,
        random_state=random_state,
    )

    model = make_pipeline(scaler, multilayer_perceptron_classifier)
    model.fit(X_train_features, y_train)

    y_train_pred = model.predict(X_train_features)
    y_test_pred = model.predict(X_test_features)

    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    train_error = 1.0 - train_accuracy
    test_error = 1.0 - test_accuracy
    generalization_gap = test_error - train_error
    bias_proxy = train_error
    variance_proxy = max(0.0, generalization_gap)

    print("Multilayer perceptron classifier results")
    print("\n")
    print(f"Train accuracy: {train_accuracy}")
    print("\n")
    print(f"Test accuracy: {test_accuracy}")
    print("\n")
    print(
        classification_report(
            y_test,
            y_test_pred,
            labels=class_labels,
            target_names=class_names,
            zero_division=0,
        )
    )

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
    predictions["true_class_name"] = predictions["true_label"].map(class_names_by_label)
    predictions["predicted_class_name"] = predictions["predicted_label"].map(
        class_names_by_label
    )

    model_path = output_dir / "multilayer_perceptron_classifier.joblib"
    preprocessing_path = output_dir / "preprocessing.npz"
    predictions_path = output_dir / "predictions.csv"
    metrics_path = output_dir / "metrics.txt"

    joblib.dump(model, model_path)

    np.savez(
        preprocessing_path,
        downsample_factor=downsample_factor,
        dataset_id=dataset_id,
        model_id=model_id,
        log_eps=log_eps,
        batch_size=batch_size,
        n_jobs=n_jobs,
        class_labels=class_labels,
        class_names=np.asarray(class_names),
    )

    predictions.to_csv(predictions_path, index=False)

    fitted_classifier = model.named_steps["mlpclassifier"]

    with open(metrics_path, "w") as f:
        f.write("Multilayer perceptron classifier evaluation\n")
        f.write("=" * 70 + "\n")
        f.write(f"Dataset ID: {dataset_id}\n")
        f.write(f"Model ID: {model_id}\n")
        f.write(f"Dataset directory: {dataset_dir}\n")
        f.write(f"Raw dataset shape: {X.shape}\n")
        f.write(f"Train feature shape: {X_train_features.shape}\n")
        f.write(f"Test feature shape: {X_test_features.shape}\n")
        f.write(f"Train samples: {len(train_indices)}\n")
        f.write(f"Test samples: {len(test_indices)}\n")
        f.write(f"Feature extraction jobs: {n_jobs}\n")
        f.write("Configured classes:\n")
        for label, class_name in zip(class_labels, class_names):
            f.write(f"  {label}: {class_name}\n")
        f.write(f"Train timesteps: {train_timesteps[0]} ... {train_timesteps[-1]}\n")
        f.write(f"Test timesteps: {test_timesteps[0]} ... {test_timesteps[-1]}\n")
        f.write(f"Hidden layer sizes: {hidden_layer_sizes}\n")
        f.write(f"Activation: {activation}\n")
        f.write(f"Solver: {solver}\n")
        f.write(f"Alpha: {alpha}\n")
        f.write(f"Model batch size: {model_batch_size}\n")
        f.write(f"Learning rate init: {learning_rate_init}\n")
        f.write(f"Max iterations: {max_iter}\n")
        f.write(f"Early stopping: {early_stopping}\n")
        f.write(f"Validation fraction: {validation_fraction}\n")
        f.write(f"Classifier classes: {list(fitted_classifier.classes_)}\n")
        f.write(f"Iterations: {fitted_classifier.n_iter_}\n")
        f.write(f"Final loss: {fitted_classifier.loss_}\n")
        f.write(f"Train accuracy: {train_accuracy}\n")
        f.write(f"Test accuracy: {test_accuracy}\n")
        f.write(f"Train error: {train_error}\n")
        f.write(f"Test error: {test_error}\n")
        f.write(f"Generalization gap: {generalization_gap}\n")
        f.write(f"Bias proxy: {bias_proxy}\n")
        f.write(f"Variance proxy: {variance_proxy}\n\n")
        f.write("Test classification report\n")
        f.write("=" * 70 + "\n")
        f.write(
            classification_report(
                y_test,
                y_test_pred,
                labels=class_labels,
                target_names=class_names,
                zero_division=0,
            )
        )
        f.write("\n\nTest confusion matrix\n")
        f.write("=" * 70 + "\n")
        f.write(str(confusion_matrix(y_test, y_test_pred, labels=class_labels)))
        f.write("\n")

    print(model_path)
    print(metrics_path)
    print(predictions)


if __name__=="__main__":
    parser = argparse.ArgumentParser(
        description="Train a multilayer perceptron classifier on VDF data."
    )

    parser.add_argument(
        "--config",
        required=True,
        help="Path to multilayer perceptron classifier training config"
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