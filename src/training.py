from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.batches import create_features_in_batches
from src.dataset_io import load_dataset
from src.model_evaluation import create_lobe_vs_rest_labels, create_predictions_dataframe
from src.model_split import split_by_timestep
from src.timesteps import create_path


def train_logistic_regression(config, dataset_id, model_id):
    """
    Train and save a logistic regression lobe-vs-rest classifier.

    Parameters
    ----------
    config : dict
        Training config.
    dataset_id : str
        Dataset identifier.
    model_id : str
        Model identifier.
    """

    model_config = config["model"]
    max_iter = int(model_config.get("max_iter", 1000))
    penalty = model_config.get("penalty", "l2")
    solver = model_config.get("solver", "lbfgs")
    C = float(model_config.get("C", 1.0))

    estimator = LogisticRegression(
        max_iter=max_iter,
        penalty=penalty,
        solver=solver,
        C=C,
        random_state=1234,
    )

    train_binary_classifier(
        config=config,
        dataset_id=dataset_id,
        model_id=model_id,
        estimator=estimator,
        model_filename="logistic_regression.joblib",
        title="Logistic regression lobe-vs-rest evaluation",
        print_title="Logistic regression lobe-vs-rest results",
        extra_metric_lines=[
            ("Max iterations", max_iter),
            ("Penalty", penalty),
            ("Solver", solver),
            ("C", C),
        ],
    )


def train_perceptron(config, dataset_id, model_id):
    """
    Train and save a perceptron lobe-vs-rest classifier.

    Parameters
    ----------
    config : dict
        Training config.
    dataset_id : str
        Dataset identifier.
    model_id : str
        Model identifier.
    """

    model_config = config["model"]
    max_iter = int(model_config.get("max_iter", 1000))
    estimator = Perceptron(
        max_iter=max_iter,
        random_state=1234,
    )

    train_binary_classifier(
        config=config,
        dataset_id=dataset_id,
        model_id=model_id,
        estimator=estimator,
        model_filename="perceptron.joblib",
        title="Perceptron lobe-vs-rest evaluation",
        print_title="Perceptron lobe-vs-rest results",
        extra_metric_lines=[
            ("Max iterations", max_iter),
        ],
    )


def train_multilayer_perceptron_classifier(config, dataset_id, model_id):
    """
    Train and save a multilayer perceptron classifier.

    Parameters
    ----------
    config : dict
        Training config.
    dataset_id : str
        Dataset identifier.
    model_id : str
        Model identifier.
    """

    labels_config = config["labels"]
    model_config = config["model"]
    class_names_by_label = {
        int(label): class_name
        for class_name, label in labels_config.items()
    }
    class_labels = np.asarray(sorted(class_names_by_label), dtype=int)
    class_names = [class_names_by_label[label] for label in class_labels]

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

    estimator = MLPClassifier(
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

    data = load_training_data(
        config=config,
        dataset_id=dataset_id,
        model_id=model_id,
        target_kind="multiclass",
    )

    print("Configured classes:")
    for label, class_name in zip(class_labels, class_names):
        print(f"  {label}: {class_name}")

    model = make_pipeline(StandardScaler(), estimator)
    model.fit(data["X_train_features"], data["y_train"])

    results = evaluate_model(
        model=model,
        data=data,
        report_labels=class_labels,
        target_names=class_names,
    )

    print("Multilayer perceptron classifier results")
    print("\n")
    print(f"Train accuracy: {results['train_accuracy']}")
    print("\n")
    print(f"Test accuracy: {results['test_accuracy']}")
    print("\n")
    print(results["print_report"])

    predictions = create_predictions(
        metadata=data["metadata"],
        train_indices=data["train_indices"],
        test_indices=data["test_indices"],
        y_train=data["y_train"],
        y_test=data["y_test"],
        y_train_pred=results["y_train_pred"],
        y_test_pred=results["y_test_pred"],
    )
    predictions["true_class_name"] = predictions["true_label"].map(class_names_by_label)
    predictions["predicted_class_name"] = predictions["predicted_label"].map(
        class_names_by_label
    )

    fitted_classifier = model.named_steps["mlpclassifier"]
    save_training_outputs(
        model=model,
        output_dir=data["output_dir"],
        model_filename="multilayer_perceptron_classifier.joblib",
        preprocessing_values={
            "downsample_factor": data["downsample_factor"],
            "dataset_id": dataset_id,
            "model_id": model_id,
            "log_eps": data["log_eps"],
            "batch_size": data["batch_size"],
            "n_jobs": data["n_jobs"],
            "class_labels": class_labels,
            "class_names": np.asarray(class_names),
        },
        predictions=predictions,
        metrics_text=create_metrics_text(
            title="Multilayer perceptron classifier evaluation",
            dataset_id=dataset_id,
            model_id=model_id,
            data=data,
            results=results,
            target_line=None,
            extra_lines=[
                "Configured classes:",
                *[
                    f"  {label}: {class_name}"
                    for label, class_name in zip(class_labels, class_names)
                ],
                f"Hidden layer sizes: {hidden_layer_sizes}",
                f"Activation: {activation}",
                f"Solver: {solver}",
                f"Alpha: {alpha}",
                f"Model batch size: {model_batch_size}",
                f"Learning rate init: {learning_rate_init}",
                f"Max iterations: {max_iter}",
                f"Early stopping: {early_stopping}",
                f"Validation fraction: {validation_fraction}",
                f"Classifier classes: {list(fitted_classifier.classes_)}",
                f"Iterations: {fitted_classifier.n_iter_}",
                f"Final loss: {fitted_classifier.loss_}",
            ],
        ),
    )


def train_binary_classifier(
    config,
    dataset_id,
    model_id,
    estimator,
    model_filename,
    title,
    print_title,
    extra_metric_lines=None,
):
    """
    Train and save a binary lobe-vs-rest classifier.

    Parameters
    ----------
    config : dict
        Training config.
    dataset_id : str
        Dataset identifier.
    model_id : str
        Model identifier.
    estimator : sklearn estimator
        Classifier to train.
    model_filename : str
        Saved model filename.
    title : str
        Metrics file title.
    print_title : str
        Console title for evaluation output.
    extra_metric_lines : list of tuple, optional
        Additional model configuration lines.
    """

    data = load_training_data(
        config=config,
        dataset_id=dataset_id,
        model_id=model_id,
        target_kind="lobe_vs_rest",
    )
    model = make_pipeline(StandardScaler(), estimator)
    model.fit(data["X_train_features"], data["y_train"])

    results = evaluate_model(model=model, data=data)

    print(print_title)
    print("\n")
    print(f"Train accuracy: {results['train_accuracy']}")
    print("\n")
    print(f"Test accuracy: {results['test_accuracy']}")
    print("\n")
    print(results["print_report"])

    predictions = create_predictions(
        metadata=data["metadata"],
        train_indices=data["train_indices"],
        test_indices=data["test_indices"],
        y_train=data["y_train"],
        y_test=data["y_test"],
        y_train_pred=results["y_train_pred"],
        y_test_pred=results["y_test_pred"],
    )

    save_training_outputs(
        model=model,
        output_dir=data["output_dir"],
        model_filename=model_filename,
        preprocessing_values={
            "downsample_factor": data["downsample_factor"],
            "dataset_id": dataset_id,
            "model_id": model_id,
            "log_eps": data["log_eps"],
            "batch_size": data["batch_size"],
            "n_jobs": data["n_jobs"],
        },
        predictions=predictions,
        metrics_text=create_metrics_text(
            title=title,
            dataset_id=dataset_id,
            model_id=model_id,
            data=data,
            results=results,
            target_line="Target: lobe=1, rest=0",
            extra_lines=[
                f"{name}: {value}"
                for name, value in (extra_metric_lines or [])
            ],
        ),
    )


def load_training_data(config, dataset_id, model_id, target_kind):
    """
    Load dataset arrays, split indices, targets, and feature matrices.

    Parameters
    ----------
    config : dict
        Training config.
    dataset_id : str
        Dataset identifier.
    model_id : str
        Model identifier.
    target_kind : {"lobe_vs_rest", "multiclass"}
        Target labeling mode.

    Returns
    -------
    dict
        Training data, split metadata, and feature settings.
    """

    dataset_dir = create_path(
        path_template=config["dataset_dir"],
        dataset_id=dataset_id,
        model_id=model_id,
    )
    output_dir = create_path(
        path_template=config["output_dir"],
        dataset_id=dataset_id,
        model_id=model_id,
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    features_config = config["features"]
    downsample_factor = int(features_config.get("downsample_factor", 8))
    batch_size = int(features_config.get("batch_size", 64))
    n_jobs = int(features_config.get("n_jobs", 1))
    log_eps = float(features_config.get("log_eps", 1e-30))

    X, y, metadata = load_dataset(dataset_dir, mmap=True)

    print(X.shape)
    print(y.shape)
    print(metadata.shape)

    train_indices, test_indices, train_timesteps, test_timesteps = split_by_timestep(
        metadata=metadata
    )

    if target_kind == "lobe_vs_rest":
        target = create_lobe_vs_rest_labels(metadata)
        y_train = target[train_indices]
        y_test = target[test_indices]
    elif target_kind == "multiclass":
        y_train = np.asarray(y[train_indices], dtype=int)
        y_test = np.asarray(y[test_indices], dtype=int)
    else:
        raise ValueError(f"Unknown target kind: {target_kind}")

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

    return {
        "X": X,
        "y": y,
        "metadata": metadata,
        "dataset_dir": dataset_dir,
        "output_dir": output_dir,
        "train_indices": train_indices,
        "test_indices": test_indices,
        "train_timesteps": train_timesteps,
        "test_timesteps": test_timesteps,
        "y_train": y_train,
        "y_test": y_test,
        "X_train_features": X_train_features,
        "X_test_features": X_test_features,
        "downsample_factor": downsample_factor,
        "batch_size": batch_size,
        "n_jobs": n_jobs,
        "log_eps": log_eps,
    }


def evaluate_model(model, data, report_labels=None, target_names=None):
    """
    Evaluate a fitted model on train and test splits.

    Parameters
    ----------
    model : sklearn.pipeline.Pipeline
        Fitted model pipeline.
    data : dict
        Training data returned by ``load_training_data``.
    report_labels : array-like of int, optional
        Labels to include in reports.
    target_names : list of str, optional
        Class names for reports.

    Returns
    -------
    dict
        Predictions, metrics, reports, and confusion matrix text.
    """

    y_train_pred = model.predict(data["X_train_features"])
    y_test_pred = model.predict(data["X_test_features"])

    train_accuracy = accuracy_score(data["y_train"], y_train_pred)
    test_accuracy = accuracy_score(data["y_test"], y_test_pred)
    train_error = 1.0 - train_accuracy
    test_error = 1.0 - test_accuracy
    generalization_gap = test_error - train_error

    report_kwargs = {}
    if report_labels is not None:
        report_kwargs["labels"] = report_labels
    if target_names is not None:
        report_kwargs["target_names"] = target_names

    print_report = classification_report(
        data["y_test"],
        y_test_pred,
        **report_kwargs,
    )
    metrics_report = classification_report(
        data["y_test"],
        y_test_pred,
        zero_division=0,
        **report_kwargs,
    )
    matrix = confusion_matrix(
        data["y_test"],
        y_test_pred,
        labels=report_labels,
    )

    return {
        "y_train_pred": y_train_pred,
        "y_test_pred": y_test_pred,
        "train_accuracy": train_accuracy,
        "test_accuracy": test_accuracy,
        "train_error": train_error,
        "test_error": test_error,
        "generalization_gap": generalization_gap,
        "bias_proxy": train_error,
        "variance_proxy": max(0.0, generalization_gap),
        "print_report": print_report,
        "metrics_report": metrics_report,
        "confusion_matrix": matrix,
    }


def create_predictions(
    metadata,
    train_indices,
    test_indices,
    y_train,
    y_test,
    y_train_pred,
    y_test_pred,
):
    """
    Create train and test prediction rows.

    Parameters
    ----------
    metadata : pandas.DataFrame
        Dataset metadata.
    train_indices : numpy.ndarray
        Training sample indices.
    test_indices : numpy.ndarray
        Test sample indices.
    y_train : numpy.ndarray
        Training labels.
    y_test : numpy.ndarray
        Test labels.
    y_train_pred : numpy.ndarray
        Training predictions.
    y_test_pred : numpy.ndarray
        Test predictions.

    Returns
    -------
    pandas.DataFrame
        Combined prediction table.
    """

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

    return pd.concat(
        [train_predictions, test_predictions],
        ignore_index=True,
    )


def save_training_outputs(
    model,
    output_dir,
    model_filename,
    preprocessing_values,
    predictions,
    metrics_text,
):
    """
    Save model, preprocessing metadata, predictions, and metrics.

    Parameters
    ----------
    model : sklearn.pipeline.Pipeline
        Fitted model pipeline.
    output_dir : pathlib.Path
        Directory where model outputs are saved.
    model_filename : str
        Saved model filename.
    preprocessing_values : dict
        Values saved to ``preprocessing.npz``.
    predictions : pandas.DataFrame
        Prediction rows to save.
    metrics_text : str
        Metrics report text.
    """

    output_dir = Path(output_dir)
    model_path = output_dir / model_filename
    metrics_path = output_dir / "metrics.txt"

    joblib.dump(model, model_path)

    save_training_artifacts(
        output_dir=output_dir,
        preprocessing_values=preprocessing_values,
        predictions=predictions,
        metrics_text=metrics_text,
    )

    print(model_path)
    print(metrics_path)
    print(predictions)


def save_training_artifacts(
    output_dir,
    preprocessing_values,
    predictions,
    metrics_text,
):
    """
    Save preprocessing metadata, predictions, and evaluation metrics.

    Parameters
    ----------
    output_dir : str or pathlib.Path
        Directory where training artifacts are saved.
    preprocessing_values : dict
        Values saved to ``preprocessing.npz``.
    predictions : pandas.DataFrame
        Prediction rows to save.
    metrics_text : str
        Metrics report text.
    """

    output_dir = Path(output_dir)
    preprocessing_path = output_dir / "preprocessing.npz"
    predictions_path = output_dir / "predictions.csv"
    metrics_path = output_dir / "metrics.txt"

    np.savez(preprocessing_path, **preprocessing_values)
    predictions.to_csv(predictions_path, index=False)

    with open(metrics_path, "w") as metrics_file:
        metrics_file.write(metrics_text)


def create_metrics_text(
    title,
    dataset_id,
    model_id,
    data,
    results,
    target_line=None,
    extra_lines=None,
):
    """
    Create metrics text for saved model evaluation.

    Parameters
    ----------
    title : str
        Report title.
    dataset_id : str
        Dataset identifier.
    model_id : str
        Model identifier.
    data : dict
        Training data returned by ``load_training_data``.
    results : dict
        Evaluation results returned by ``evaluate_model``.
    target_line : str, optional
        Target label description.
    extra_lines : list of str, optional
        Additional model-specific metric lines.

    Returns
    -------
    str
        Metrics report text.
    """

    lines = [
        title,
        "=" * 70,
        f"Dataset ID: {dataset_id}",
        f"Model ID: {model_id}",
        f"Dataset directory: {data['dataset_dir']}",
        f"Raw dataset shape: {data['X'].shape}",
        f"Train feature shape: {data['X_train_features'].shape}",
        f"Test feature shape: {data['X_test_features'].shape}",
        f"Train samples: {len(data['train_indices'])}",
        f"Test samples: {len(data['test_indices'])}",
        f"Feature extraction jobs: {data['n_jobs']}",
    ]

    if target_line is not None:
        lines.append(target_line)

    lines.extend(
        [
            f"Train timesteps: {data['train_timesteps'][0]} ... {data['train_timesteps'][-1]}",
            f"Test timesteps: {data['test_timesteps'][0]} ... {data['test_timesteps'][-1]}",
        ]
    )
    lines.extend(extra_lines or [])
    lines.extend(
        [
            f"Train accuracy: {results['train_accuracy']}",
            f"Test accuracy: {results['test_accuracy']}",
            f"Train error: {results['train_error']}",
            f"Test error: {results['test_error']}",
            f"Generalization gap: {results['generalization_gap']}",
            f"Bias proxy: {results['bias_proxy']}",
            f"Variance proxy: {results['variance_proxy']}",
            "",
            "Test classification report",
            "=" * 70,
            results["metrics_report"],
            "",
            "Test confusion matrix",
            "=" * 70,
            str(results["confusion_matrix"]),
            "",
        ]
    )

    return "\n".join(lines)
