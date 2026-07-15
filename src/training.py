from copy import deepcopy
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.autoencoder_data import create_or_load_log_slice_cache, resolve_cache_config
from src.batches import create_features_in_batches
from src.dataset_io import load_dataset
from src.features import create_features_from_log_slice_cache
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
    n_iter_no_change = int(model_config.get("n_iter_no_change", 10))
    tol = float(model_config.get("tol", 1e-4))
    random_state = int(model_config.get("random_state", 1234))

    estimator = MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
        solver=solver,
        alpha=alpha,
        batch_size=model_batch_size,
        learning_rate_init=learning_rate_init,
        max_iter=1,
        early_stopping=False,
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

    model, training_result = _fit_sklearn_mlp_with_timestep_validation(
        estimator=estimator,
        data=data,
        class_labels=class_labels,
        max_iter=max_iter,
        early_stopping=early_stopping,
        n_iter_no_change=n_iter_no_change,
        tol=tol,
    )

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
    print(f"Validation accuracy: {results['validation_accuracy']}")
    print("\n")
    print(f"Test accuracy: {results['test_accuracy']}")
    print("\n")
    print(results["print_report"])

    predictions = create_predictions(
        metadata=data["metadata"],
        train_indices=data["train_indices"],
        validation_indices=data["validation_indices"],
        test_indices=data["test_indices"],
        y_train=data["y_train"],
        y_validation=data["y_validation"],
        y_test=data["y_test"],
        y_train_pred=results["y_train_pred"],
        y_validation_pred=results["y_validation_pred"],
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
            "train_fraction": data["train_fraction"],
            "validation_fraction": data["validation_fraction"],
            "gap_timesteps": data["gap_timesteps"],
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
                f"Iterations without improvement: {n_iter_no_change}",
                f"Tolerance: {tol}",
                f"Classifier classes: {list(fitted_classifier.classes_)}",
                f"Iterations: {training_result['n_epochs']}",
                f"Best iteration: {training_result['best_epoch']}",
                "Best validation accuracy: "
                f"{training_result['best_validation_accuracy']}",
                f"Final loss: {fitted_classifier.loss_}",
            ],
        ),
    )


def _fit_sklearn_mlp_with_timestep_validation(
    estimator,
    data,
    class_labels,
    max_iter,
    early_stopping,
    n_iter_no_change,
    tol,
):
    """
    Fit a scikit-learn MLP using the timestep validation partition.

    Parameters
    ----------
    estimator : sklearn.neural_network.MLPClassifier
        Unfitted classifier.
    data : dict
        Training data returned by ``load_training_data``.
    class_labels : array-like of int
        Configured project labels.
    max_iter : int
        Maximum number of training epochs.
    early_stopping : bool
        Whether to stop based on validation accuracy.
    n_iter_no_change : int
        Epochs without sufficient improvement before stopping.
    tol : float
        Minimum validation-accuracy improvement that resets patience.

    Returns
    -------
    model : sklearn.pipeline.Pipeline
        Fitted scaler and classifier pipeline.
    training_result : dict
        Epoch count and best validation metrics.
    """

    if estimator.solver not in {"adam", "sgd"}:
        raise ValueError(
            "Timestep-validation MLP training supports only the adam and "
            "sgd solvers"
        )
    if max_iter <= 0:
        raise ValueError("max_iter must be positive")
    if n_iter_no_change <= 0:
        raise ValueError("n_iter_no_change must be positive")
    if tol < 0.0:
        raise ValueError("tol must be non-negative")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(data["X_train_features"])
    X_validation_scaled = scaler.transform(
        data["X_validation_features"]
    )

    best_estimator = None
    best_validation_accuracy = None
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(1, max_iter + 1):
        if epoch == 1:
            estimator.partial_fit(
                X_train_scaled,
                data["y_train"],
                classes=class_labels,
            )
        else:
            estimator.partial_fit(X_train_scaled, data["y_train"])

        validation_accuracy = estimator.score(
            X_validation_scaled,
            data["y_validation"],
        )
        significantly_improved = (
            best_validation_accuracy is None
            or validation_accuracy > best_validation_accuracy + tol
        )
        if (
            best_validation_accuracy is None
            or validation_accuracy > best_validation_accuracy
        ):
            best_validation_accuracy = validation_accuracy
            best_epoch = epoch
            if early_stopping:
                best_estimator = deepcopy(estimator)

        if significantly_improved:
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if early_stopping and epochs_without_improvement >= n_iter_no_change:
            break

    if best_estimator is not None:
        estimator = best_estimator

    return make_pipeline(scaler, estimator), {
        "n_epochs": epoch,
        "best_epoch": best_epoch,
        "best_validation_accuracy": best_validation_accuracy,
    }


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
    print(f"Validation accuracy: {results['validation_accuracy']}")
    print("\n")
    print(f"Test accuracy: {results['test_accuracy']}")
    print("\n")
    print(results["print_report"])

    predictions = create_predictions(
        metadata=data["metadata"],
        train_indices=data["train_indices"],
        validation_indices=data["validation_indices"],
        test_indices=data["test_indices"],
        y_train=data["y_train"],
        y_validation=data["y_validation"],
        y_test=data["y_test"],
        y_train_pred=results["y_train_pred"],
        y_validation_pred=results["y_validation_pred"],
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
            "train_fraction": data["train_fraction"],
            "validation_fraction": data["validation_fraction"],
            "gap_timesteps": data["gap_timesteps"],
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
    cache_config = resolve_cache_config(
        config=features_config.get("cache", {}),
        dataset_dir=dataset_dir,
        dataset_id=dataset_id,
        model_id=model_id,
    )

    split_config = config.get("split", {})
    train_fraction = float(split_config.get("train_fraction", 0.6))
    validation_fraction = float(split_config.get("validation_fraction", 0.2))
    gap_timesteps = int(split_config.get("gap_timesteps", 10))

    X, y, metadata = load_dataset(dataset_dir, mmap=True)
    X_log, cache_metadata = create_or_load_log_slice_cache(
        X=X,
        input_config=_create_feature_cache_input_config(log_eps=log_eps),
        cache_config=cache_config,
    )

    print(X.shape)
    print(y.shape)
    print(metadata.shape)

    (
        train_indices,
        validation_indices,
        test_indices,
        train_timesteps,
        validation_timesteps,
        test_timesteps,
        train_validation_gap,
        validation_test_gap,
    ) = split_by_timestep(
        metadata=metadata,
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
        gap_timesteps=gap_timesteps,
    )

    if target_kind == "lobe_vs_rest":
        target = create_lobe_vs_rest_labels(metadata)
        y_train = target[train_indices]
        y_validation = target[validation_indices]
        y_test = target[test_indices]
    elif target_kind == "multiclass":
        y_train = np.asarray(y[train_indices], dtype=int)
        y_validation = np.asarray(y[validation_indices], dtype=int)
        y_test = np.asarray(y[test_indices], dtype=int)
    else:
        raise ValueError(f"Unknown target kind: {target_kind}")

    print("Creating train features")
    X_train_features = _create_training_feature_matrix(
        X=X,
        X_log=X_log,
        indices=train_indices,
        downsample_factor=downsample_factor,
        batch_size=batch_size,
        n_jobs=n_jobs,
        log_eps=log_eps,
    )

    print("Creating validation features")
    X_validation_features = _create_training_feature_matrix(
        X=X,
        X_log=X_log,
        indices=validation_indices,
        downsample_factor=downsample_factor,
        batch_size=batch_size,
        n_jobs=n_jobs,
        log_eps=log_eps,
    )

    print("Creating test features")
    X_test_features = _create_training_feature_matrix(
        X=X,
        X_log=X_log,
        indices=test_indices,
        downsample_factor=downsample_factor,
        batch_size=batch_size,
        n_jobs=n_jobs,
        log_eps=log_eps,
    )

    print(f"Train features: {X_train_features.shape}")
    print(f"Validation features: {X_validation_features.shape}")
    print(f"Test features: {X_test_features.shape}")

    return {
        "X": X,
        "y": y,
        "metadata": metadata,
        "dataset_dir": dataset_dir,
        "output_dir": output_dir,
        "train_indices": train_indices,
        "validation_indices": validation_indices,
        "test_indices": test_indices,
        "train_timesteps": train_timesteps,
        "validation_timesteps": validation_timesteps,
        "test_timesteps": test_timesteps,
        "train_validation_gap": train_validation_gap,
        "validation_test_gap": validation_test_gap,
        "y_train": y_train,
        "y_validation": y_validation,
        "y_test": y_test,
        "X_train_features": X_train_features,
        "X_validation_features": X_validation_features,
        "X_test_features": X_test_features,
        "downsample_factor": downsample_factor,
        "batch_size": batch_size,
        "n_jobs": n_jobs,
        "log_eps": log_eps,
        "feature_cache_metadata": cache_metadata,
        "train_fraction": train_fraction,
        "validation_fraction": validation_fraction,
        "gap_timesteps": gap_timesteps,
    }


def _create_feature_cache_input_config(log_eps):
    """
    Create input settings for the shared log-slice feature cache.

    Parameters
    ----------
    log_eps : float
        Small positive value added before log scaling.

    Returns
    -------
    dict
        Input settings accepted by ``create_or_load_log_slice_cache``.
    """

    return {
        "slice": "xz",
        "orientation": "plot",
        "normalization": "train_global_standard",
        "log_eps": float(log_eps),
        "clip_negative_to_zero": True,
    }


def _create_training_feature_matrix(
    X,
    X_log,
    indices,
    downsample_factor,
    batch_size,
    n_jobs,
    log_eps,
):
    """
    Create CNN/ML feature matrix from cache when available.

    Parameters
    ----------
    X : numpy.ndarray
        VDF samples.
    X_log : numpy.ndarray or None
        Optional cached log-scaled xz slices.
    indices : array-like of int
        Sample indices.
    downsample_factor : int
        Factor used to downsample the xz slice.
    batch_size : int
        Number of samples per feature batch.
    n_jobs : int
        Number of parallel workers for raw VDF feature extraction.
    log_eps : float
        Small positive value added before log scaling.

    Returns
    -------
    numpy.ndarray
        Feature matrix.
    """

    if X_log is not None:
        return create_features_from_log_slice_cache(
            X_log=X_log,
            indices=indices,
            downsample_factor=downsample_factor,
            batch_size=batch_size,
        )

    return create_features_in_batches(
        X=X,
        indices=indices,
        downsample_factor=downsample_factor,
        batch_size=batch_size,
        n_jobs=n_jobs,
        log_eps=log_eps,
    )


def evaluate_model(
    model,
    data,
    report_labels=None,
    target_names=None,
    predict_kwargs_by_split=None,
):
    """
    Evaluate a fitted model on train, validation, and test splits.

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
    predict_kwargs_by_split : dict, optional
        Extra keyword arguments passed to ``model.predict`` for each split.

    Returns
    -------
    dict
        Predictions, metrics, reports, and confusion matrix text.
    """

    predict_kwargs_by_split = predict_kwargs_by_split or {}
    y_train_pred = model.predict(
        data["X_train_features"],
        **predict_kwargs_by_split.get("train", {}),
    )
    y_validation_pred = model.predict(
        data["X_validation_features"],
        **predict_kwargs_by_split.get("validation", {}),
    )
    y_test_pred = model.predict(
        data["X_test_features"],
        **predict_kwargs_by_split.get("test", {}),
    )

    train_accuracy = accuracy_score(data["y_train"], y_train_pred)
    validation_accuracy = accuracy_score(
        data["y_validation"], y_validation_pred
    )
    test_accuracy = accuracy_score(data["y_test"], y_test_pred)
    macro_f1_kwargs = {"average": "macro", "zero_division": 0}
    if report_labels is not None:
        macro_f1_kwargs["labels"] = report_labels

    train_macro_f1 = f1_score(
        data["y_train"],
        y_train_pred,
        **macro_f1_kwargs,
    )
    validation_macro_f1 = f1_score(
        data["y_validation"],
        y_validation_pred,
        **macro_f1_kwargs,
    )
    test_macro_f1 = f1_score(
        data["y_test"],
        y_test_pred,
        **macro_f1_kwargs,
    )
    train_error = 1.0 - train_accuracy
    validation_error = 1.0 - validation_accuracy
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
    train_metrics_report = classification_report(
        data["y_train"],
        y_train_pred,
        zero_division=0,
        **report_kwargs,
    )
    validation_metrics_report = classification_report(
        data["y_validation"],
        y_validation_pred,
        zero_division=0,
        **report_kwargs,
    )
    test_metrics_report = classification_report(
        data["y_test"],
        y_test_pred,
        zero_division=0,
        **report_kwargs,
    )
    train_confusion_matrix = confusion_matrix(
        data["y_train"],
        y_train_pred,
        labels=report_labels,
    )
    validation_confusion_matrix = confusion_matrix(
        data["y_validation"],
        y_validation_pred,
        labels=report_labels,
    )
    test_confusion_matrix = confusion_matrix(
        data["y_test"],
        y_test_pred,
        labels=report_labels,
    )

    return {
        "y_train_pred": y_train_pred,
        "y_validation_pred": y_validation_pred,
        "y_test_pred": y_test_pred,
        "train_accuracy": train_accuracy,
        "validation_accuracy": validation_accuracy,
        "test_accuracy": test_accuracy,
        "train_macro_f1": train_macro_f1,
        "validation_macro_f1": validation_macro_f1,
        "test_macro_f1": test_macro_f1,
        "train_error": train_error,
        "validation_error": validation_error,
        "test_error": test_error,
        "generalization_gap": generalization_gap,
        "bias_proxy": train_error,
        "variance_proxy": max(0.0, generalization_gap),
        "print_report": print_report,
        "train_metrics_report": train_metrics_report,
        "validation_metrics_report": validation_metrics_report,
        "test_metrics_report": test_metrics_report,
        "metrics_report": test_metrics_report,
        "train_confusion_matrix": train_confusion_matrix,
        "validation_confusion_matrix": validation_confusion_matrix,
        "test_confusion_matrix": test_confusion_matrix,
        "confusion_matrix": test_confusion_matrix,
    }


def create_predictions(
    metadata,
    train_indices,
    validation_indices,
    test_indices,
    y_train,
    y_validation,
    y_test,
    y_train_pred,
    y_validation_pred,
    y_test_pred,
):
    """
    Create train, validation, and test prediction rows.

    Parameters
    ----------
    metadata : pandas.DataFrame
        Dataset metadata.
    train_indices : numpy.ndarray
        Training sample indices.
    validation_indices : numpy.ndarray
        Validation sample indices.
    test_indices : numpy.ndarray
        Test sample indices.
    y_train : numpy.ndarray
        Training labels.
    y_validation : numpy.ndarray
        Validation labels.
    y_test : numpy.ndarray
        Test labels.
    y_train_pred : numpy.ndarray
        Training predictions.
    y_validation_pred : numpy.ndarray
        Validation predictions.
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
    validation_predictions = create_predictions_dataframe(
        metadata=metadata,
        indices=validation_indices,
        y_true=y_validation,
        y_pred=y_validation_pred,
        split_name="validation",
    )
    test_predictions = create_predictions_dataframe(
        metadata=metadata,
        indices=test_indices,
        y_true=y_test,
        y_pred=y_test_pred,
        split_name="test",
    )

    return pd.concat(
        [train_predictions, validation_predictions, test_predictions],
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
        f"Validation feature shape: {data['X_validation_features'].shape}",
        f"Test feature shape: {data['X_test_features'].shape}",
        f"Train samples: {len(data['train_indices'])}",
        f"Validation samples: {len(data['validation_indices'])}",
        f"Test samples: {len(data['test_indices'])}",
        f"Feature extraction jobs: {data['n_jobs']}",
        "Feature cache enabled: "
        f"{data.get('feature_cache_metadata', {}).get('enabled', False)}",
        "Feature cache path: "
        f"{data.get('feature_cache_metadata', {}).get('cache_path', 'none')}",
        "Train fraction of usable timesteps: "
        f"{data['train_fraction']}",
        "Validation fraction of usable timesteps: "
        f"{data['validation_fraction']}",
        "Test fraction of usable timesteps: "
        f"{1.0 - data['train_fraction'] - data['validation_fraction']}",
        f"Gap timesteps per boundary: {data['gap_timesteps']}",
    ]

    if target_line is not None:
        lines.append(target_line)

    lines.extend(
        [
            "Train timesteps: "
            f"{_format_timestep_range(data['train_timesteps'])}",
            "Train-validation gap: "
            f"{_format_timestep_range(data['train_validation_gap'])}",
            "Validation timesteps: "
            f"{_format_timestep_range(data['validation_timesteps'])}",
            "Validation-test gap: "
            f"{_format_timestep_range(data['validation_test_gap'])}",
            "Test timesteps: "
            f"{_format_timestep_range(data['test_timesteps'])}",
        ]
    )
    lines.extend(extra_lines or [])
    lines.extend(
        [
            f"Train accuracy: {results['train_accuracy']}",
            f"Validation accuracy: {results['validation_accuracy']}",
            f"Test accuracy: {results['test_accuracy']}",
            f"Train macro F1: {results['train_macro_f1']}",
            f"Validation macro F1: {results['validation_macro_f1']}",
            f"Test macro F1: {results['test_macro_f1']}",
            f"Train error: {results['train_error']}",
            f"Validation error: {results['validation_error']}",
            f"Test error: {results['test_error']}",
            f"Generalization gap: {results['generalization_gap']}",
            f"Bias proxy: {results['bias_proxy']}",
            f"Variance proxy: {results['variance_proxy']}",
            "",
            "Train classification report",
            "=" * 70,
            results["train_metrics_report"],
            "",
            "Validation classification report",
            "=" * 70,
            results["validation_metrics_report"],
            "",
            "Test classification report",
            "=" * 70,
            results["test_metrics_report"],
            "",
            "Train confusion matrix",
            "=" * 70,
            str(results["train_confusion_matrix"]),
            "",
            "Validation confusion matrix",
            "=" * 70,
            str(results["validation_confusion_matrix"]),
            "",
            "Test confusion matrix",
            "=" * 70,
            str(results["test_confusion_matrix"]),
            "",
        ]
    )

    return "\n".join(lines)


def _format_timestep_range(timesteps):
    if len(timesteps) == 0:
        return "none"
    if len(timesteps) == 1:
        return str(timesteps[0])
    return f"{timesteps[0]} ... {timesteps[-1]}"
