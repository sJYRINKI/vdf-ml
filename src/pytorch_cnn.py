from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import os

os.environ["PTNOLATEX"] = "1"

PCA_FILTER_CANDIDATE_COLUMN = "filter_preview_candidate"
PCA_FILTER_METRICS_FILENAME = "pca_filter_metrics.csv"
PCA_FILTER_REMOVED_FILENAME = "pca_filter_removed_samples.csv"
PCA_FILTER_SUMMARY_FILENAME = "pca_filter_summary.txt"

from src.training import (
    create_metrics_text,
    create_predictions,
    evaluate_model,
    load_training_data,
    save_training_artifacts,
)
from src.dataset_pca import plot_dataset_pca


class PyTorchCNNClassifier(nn.Module):
    """
    Classify flattened VDF xz-slice features with a 2D CNN.

    Parameters
    ----------
    input_size : int
        Number of flattened input features.
    channels : sequence of int
        Number of output channels in each convolutional block.
    classifier_size : int
        Number of neurons in the fully connected hidden layer.
    dropout : float
        Dropout probability before the output layer.
    class_labels : array-like of int
        Project labels in model-output order.
    feature_mean : array-like of float
        Training-feature means used for standardization.
    feature_scale : array-like of float
        Training-feature scales used for standardization.
    adaptive_pool_shape : sequence of int, optional
        Spatial output shape of the final adaptive average pooling layer.
    prediction_batch_size : int, optional
        Number of feature rows predicted at once.
    """

    def __init__(
        self,
        input_size,
        channels,
        classifier_size,
        dropout,
        class_labels,
        feature_mean,
        feature_scale,
        adaptive_pool_shape=(4, 4),
        prediction_batch_size=64,
    ):
        super().__init__()

        self.input_size = int(input_size)
        self.image_size = int(np.sqrt(self.input_size))
        self.channels = tuple(int(channel) for channel in channels)
        self.classifier_size = int(classifier_size)
        self.dropout = float(dropout)
        self.classes_ = np.asarray(class_labels, dtype=int)
        self.adaptive_pool_shape = _resolve_adaptive_pool_shape(
            adaptive_pool_shape
        )
        self.prediction_batch_size = int(prediction_batch_size)

        if self.image_size ** 2 != self.input_size:
            raise ValueError("CNN input features must form a square image")
        if not self.channels or any(channel <= 0 for channel in self.channels):
            raise ValueError("channels must contain positive integers")
        if self.classifier_size <= 0:
            raise ValueError("classifier_size must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be between zero and one")
        if self.prediction_batch_size <= 0:
            raise ValueError("prediction_batch_size must be positive")

        self.register_buffer(
            "feature_mean",
            torch.as_tensor(feature_mean, dtype=torch.float32),
        )
        self.register_buffer(
            "feature_scale",
            torch.as_tensor(feature_scale, dtype=torch.float32),
        )

        convolution_layers = []
        input_channels = 1
        for output_channels in self.channels:
            convolution_layers.extend(
                [
                    nn.Conv2d(
                        input_channels,
                        output_channels,
                        kernel_size=3,
                        padding=1,
                    ),
                    nn.ReLU(),
                    nn.AvgPool2d(kernel_size=2),
                ]
            )
            input_channels = output_channels

        convolution_layers.append(nn.AdaptiveAvgPool2d(self.adaptive_pool_shape))
        self.convolutions = nn.Sequential(*convolution_layers)
        pooled_height, pooled_width = self.adaptive_pool_shape
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                self.channels[-1] * pooled_height * pooled_width,
                self.classifier_size,
            ),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.classifier_size, len(self.classes_)),
        )

    def forward(self, features):
        """Return unnormalized class scores for a feature batch."""

        features = (features - self.feature_mean) / self.feature_scale
        images = features.reshape(
            -1,
            1,
            self.image_size,
            self.image_size,
        )
        return self.classifier(self.convolutions(images))

    def predict(self, features):
        """Predict project class labels."""

        class_indices = np.argmax(self.predict_proba(features), axis=1)
        return self.classes_[class_indices]

    def predict_proba(self, features):
        """Predict class probabilities in ``classes_`` order."""

        features = np.asarray(features, dtype=np.float32)
        if features.ndim != 2 or features.shape[1] != self.input_size:
            raise ValueError(
                "Expected feature matrix with shape "
                f"(n_samples, {self.input_size})"
            )

        device = self.feature_mean.device
        was_training = self.training
        self.eval()
        probabilities = np.empty(
            (len(features), len(self.classes_)),
            dtype=np.float32,
        )
        with torch.inference_mode():
            for start in range(0, len(features), self.prediction_batch_size):
                end = start + self.prediction_batch_size
                feature_batch = torch.as_tensor(
                    features[start:end],
                    dtype=torch.float32,
                    device=device,
                )
                probabilities[start:end] = (
                    torch.softmax(self(feature_batch), dim=1).cpu().numpy()
                )

        if was_training:
            self.train()
        return probabilities


def train_pytorch_convolutional_neural_network_classifier(
    config,
    dataset_id,
    model_id,
):
    """
    Train and save a PyTorch convolutional neural network classifier.

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
    class_names_by_label = {
        int(label): class_name
        for class_name, label in labels_config.items()
    }
    if len(class_names_by_label) != len(labels_config):
        raise ValueError("Configured class labels must be unique")

    class_labels = np.asarray(sorted(class_names_by_label), dtype=int)
    class_names = [class_names_by_label[label] for label in class_labels]
    if len(class_labels) < 2:
        raise ValueError("At least two configured classes are required")

    model_config = config["model"]
    channels = tuple(
        int(channel)
        for channel in model_config.get("channels", [16, 32, 64])
    )
    adaptive_pool_shape = _resolve_adaptive_pool_shape(
        model_config.get("adaptive_pool_shape", [4, 4])
    )
    classifier_size = int(model_config.get("classifier_size", 64))
    dropout = float(model_config.get("dropout", 0.2))
    class_weight = _resolve_class_weight(model_config.get("class_weight", "none"))
    weight_decay = float(model_config.get("weight_decay", 0.0001))
    learning_rate = float(model_config.get("learning_rate", 0.0003))
    max_epochs = int(model_config.get("max_epochs", 300))
    early_stopping = bool(model_config.get("early_stopping", True))
    patience = int(model_config.get("patience", 15))
    tolerance = float(model_config.get("tolerance", 1e-4))
    random_seed = int(model_config.get("random_state", 1234))

    if weight_decay < 0.0 or learning_rate <= 0.0:
        raise ValueError("weight_decay must be non-negative and learning rate positive")
    if max_epochs <= 0 or patience <= 0 or tolerance < 0.0:
        raise ValueError("Invalid epoch or early-stopping configuration")

    _run_training_filter_pca(
        config=config,
        dataset_id=dataset_id,
        model_id=model_id,
    )

    data = load_training_data(
        config=config,
        dataset_id=dataset_id,
        model_id=model_id,
        target_kind="multiclass",
    )
    training_filter_result = _apply_training_filter(
        data=data,
        config=config.get("training_filter", {}),
    )
    y_train = _encode_labels(data["y_train"], class_labels)
    y_validation = _encode_labels(data["y_validation"], class_labels)
    _encode_labels(data["y_test"], class_labels)

    missing_classes = set(range(len(class_labels))) - set(y_train)
    if missing_classes:
        missing_labels = class_labels[sorted(missing_classes)]
        raise ValueError(
            "Configured classes have no training samples: "
            f"{list(missing_labels)}"
        )

    class_weights = _create_class_weights(
        targets=y_train,
        n_classes=len(class_labels),
        class_weight=class_weight,
    )

    scaler = StandardScaler().fit(data["X_train_features"])
    device = _resolve_device(model_config.get("device", "auto"))
    deterministic = bool(model_config.get("deterministic", False))
    _set_random_seed(random_seed, deterministic)

    model_batch_size = _resolve_batch_size(
        model_config.get("batch_size", 32),
        len(y_train),
    )
    prediction_batch_size = _resolve_prediction_batch_size(
        configured_batch_size=model_config.get("prediction_batch_size", 64),
        model_batch_size=model_batch_size,
        n_samples=len(y_train),
    )

    model = PyTorchCNNClassifier(
        input_size=data["X_train_features"].shape[1],
        channels=channels,
        classifier_size=classifier_size,
        dropout=dropout,
        class_labels=class_labels,
        feature_mean=np.asarray(scaler.mean_, dtype=np.float32),
        feature_scale=np.asarray(scaler.scale_, dtype=np.float32),
        adaptive_pool_shape=adaptive_pool_shape,
        prediction_batch_size=prediction_batch_size,
    ).to(device)

    print("Configured classes:")
    for label, class_name in zip(class_labels, class_names):
        print(f"  {label}: {class_name}")
    print(f"CNN input image: {model.image_size} x {model.image_size}")
    print(f"Training device: {device}")

    training_result = _fit_model(
        model=model,
        features=data["X_train_features"],
        targets=y_train,
        validation_features=data["X_validation_features"],
        validation_targets=y_validation,
        class_weights=class_weights,
        device=device,
        batch_size=model_batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        max_epochs=max_epochs,
        early_stopping=early_stopping,
        tolerance=tolerance,
        patience=patience,
        random_seed=random_seed,
    )
    model.eval()

    results = evaluate_model(
        model=model,
        data=data,
        report_labels=class_labels,
        target_names=class_names,
    )
    print("PyTorch convolutional neural network classifier results")
    print(f"Train accuracy: {results['train_accuracy']}")
    print(f"Validation accuracy: {results['validation_accuracy']}")
    print(f"Test accuracy: {results['test_accuracy']}")
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
    predictions["true_class_name"] = predictions["true_label"].map(
        class_names_by_label
    )
    predictions["predicted_class_name"] = predictions["predicted_label"].map(
        class_names_by_label
    )

    failure_plot_paths = _plot_failure_cases(
        data=data,
        predictions=predictions,
        class_names_by_label=class_names_by_label,
        plot_config=config.get("failure_plots", {}),
    )

    checkpoint_path = (
        data["output_dir"]
        / "pytorch_convolutional_neural_network_classifier.pt"
    )
    save_pytorch_cnn_checkpoint(model, checkpoint_path)

    metric_lines = [
        "Configured classes:",
        *[
            f"  {label}: {class_name}"
            for label, class_name in zip(class_labels, class_names)
        ],
        f"Input image shape: (1, {model.image_size}, {model.image_size})",
        f"Convolution channels: {channels}",
        f"Adaptive pool shape: {adaptive_pool_shape}",
        f"Classifier size: {classifier_size}",
        f"Dropout: {dropout}",
        f"Class weight: {class_weight}",
        "Activation: relu",
        "Pooling: average",
        "Optimizer: AdamW",
        f"Weight decay: {weight_decay}",
        f"Model batch size: {model_batch_size}",
        f"Prediction batch size: {prediction_batch_size}",
        f"Learning rate: {learning_rate}",
        f"Max epochs: {max_epochs}",
        f"Early stopping: {early_stopping}",
        "Early stopping metric: validation macro F1",
        f"Patience: {patience}",
        f"Tolerance: {tolerance}",
        f"Random state: {random_seed}",
        f"Device: {device}",
        f"Deterministic algorithms: {deterministic}",
        f"PyTorch version: {torch.__version__}",
        f"Classifier classes: {list(model.classes_)}",
        *training_filter_result["metric_lines"],
        f"Epochs: {training_result['n_epochs']}",
        f"Best epoch: {training_result['best_epoch']}",
        f"Final training loss: {training_result['final_training_loss']}",
        f"Failure plots saved: {len(failure_plot_paths)}",
    ]
    if failure_plot_paths:
        metric_lines.append(f"Failure plot directory: {data['output_dir'] / 'failure_plots'}")
    if class_weights is not None:
        metric_lines.append(f"Class weights: {class_weights.tolist()}")
    if training_result["best_validation_macro_f1"] is not None:
        metric_lines.append(
            "Best validation macro F1: "
            f"{training_result['best_validation_macro_f1']}"
        )

    save_training_artifacts(
        output_dir=data["output_dir"],
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
            "adaptive_pool_shape": np.asarray(adaptive_pool_shape, dtype=int),
        },
        predictions=predictions,
        metrics_text=create_metrics_text(
            title="PyTorch convolutional neural network classifier evaluation",
            dataset_id=dataset_id,
            model_id=model_id,
            data=data,
            results=results,
            extra_lines=metric_lines,
        ),
    )

    print(checkpoint_path)
    print(data["output_dir"] / "metrics.txt")
    print(predictions)


def load_pytorch_cnn_checkpoint(
    checkpoint_path,
    device="cpu",
    prediction_batch_size=64,
):
    """
    Load a PyTorch convolutional neural network checkpoint.

    Parameters
    ----------
    checkpoint_path : str or pathlib.Path
        Saved checkpoint path.
    device : str or torch.device, optional
        Device used for prediction.
    prediction_batch_size : int, optional
        Number of feature rows predicted at once.

    Returns
    -------
    PyTorchCNNClassifier
        Loaded classifier with a NumPy prediction interface.
    """

    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

    if int(checkpoint["format_version"]) != 1:
        raise ValueError(
            "Unsupported PyTorch CNN checkpoint format version: "
            f"{checkpoint['format_version']}"
        )

    class_labels = checkpoint["class_labels"]
    if isinstance(class_labels, torch.Tensor):
        class_labels = class_labels.cpu().numpy()
    if len(class_labels) != int(checkpoint["n_classes"]):
        raise ValueError("Checkpoint class-label count does not match model output")

    input_size = int(checkpoint["input_size"])
    model = PyTorchCNNClassifier(
        input_size=input_size,
        channels=checkpoint["channels"],
        classifier_size=int(checkpoint["classifier_size"]),
        dropout=float(checkpoint["dropout"]),
        class_labels=class_labels,
        feature_mean=np.zeros(input_size, dtype=np.float32),
        feature_scale=np.ones(input_size, dtype=np.float32),
        adaptive_pool_shape=checkpoint.get("adaptive_pool_shape", [4, 4]),
        prediction_batch_size=prediction_batch_size,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.to(device).eval()


def save_pytorch_cnn_checkpoint(model, checkpoint_path):
    """Save model weights and architecture information."""

    torch.save(
        {
            "format_version": 1,
            "model_state_dict": {
                name: value.detach().cpu()
                for name, value in model.state_dict().items()
            },
            "input_size": model.input_size,
            "channels": list(model.channels),
            "classifier_size": model.classifier_size,
            "adaptive_pool_shape": list(model.adaptive_pool_shape),
            "dropout": model.dropout,
            "n_classes": len(model.classes_),
            "class_labels": torch.as_tensor(model.classes_, dtype=torch.int64),
        },
        checkpoint_path,
    )


def _fit_model(
    model,
    features,
    targets,
    validation_features,
    validation_targets,
    class_weights,
    device,
    batch_size,
    learning_rate,
    weight_decay,
    max_epochs,
    early_stopping,
    tolerance,
    patience,
    random_seed,
):
    features = np.asarray(features, dtype=np.float32)
    dataset = TensorDataset(
        torch.from_numpy(features),
        torch.from_numpy(targets),
    )
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
        generator=torch.Generator().manual_seed(random_seed),
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    if class_weights is None:
        loss_weights = None
    else:
        loss_weights = torch.as_tensor(
            class_weights,
            dtype=torch.float32,
            device=device,
        )
    loss_function = nn.CrossEntropyLoss(weight=loss_weights)
    best_score = float("-inf")
    best_state_dict = None
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        total_loss = torch.zeros((), dtype=torch.float32, device=device)

        for feature_batch, target_batch in data_loader:
            feature_batch = feature_batch.to(
                device,
                non_blocking=device.type == "cuda",
            )
            target_batch = target_batch.to(
                device,
                non_blocking=device.type == "cuda",
            )
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model(feature_batch), target_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.detach() * len(feature_batch)

        training_loss = float(total_loss.cpu()) / len(dataset)
        if early_stopping:
            validation_predictions = np.argmax(
                model.predict_proba(validation_features),
                axis=1,
            )
            score = f1_score(
                validation_targets,
                validation_predictions,
                labels=np.arange(len(model.classes_)),
                average="macro",
                zero_division=0,
            )
        else:
            score = -training_loss

        significantly_improved = score > best_score + tolerance
        if score > best_score:
            best_score = score
            best_epoch = epoch
            if early_stopping:
                best_state_dict = {
                    name: value.detach().cpu().clone()
                    for name, value in model.state_dict().items()
                }

        if significantly_improved:
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= patience:
            break

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    return {
        "n_epochs": epoch,
        "best_epoch": best_epoch,
        "best_validation_macro_f1": best_score if early_stopping else None,
        "final_training_loss": training_loss,
    }


def _run_training_filter_pca(config, dataset_id, model_id):
    """
    Run PCA metrics needed by the optional training filter.

    Parameters
    ----------
    config : dict
        Full CNN training config.
    dataset_id : str
        Dataset identifier.
    model_id : str
        Model identifier also used as the PCA output version.
    """

    filter_config = config.get("training_filter", {}) or {}
    if not bool(filter_config.get("enabled", False)):
        return

    source = str(filter_config.get("source", "pca")).lower()
    if source != "pca":
        raise ValueError("training_filter.source currently supports only 'pca'")

    pca_config = _create_training_filter_pca_config(
        config=config,
        filter_config=filter_config,
        dataset_id=dataset_id,
        model_id=model_id,
    )
    print("Running PCA for CNN training filter")
    plot_dataset_pca(
        config=pca_config,
        timestep=dataset_id,
        pca_id=None,
    )


def _create_training_filter_pca_config(config, filter_config, dataset_id, model_id):
    """
    Create a PCA config from the CNN training-filter settings.

    Parameters
    ----------
    config : dict
        Full CNN training config.
    filter_config : dict
        Training filter config.
    dataset_id : str
        Dataset identifier.
    model_id : str
        Model identifier.

    Returns
    -------
    dict
        Config accepted by ``plot_dataset_pca``.
    """

    dataset_dir = _format_training_path(
        path_template=config["dataset_dir"],
        dataset_id=dataset_id,
        model_id=model_id,
    )
    model_output_dir = _format_training_path(
        path_template=config["output_dir"],
        dataset_id=dataset_id,
        model_id=model_id,
    )
    pca_filter_config = filter_config.get("pca", {}) or {}
    pca_filter_preview_config = dict(
        pca_filter_config.get("filter_preview", {}) or {}
    )
    pca_filter_preview_config.setdefault("enabled", True)
    pca_filter_preview_config.setdefault(
        "candidate_classes",
        filter_config.get("candidate_classes", ["exhaust", "dayside"]),
    )
    pca_filter_preview_config.setdefault(
        "point_neighbor_classes",
        filter_config.get(
            "point_neighbor_classes",
            filter_config.get("protected_classes", ["reconnection", "o_point"]),
        ),
    )
    pca_filter_preview_config.setdefault(
        "protected_classes",
        filter_config.get("protected_classes", ["reconnection", "o_point"]),
    )
    pca_filter_preview_config.setdefault("apply_splits", ["train"])

    return {
        "dataset_dir": str(dataset_dir),
        "output_dir": str(model_output_dir / "pca"),
        "split": {
            "train_fraction": config.get("split", {}).get("train_fraction", 0.6),
            "validation_fraction": config.get("split", {}).get(
                "validation_fraction",
                0.2,
            ),
            "gap_timesteps": config.get("split", {}).get("gap_timesteps", 10),
        },
        "features": pca_filter_config.get("features", {}),
        "pca": pca_filter_config.get("pca", {}),
        "pca_fit": pca_filter_config.get("pca_fit", {}),
        "neighbor_metrics": pca_filter_config.get("neighbor_metrics", {}),
        "filter_preview": pca_filter_preview_config,
        "plot": pca_filter_config.get("plot", {"enabled": False}),
        "embedding_plot": pca_filter_config.get(
            "embedding_plot",
            {"enabled": False},
        ),
    }


def _format_training_path(path_template, dataset_id, model_id):
    """
    Format a path template with common training identifiers.

    Parameters
    ----------
    path_template : str
        Path template.
    dataset_id : str
        Dataset identifier.
    model_id : str
        Model identifier.

    Returns
    -------
    pathlib.Path
        Formatted path.
    """

    return Path(
        str(path_template).format(
            dataset_id=dataset_id,
            model_id=model_id,
            timestep=dataset_id,
        )
    )


def _create_pca_metrics_path(output_dir):
    """
    Create the model-local PCA sample-metrics path.

    Parameters
    ----------
    output_dir : str or pathlib.Path
        CNN model output directory.

    Returns
    -------
    pathlib.Path
        Path to PCA sample metrics.
    """

    return Path(output_dir) / "pca" / "pca_sample_metrics.csv"


def _apply_training_filter(data, config):
    """
    Apply optional non-destructive filtering to CNN training arrays.

    Parameters
    ----------
    data : dict
        Training data returned by ``load_training_data``.
    config : dict
        Training filter configuration.

    Returns
    -------
    dict
        Summary lines to include in model metrics.
    """

    config = config or {}
    if not bool(config.get("enabled", False)):
        return {"metric_lines": ["Training filter enabled: False"]}

    source = str(config.get("source", "pca")).lower()
    if source != "pca":
        raise ValueError("training_filter.source currently supports only 'pca'")

    dry_run = bool(config.get("dry_run", True))
    candidate_classes = _resolve_training_filter_classes(
        config.get("candidate_classes"),
        "training_filter.candidate_classes",
        require_nonempty=True,
    )
    protected_classes = _resolve_training_filter_classes(
        config.get("protected_classes", []),
        "training_filter.protected_classes",
        require_nonempty=False,
    )
    max_removed_fraction = _resolve_max_removed_fraction(
        config.get("max_removed_fraction_per_class", 1.0)
    )

    pca_metrics_path = _create_pca_metrics_path(output_dir=data["output_dir"])
    if not pca_metrics_path.exists():
        raise FileNotFoundError(
            "PCA training filter requires metrics at "
            f"{pca_metrics_path}"
        )

    pca_metrics = pd.read_csv(pca_metrics_path)
    filter_metrics = _create_pca_filter_metrics(
        pca_metrics=pca_metrics,
        train_indices=data["train_indices"],
        candidate_classes=candidate_classes,
        protected_classes=protected_classes,
        max_removed_fraction=max_removed_fraction,
        dry_run=dry_run,
    )

    selected_sample_indices = filter_metrics.loc[
        filter_metrics["pca_filter_selected"],
        "sample_index",
    ].to_numpy(dtype=int)

    if not dry_run and len(selected_sample_indices) > 0:
        _remove_training_samples(data, selected_sample_indices)

    output_dir = data["output_dir"]
    metrics_path = output_dir / PCA_FILTER_METRICS_FILENAME
    removed_path = output_dir / PCA_FILTER_REMOVED_FILENAME
    summary_path = output_dir / PCA_FILTER_SUMMARY_FILENAME

    filter_metrics.to_csv(metrics_path, index=False)
    filter_metrics[filter_metrics["pca_filter_selected"]].to_csv(
        removed_path,
        index=False,
    )

    summary_text = _create_pca_filter_summary_text(
        pca_metrics_path=pca_metrics_path,
        metrics_path=metrics_path,
        removed_path=removed_path,
        dry_run=dry_run,
        candidate_classes=candidate_classes,
        protected_classes=protected_classes,
        max_removed_fraction=max_removed_fraction,
        filter_metrics=filter_metrics,
        train_samples_after_filter=len(data["train_indices"]),
    )
    with open(summary_path, "w") as summary_file:
        summary_file.write(summary_text)

    selected_count = int(filter_metrics["pca_filter_selected"].sum())
    removed_count = int(filter_metrics["pca_filter_removed"].sum())
    print("PCA training filter")
    print(f"  Metrics: {metrics_path}")
    print(f"  Selected samples: {selected_count}")
    print(f"  Removed samples: {removed_count}")
    if dry_run:
        print("  Dry run: training samples were not removed")

    return {
        "metric_lines": _create_pca_filter_metric_lines(
            pca_metrics_path=pca_metrics_path,
            metrics_path=metrics_path,
            removed_path=removed_path,
            summary_path=summary_path,
            dry_run=dry_run,
            candidate_classes=candidate_classes,
            protected_classes=protected_classes,
            max_removed_fraction=max_removed_fraction,
            filter_metrics=filter_metrics,
            train_samples_after_filter=len(data["train_indices"]),
        )
    }


def _create_pca_filter_metrics(
    pca_metrics,
    train_indices,
    candidate_classes,
    protected_classes,
    max_removed_fraction,
    dry_run,
):
    """
    Create per-training-sample PCA filter decisions.

    Parameters
    ----------
    pca_metrics : pandas.DataFrame
        PCA per-sample metrics.
    train_indices : numpy.ndarray
        Original dataset sample indices in the CNN training split.
    candidate_classes : list of str
        Class names that may be removed from CNN training.
    protected_classes : list of str
        Class names that must never be removed.
    max_removed_fraction : float
        Maximum selected fraction per removable class.
    dry_run : bool
        Whether selected samples are only reported instead of removed.

    Returns
    -------
    pandas.DataFrame
        PCA metrics with CNN training-filter decision columns.
    """

    required_columns = {
        "sample_index",
        "split",
        "class_name",
        PCA_FILTER_CANDIDATE_COLUMN,
    }
    missing_columns = sorted(required_columns - set(pca_metrics.columns))
    if missing_columns:
        raise ValueError(
            "PCA filter metrics are missing required columns: "
            f"{missing_columns}"
        )
    if pca_metrics["sample_index"].duplicated().any():
        raise ValueError("PCA filter metrics contain duplicate sample_index values")

    train_order = pd.DataFrame(
        {
            "sample_index": np.asarray(train_indices, dtype=int),
            "cnn_train_order": np.arange(len(train_indices), dtype=int),
        }
    )
    filter_metrics = train_order.merge(
        pca_metrics,
        on="sample_index",
        how="left",
        validate="one_to_one",
    )
    missing_rows = filter_metrics["split"].isna()
    if missing_rows.any():
        missing_sample_indices = filter_metrics.loc[
            missing_rows,
            "sample_index",
        ].head(10).tolist()
        raise ValueError(
            "PCA filter metrics do not cover all CNN training samples. "
            f"First missing sample indices: {missing_sample_indices}"
        )

    non_train_rows = filter_metrics["split"] != "train"
    if non_train_rows.any():
        bad_sample_indices = filter_metrics.loc[
            non_train_rows,
            "sample_index",
        ].head(10).tolist()
        raise ValueError(
            "PCA filter metrics do not use the same train split as the CNN run. "
            f"First mismatched sample indices: {bad_sample_indices}"
        )

    pca_candidate = _coerce_boolean_column(
        filter_metrics[PCA_FILTER_CANDIDATE_COLUMN],
        PCA_FILTER_CANDIDATE_COLUMN,
    )
    candidate_class_mask = filter_metrics["class_name"].isin(
        candidate_classes
    ).to_numpy(dtype=bool)
    protected_class_mask = filter_metrics["class_name"].isin(
        protected_classes
    ).to_numpy(dtype=bool)
    eligible_mask = pca_candidate & candidate_class_mask & ~protected_class_mask

    filter_metrics["pca_filter_candidate"] = pca_candidate
    filter_metrics["pca_filter_eligible"] = eligible_mask
    filter_metrics["pca_filter_selected"] = False
    selected_indices = _select_pca_filter_candidates(
        filter_metrics=filter_metrics,
        eligible_mask=eligible_mask,
        candidate_classes=candidate_classes,
        max_removed_fraction=max_removed_fraction,
    )
    if selected_indices:
        filter_metrics.loc[selected_indices, "pca_filter_selected"] = True

    filter_metrics["pca_filter_removed"] = (
        filter_metrics["pca_filter_selected"] & (not dry_run)
    )
    filter_metrics["pca_filter_dry_run"] = dry_run
    filter_metrics["pca_filter_reason"] = _create_pca_filter_reasons(
        filter_metrics=filter_metrics,
        pca_candidate=pca_candidate,
        candidate_class_mask=candidate_class_mask,
        protected_class_mask=protected_class_mask,
    )
    return filter_metrics


def _select_pca_filter_candidates(
    filter_metrics,
    eligible_mask,
    candidate_classes,
    max_removed_fraction,
):
    selected_indices = []
    for class_name in candidate_classes:
        class_mask = (
            filter_metrics["class_name"] == class_name
        ).to_numpy(dtype=bool)
        class_rows = filter_metrics[class_mask]
        class_candidates = filter_metrics[class_mask & eligible_mask]
        class_limit = int(np.floor(max_removed_fraction * len(class_rows)))
        if class_limit <= 0 or class_candidates.empty:
            continue

        ranked_candidates = _rank_pca_filter_candidates(class_candidates)
        selected_indices.extend(
            ranked_candidates.index[: min(class_limit, len(ranked_candidates))]
        )

    return selected_indices


def _rank_pca_filter_candidates(rows):
    sort_columns = []
    ascending = []
    if "point_neighbor_fraction" in rows.columns:
        sort_columns.append("point_neighbor_fraction")
        ascending.append(False)
    if "same_class_fraction" in rows.columns:
        sort_columns.append("same_class_fraction")
        ascending.append(True)
    sort_columns.append("sample_index")
    ascending.append(True)
    return rows.sort_values(
        sort_columns,
        ascending=ascending,
        na_position="last",
    )


def _remove_training_samples(data, selected_sample_indices):
    selected_sample_indices = np.asarray(selected_sample_indices, dtype=int)
    keep_mask = ~np.isin(
        np.asarray(data["train_indices"], dtype=int),
        selected_sample_indices,
    )
    data["X_train_features"] = data["X_train_features"][keep_mask]
    data["y_train"] = data["y_train"][keep_mask]
    data["train_indices"] = data["train_indices"][keep_mask]


def _create_pca_filter_reasons(
    filter_metrics,
    pca_candidate,
    candidate_class_mask,
    protected_class_mask,
):
    reasons = np.full(len(filter_metrics), "", dtype=object)
    selected = filter_metrics["pca_filter_selected"].to_numpy(dtype=bool)
    eligible = filter_metrics["pca_filter_eligible"].to_numpy(dtype=bool)

    reasons[pca_candidate & ~candidate_class_mask] = "not_candidate_class"
    reasons[pca_candidate & protected_class_mask] = "protected_class"
    reasons[eligible & ~selected] = "class_fraction_cap"
    reasons[selected] = "selected_by_pca_filter"
    return reasons


def _create_pca_filter_metric_lines(
    pca_metrics_path,
    metrics_path,
    removed_path,
    summary_path,
    dry_run,
    candidate_classes,
    protected_classes,
    max_removed_fraction,
    filter_metrics,
    train_samples_after_filter,
):
    summary_counts = _summarize_pca_filter_counts(filter_metrics)
    lines = [
        "Training filter enabled: True",
        "Training filter source: pca",
        f"Training filter dry run: {dry_run}",
        f"PCA filter source metrics: {pca_metrics_path}",
        f"PCA filter candidate column: {PCA_FILTER_CANDIDATE_COLUMN}",
        f"PCA filter candidate classes: {', '.join(candidate_classes)}",
        f"PCA filter protected classes: {', '.join(protected_classes)}",
        f"PCA filter max removed fraction per class: {max_removed_fraction}",
        f"PCA filter train samples before filter: {len(filter_metrics)}",
        f"PCA filter selected samples: {int(filter_metrics['pca_filter_selected'].sum())}",
        f"PCA filter removed samples: {int(filter_metrics['pca_filter_removed'].sum())}",
        f"PCA filter train samples after filter: {train_samples_after_filter}",
        f"PCA filter metrics: {metrics_path}",
        f"PCA filter selected-sample file: {removed_path}",
        f"PCA filter summary: {summary_path}",
    ]
    for _, row in summary_counts.iterrows():
        lines.append(
            "PCA filter class "
            f"{row['class_name']}: train={int(row['train_samples'])}, "
            f"candidate={int(row['pca_candidates'])}, "
            f"eligible={int(row['eligible_candidates'])}, "
            f"selected={int(row['selected_samples'])}, "
            f"removed={int(row['removed_samples'])}"
        )
    return lines


def _create_pca_filter_summary_text(
    pca_metrics_path,
    metrics_path,
    removed_path,
    dry_run,
    candidate_classes,
    protected_classes,
    max_removed_fraction,
    filter_metrics,
    train_samples_after_filter,
):
    summary_counts = _summarize_pca_filter_counts(filter_metrics)
    lines = [
        "PCA training filter summary",
        "=" * 70,
        f"Source PCA metrics: {pca_metrics_path}",
        f"Output metrics: {metrics_path}",
        f"Selected-sample file: {removed_path}",
        f"Candidate column: {PCA_FILTER_CANDIDATE_COLUMN}",
        f"Dry run: {dry_run}",
        f"Candidate classes: {', '.join(candidate_classes)}",
        f"Protected classes: {', '.join(protected_classes)}",
        f"Max removed fraction per class: {max_removed_fraction}",
        f"Train samples before filter: {len(filter_metrics)}",
        f"Selected samples: {int(filter_metrics['pca_filter_selected'].sum())}",
        f"Removed samples: {int(filter_metrics['pca_filter_removed'].sum())}",
        f"Train samples after filter: {train_samples_after_filter}",
        "",
        "Counts by class",
        "=" * 70,
        summary_counts.to_string(index=False),
        "",
    ]
    if dry_run:
        lines.extend([
            "Dry-run note",
            "=" * 70,
            "Selected samples were saved for inspection but not removed from CNN training.",
            "",
        ])
    return "\n".join(lines)


def _summarize_pca_filter_counts(filter_metrics):
    return (
        filter_metrics.groupby("class_name", sort=True)
        .agg(
            train_samples=("sample_index", "size"),
            pca_candidates=("pca_filter_candidate", "sum"),
            eligible_candidates=("pca_filter_eligible", "sum"),
            selected_samples=("pca_filter_selected", "sum"),
            removed_samples=("pca_filter_removed", "sum"),
        )
        .reset_index()
    )


def _resolve_training_filter_classes(
    configured_classes,
    config_name,
    require_nonempty,
):
    if configured_classes is None:
        classes = []
    elif isinstance(configured_classes, str):
        classes = [configured_classes]
    else:
        classes = [str(class_name) for class_name in configured_classes]

    if require_nonempty and not classes:
        raise ValueError(f"{config_name} must not be empty")
    return classes


def _resolve_max_removed_fraction(configured_fraction):
    max_removed_fraction = float(configured_fraction)
    if not 0.0 <= max_removed_fraction <= 1.0:
        raise ValueError(
            "training_filter.max_removed_fraction_per_class must be "
            "between zero and one"
        )
    return max_removed_fraction


def _coerce_boolean_column(values, column_name):
    if pd.api.types.is_bool_dtype(values):
        return values.to_numpy(dtype=bool)

    normalized = values.astype(str).str.strip().str.lower()
    true_values = {"true", "1", "yes", "y"}
    false_values = {"false", "0", "no", "n"}
    valid_values = true_values | false_values
    invalid = ~normalized.isin(valid_values)
    if invalid.any():
        invalid_values = values[invalid].head(10).tolist()
        raise ValueError(
            f"Column {column_name} contains non-boolean values: "
            f"{invalid_values}"
        )
    return normalized.isin(true_values).to_numpy(dtype=bool)


def _encode_labels(labels, class_labels):
    label_to_index = {
        int(label): index
        for index, label in enumerate(class_labels)
    }
    try:
        return np.asarray(
            [label_to_index[int(label)] for label in labels],
            dtype=np.int64,
        )
    except KeyError as error:
        raise ValueError(
            f"Dataset contains label {error.args[0]} that is not configured"
        ) from error


def _plot_failure_cases(data, predictions, class_names_by_label, plot_config):
    if not bool(plot_config.get("enabled", False)):
        return []
    if "sample_index" not in predictions.columns:
        print("Skipping failure plots because predictions lack sample_index")
        return []

    from src.dataset_plot import plot_vdf_xz_slice
    from src.vdf_helpers import get_vdf_plot_parameters_from_file

    splits = _resolve_failure_plot_splits(
        plot_config.get("splits", ["validation", "test"])
    )
    max_per_pair = int(plot_config.get("max_per_pair", 8))
    if max_per_pair <= 0:
        raise ValueError("failure_plots.max_per_pair must be positive")

    vdflim = float(plot_config.get("vdflim", 2e6))
    failures = predictions[
        (~predictions["correct"])
        & predictions["split"].isin(splits)
    ]
    if failures.empty:
        return []

    output_paths = []
    plot_parameter_cache = {}
    vdf_shape = tuple(data["X"].shape[1:])

    group_columns = ["split", "true_label", "predicted_label"]
    for (split, true_label, predicted_label), group in failures.groupby(
        group_columns,
        sort=True,
    ):
        true_label = int(true_label)
        predicted_label = int(predicted_label)
        true_class_name = class_names_by_label[true_label]
        predicted_class_name = class_names_by_label[predicted_label]

        for _, failure in group.head(max_per_pair).iterrows():
            sample_index = int(failure["sample_index"])
            metadata_row = data["metadata"].iloc[sample_index].to_dict()
            file_location = metadata_row["file_location"]
            cid = int(metadata_row["cid"])
            cache_key = (file_location, cid, vdf_shape)

            if cache_key not in plot_parameter_cache:
                plot_parameter_cache[cache_key] = get_vdf_plot_parameters_from_file(
                    file_location=file_location,
                    cid=cid,
                    vdf_shape=vdf_shape,
                )
            extent, dv, threshold = plot_parameter_cache[cache_key]

            output_path = (
                data["output_dir"]
                / "failure_plots"
                / str(split)
                / f"true_{true_class_name}"
                / f"pred_{predicted_class_name}"
                / _create_failure_plot_filename(metadata_row, sample_index)
            )
            plot_vdf_xz_slice(
                vdf=data["X"][sample_index],
                y_label=true_label,
                metadata_row=metadata_row,
                extent=extent,
                output_path=output_path,
                dv=dv,
                threshold=threshold,
                vdflim=vdflim,
                predicted_class_name=predicted_class_name,
            )
            output_paths.append(output_path)

    return output_paths


def _resolve_failure_plot_splits(configured_splits):
    if isinstance(configured_splits, str):
        return [configured_splits]
    return [str(split) for split in configured_splits]


def _create_failure_plot_filename(metadata_row, sample_index):
    timestep = metadata_row.get("timestep", "unknown")
    cid = metadata_row.get("cid", "unknown")
    return f"sample_{sample_index:06d}_t{timestep}_cid{cid}.png"


def _resolve_class_weight(configured_class_weight):
    if configured_class_weight is None:
        return "none"

    class_weight = str(configured_class_weight).lower()
    if class_weight in {"false", "no", "none", "unweighted"}:
        return "none"
    if class_weight in {"soft", "sqrt", "sqrt_balanced", "soft_balanced"}:
        return "sqrt_balanced"
    if class_weight in {"true", "yes", "balanced", "weight", "weights", "weighted"}:
        return "balanced"

    raise ValueError("class_weight must be 'none', 'sqrt_balanced', or 'balanced'")


def _create_class_weights(targets, n_classes, class_weight):
    if class_weight == "none":
        return None

    class_counts = np.bincount(targets, minlength=n_classes).astype(float)
    if np.any(class_counts == 0.0):
        raise ValueError("Cannot create class weights for empty classes")

    weights = len(targets) / (n_classes * class_counts)
    if class_weight == "sqrt_balanced":
        weights = np.sqrt(weights)

    return weights.astype(np.float32)


def _resolve_batch_size(configured_batch_size, n_samples):
    if configured_batch_size == "auto":
        return min(32, n_samples)
    batch_size = min(int(configured_batch_size), n_samples)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive or 'auto'")
    return batch_size


def _resolve_prediction_batch_size(configured_batch_size, model_batch_size, n_samples):
    prediction_batch_size = _resolve_batch_size(configured_batch_size, n_samples)
    return min(prediction_batch_size, model_batch_size)


def _resolve_adaptive_pool_shape(configured_shape):
    if configured_shape is None:
        configured_shape = [4, 4]

    if isinstance(configured_shape, str):
        values = [int(value.strip()) for value in configured_shape.split(",")]
    else:
        values = [int(value) for value in configured_shape]

    if len(values) != 2 or any(value <= 0 for value in values):
        raise ValueError("adaptive_pool_shape must contain two positive integers")

    return tuple(values)


def _resolve_device(device_name):
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def _set_random_seed(random_seed, deterministic):
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)
    torch.use_deterministic_algorithms(deterministic)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = deterministic
