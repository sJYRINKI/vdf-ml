import numpy as np
import torch
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import os

os.environ["PTNOLATEX"] = "1"

from src.training import (
    create_metrics_text,
    create_predictions,
    evaluate_model,
    load_training_data,
    save_training_artifacts,
)


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
        prediction_batch_size=64,
    ):
        super().__init__()

        self.input_size = int(input_size)
        self.image_size = int(np.sqrt(self.input_size))
        self.channels = tuple(int(channel) for channel in channels)
        self.classifier_size = int(classifier_size)
        self.dropout = float(dropout)
        self.classes_ = np.asarray(class_labels, dtype=int)
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

        convolution_layers.append(nn.AdaptiveAvgPool2d((4, 4)))
        self.convolutions = nn.Sequential(*convolution_layers)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.channels[-1] * 4 * 4, self.classifier_size),
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

    data = load_training_data(
        config=config,
        dataset_id=dataset_id,
        model_id=model_id,
        target_kind="multiclass",
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
