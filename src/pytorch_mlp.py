import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.training import (
    create_metrics_text,
    create_predictions,
    evaluate_model,
    load_training_data,
    save_training_artifacts,
)


class PyTorchMLPClassifier(nn.Module):
    """
    Classify flattened VDF features with a PyTorch neural network.

    Parameters
    ----------
    input_size : int
        Number of input features.
    hidden_layer_sizes : sequence of int
        Number of neurons in each hidden layer.
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
        hidden_layer_sizes,
        class_labels,
        feature_mean,
        feature_scale,
        prediction_batch_size=4096,
    ):
        super().__init__()

        self.input_size = int(input_size)
        self.hidden_layer_sizes = tuple(
            int(size) for size in hidden_layer_sizes
        )
        self.classes_ = np.asarray(class_labels, dtype=int)
        self.prediction_batch_size = int(prediction_batch_size)
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

        layer_sizes = (
            self.input_size,
            *self.hidden_layer_sizes,
            len(self.classes_),
        )
        layers = []
        for layer_index, (input_features, output_features) in enumerate(
            zip(layer_sizes[:-1], layer_sizes[1:])
        ):
            layers.append(nn.Linear(input_features, output_features))
            if layer_index < len(layer_sizes) - 2:
                layers.append(nn.ReLU())

        self.layers = nn.Sequential(*layers)

    def forward(self, features):
        """Return unnormalized class scores for a feature batch."""

        features = (features - self.feature_mean) / self.feature_scale
        return self.layers(features)

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


def train_pytorch_multilayer_perceptron_classifier(config, dataset_id, model_id):
    """
    Train and save a PyTorch multilayer perceptron classifier.

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
    hidden_layer_sizes = tuple(
        int(size)
        for size in model_config.get("hidden_layer_sizes", [128, 64])
    )
    if (
        model_config.get("activation", "relu") != "relu"
        or model_config.get("solver", "adam") != "adam"
    ):
        raise ValueError("PyTorch MLP supports only relu activation and adam solver")
    if not hidden_layer_sizes or any(size <= 0 for size in hidden_layer_sizes):
        raise ValueError("hidden_layer_sizes must contain positive integers")

    alpha = float(model_config.get("alpha", 0.0001))
    learning_rate = float(model_config.get("learning_rate_init", 0.001))
    max_epochs = int(model_config.get("max_iter", 300))
    early_stopping = bool(model_config.get("early_stopping", True))
    patience = int(model_config.get("n_iter_no_change", 10))
    tolerance = float(model_config.get("tol", 1e-4))
    random_seed = int(model_config.get("random_state", 1234))

    if alpha < 0.0 or learning_rate <= 0.0:
        raise ValueError("alpha must be non-negative and learning rate positive")
    if max_epochs <= 0 or patience <= 0 or tolerance < 0.0:
        raise ValueError("Invalid iteration or early-stopping configuration")

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

    scaler = StandardScaler().fit(data["X_train_features"])
    device_name = model_config.get("device", "auto")
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    deterministic = bool(model_config.get("deterministic", False))
    _set_random_seed(random_seed, deterministic)

    model = PyTorchMLPClassifier(
        input_size=data["X_train_features"].shape[1],
        hidden_layer_sizes=hidden_layer_sizes,
        class_labels=class_labels,
        feature_mean=np.asarray(scaler.mean_, dtype=np.float32),
        feature_scale=np.asarray(scaler.scale_, dtype=np.float32),
        prediction_batch_size=int(
            model_config.get("prediction_batch_size", 4096)
        ),
    ).to(device)
    model_batch_size = model_config.get("batch_size", "auto")
    if model_batch_size == "auto":
        model_batch_size = min(200, len(y_train))
    else:
        model_batch_size = min(int(model_batch_size), len(y_train))
        if model_batch_size <= 0:
            raise ValueError("batch_size must be positive or 'auto'")

    print("Configured classes:")
    for label, class_name in zip(class_labels, class_names):
        print(f"  {label}: {class_name}")
    print(f"Training device: {device}")

    training_result = _fit_model(
        model=model,
        features=data["X_train_features"],
        targets=y_train,
        validation_features=data["X_validation_features"],
        validation_targets=y_validation,
        device=device,
        batch_size=model_batch_size,
        learning_rate=learning_rate,
        alpha=alpha,
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
    print("PyTorch multilayer perceptron classifier results")
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

    checkpoint_path = (
        data["output_dir"]
        / "pytorch_multilayer_perceptron_classifier.pt"
    )
    save_pytorch_mlp_checkpoint(model, checkpoint_path)

    metric_lines = [
        "Configured classes:",
        *[
            f"  {label}: {class_name}"
            for label, class_name in zip(class_labels, class_names)
        ],
        f"Hidden layer sizes: {hidden_layer_sizes}",
        "Activation: relu",
        "Solver: adam",
        f"Alpha: {alpha}",
        f"Model batch size: {model_batch_size}",
        f"Learning rate init: {learning_rate}",
        f"Max iterations: {max_epochs}",
        f"Early stopping: {early_stopping}",
        f"Iterations without improvement: {patience}",
        f"Tolerance: {tolerance}",
        f"Random state: {random_seed}",
        f"Device: {device}",
        f"Deterministic algorithms: {deterministic}",
        f"PyTorch version: {torch.__version__}",
        f"Classifier classes: {list(model.classes_)}",
        f"Epochs: {training_result['n_epochs']}",
        f"Best epoch: {training_result['best_epoch']}",
        f"Final training loss: {training_result['final_training_loss']}",
    ]
    if training_result["best_validation_accuracy"] is not None:
        metric_lines.append(
            "Best validation accuracy: "
            f"{training_result['best_validation_accuracy']}"
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
            title="PyTorch multilayer perceptron classifier evaluation",
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


def load_pytorch_mlp_checkpoint(
    checkpoint_path,
    device="cpu",
    prediction_batch_size=4096,
):
    """
    Load a PyTorch multilayer perceptron checkpoint.

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
    PyTorchMLPClassifier
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
            "Unsupported PyTorch checkpoint format version: "
            f"{checkpoint['format_version']}"
        )
    if checkpoint.get("activation", "relu") != "relu":
        raise ValueError("Checkpoint activation is not supported")

    class_labels = checkpoint["class_labels"]
    if isinstance(class_labels, torch.Tensor):
        class_labels = class_labels.cpu().numpy()
    if len(class_labels) != int(checkpoint["n_classes"]):
        raise ValueError("Checkpoint class-label count does not match model output")

    input_size = int(checkpoint["input_size"])
    model = PyTorchMLPClassifier(
        input_size=input_size,
        hidden_layer_sizes=checkpoint["hidden_layer_sizes"],
        class_labels=class_labels,
        feature_mean=np.zeros(input_size, dtype=np.float32),
        feature_scale=np.ones(input_size, dtype=np.float32),
        prediction_batch_size=prediction_batch_size,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.to(device).eval()


def save_pytorch_mlp_checkpoint(model, checkpoint_path):
    """Save model weights and architecture information."""

    torch.save(
        {
            "format_version": 1,
            "model_state_dict": {
                name: value.detach().cpu()
                for name, value in model.state_dict().items()
            },
            "input_size": model.input_size,
            "hidden_layer_sizes": list(model.hidden_layer_sizes),
            "n_classes": len(model.classes_),
            "activation": "relu",
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
    device,
    batch_size,
    learning_rate,
    alpha,
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
    generator = torch.Generator().manual_seed(random_seed)
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
        generator=generator,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_function = nn.CrossEntropyLoss()
    weights = [
        layer.weight
        for layer in model.layers
        if isinstance(layer, nn.Linear)
    ]
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
            if alpha > 0.0:
                squared_weights = sum(
                    weight.square().sum() for weight in weights
                )
                loss = loss + alpha * squared_weights / (2 * len(dataset))
            loss.backward()
            optimizer.step()
            total_loss += loss.detach() * len(feature_batch)

        training_loss = float(total_loss.cpu()) / len(dataset)
        if early_stopping:
            validation_predictions = np.argmax(
                model.predict_proba(validation_features),
                axis=1,
            )
            score = float(np.mean(validation_predictions == validation_targets))
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
        "best_validation_accuracy": best_score if early_stopping else None,
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


def _set_random_seed(random_seed, deterministic):
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)
    torch.use_deterministic_algorithms(deterministic)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = deterministic
