from pathlib import Path
import random

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.training import (
    create_metrics_text,
    create_predictions,
    evaluate_model,
    load_training_data,
    save_training_artifacts,
)


class MultilayerPerceptron(nn.Module):
    """
    Classify flattened VDF features with a feed-forward neural network.

    Parameters
    ----------
    input_size : int
        Number of input features.
    hidden_layer_sizes : sequence of int
        Number of neurons in each hidden layer.
    n_classes : int
        Number of output classes.
    feature_mean : array-like of float
        Training-feature means used for standardization.
    feature_scale : array-like of float
        Training-feature scales used for standardization.
    activation : {"relu"}, optional
        Hidden-layer activation function.
    """

    def __init__(
        self,
        input_size,
        hidden_layer_sizes,
        n_classes,
        feature_mean,
        feature_scale,
        activation="relu",
    ):
        super().__init__()

        if activation != "relu":
            raise ValueError(
                "PyTorch multilayer perceptron currently supports only "
                "activation='relu'"
            )

        self.input_size = int(input_size)
        self.hidden_layer_sizes = tuple(int(size) for size in hidden_layer_sizes)
        self.n_classes = int(n_classes)
        self.activation_name = activation

        self.register_buffer(
            "feature_mean",
            torch.as_tensor(feature_mean, dtype=torch.float32),
        )
        self.register_buffer(
            "feature_scale",
            torch.as_tensor(feature_scale, dtype=torch.float32),
        )

        layers = []
        layer_sizes = (
            self.input_size,
            *self.hidden_layer_sizes,
            self.n_classes,
        )
        for layer_index, (input_features, output_features) in enumerate(
            zip(layer_sizes[:-1], layer_sizes[1:])
        ):
            layers.append(nn.Linear(input_features, output_features))
            if layer_index < len(layer_sizes) - 2:
                layers.append(nn.ReLU())

        self.layers = nn.Sequential(*layers)

    def forward(self, features):
        """
        Return unnormalized class scores for a feature batch.

        Parameters
        ----------
        features : torch.Tensor
            Feature matrix with shape ``(n_samples, n_features)``.

        Returns
        -------
        torch.Tensor
            Class logits with shape ``(n_samples, n_classes)``.
        """

        standardized_features = (
            features - self.feature_mean
        ) / self.feature_scale
        return self.layers(standardized_features)


class PyTorchClassifierAdapter:
    """
    Provide a scikit-learn-like prediction interface for a PyTorch model.

    Parameters
    ----------
    model : MultilayerPerceptron
        Trained PyTorch model.
    class_labels : array-like of int
        Project labels in model-output order.
    device : str or torch.device, optional
        Device used for prediction.
    batch_size : int, optional
        Number of feature rows predicted at once.
    """

    def __init__(self, model, class_labels, device="cpu", batch_size=4096):
        self.model = model
        self.classes_ = np.asarray(class_labels, dtype=int)
        self.device = torch.device(device)
        self.batch_size = int(batch_size)

        self.model.to(self.device)
        self.model.eval()

    def predict(self, features):
        """
        Predict project class labels.

        Parameters
        ----------
        features : numpy.ndarray
            Feature matrix.

        Returns
        -------
        numpy.ndarray
            Predicted project labels.
        """

        probabilities = self.predict_proba(features)
        class_indices = np.argmax(probabilities, axis=1)
        return self.classes_[class_indices]

    def predict_proba(self, features):
        """
        Predict class probabilities.

        Parameters
        ----------
        features : numpy.ndarray
            Feature matrix.

        Returns
        -------
        numpy.ndarray
            Class probabilities in ``classes_`` order.
        """

        return _predict_probabilities(
            model=self.model,
            features=features,
            device=self.device,
            batch_size=self.batch_size,
        )


class _IndexedFeatureDataset(Dataset):
    def __init__(self, features, targets, indices):
        self.features = features
        self.targets = targets
        self.indices = np.asarray(indices, dtype=int)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        sample_index = self.indices[item]
        features = torch.from_numpy(self.features[sample_index])
        target = int(self.targets[sample_index])
        return features, target


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
    model_config = config["model"]
    class_names_by_label = {
        int(label): class_name
        for class_name, label in labels_config.items()
    }
    class_labels = np.asarray(sorted(class_names_by_label), dtype=int)
    class_names = [class_names_by_label[label] for label in class_labels]
    if len(class_labels) < 2:
        raise ValueError("At least two configured classes are required")

    hidden_layer_sizes = tuple(
        int(size)
        for size in model_config.get("hidden_layer_sizes", [128, 64])
    )
    activation = model_config.get("activation", "relu")
    solver = model_config.get("solver", "adam")
    alpha = float(model_config.get("alpha", 0.0001))
    configured_batch_size = model_config.get("batch_size", "auto")
    learning_rate_init = float(model_config.get("learning_rate_init", 0.001))
    max_iter = int(model_config.get("max_iter", 300))
    early_stopping = bool(model_config.get("early_stopping", True))
    validation_fraction = float(model_config.get("validation_fraction", 0.15))
    n_iter_no_change = int(model_config.get("n_iter_no_change", 10))
    tol = float(model_config.get("tol", 1e-4))
    random_state = int(model_config.get("random_state", 1234))
    device_name = model_config.get("device", "auto")
    deterministic = bool(model_config.get("deterministic", False))
    prediction_batch_size = int(model_config.get("prediction_batch_size", 4096))

    _validate_training_config(
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
        solver=solver,
        alpha=alpha,
        learning_rate_init=learning_rate_init,
        max_iter=max_iter,
        early_stopping=early_stopping,
        validation_fraction=validation_fraction,
        n_iter_no_change=n_iter_no_change,
        tol=tol,
        prediction_batch_size=prediction_batch_size,
    )

    data = load_training_data(
        config=config,
        dataset_id=dataset_id,
        model_id=model_id,
        target_kind="multiclass",
    )
    encoded_y_train = _encode_labels(data["y_train"], class_labels)
    _validate_dataset_labels(
        y_train=data["y_train"],
        y_test=data["y_test"],
        class_labels=class_labels,
    )

    fit_indices, validation_indices = _create_validation_split(
        encoded_y_train=encoded_y_train,
        early_stopping=early_stopping,
        validation_fraction=validation_fraction,
        random_state=random_state,
    )
    model_batch_size = _resolve_batch_size(
        configured_batch_size=configured_batch_size,
        n_samples=len(fit_indices),
    )

    scaler = StandardScaler()
    scaler.fit(data["X_train_features"])
    feature_mean = np.asarray(scaler.mean_, dtype=np.float32)
    feature_scale = np.asarray(scaler.scale_, dtype=np.float32)

    device = _resolve_device(device_name)
    _set_random_seeds(
        random_state=random_state,
        deterministic=deterministic,
    )
    model = MultilayerPerceptron(
        input_size=data["X_train_features"].shape[1],
        hidden_layer_sizes=hidden_layer_sizes,
        n_classes=len(class_labels),
        feature_mean=feature_mean,
        feature_scale=feature_scale,
        activation=activation,
    ).to(device)

    print("Configured classes:")
    for label, class_name in zip(class_labels, class_names):
        print(f"  {label}: {class_name}")
    print(f"Training device: {device}")

    training_result = _fit_model(
        model=model,
        features=data["X_train_features"],
        encoded_targets=encoded_y_train,
        fit_indices=fit_indices,
        validation_indices=validation_indices,
        device=device,
        batch_size=model_batch_size,
        prediction_batch_size=prediction_batch_size,
        learning_rate_init=learning_rate_init,
        alpha=alpha,
        max_iter=max_iter,
        tol=tol,
        n_iter_no_change=n_iter_no_change,
        random_state=random_state,
    )

    classifier = PyTorchClassifierAdapter(
        model=model,
        class_labels=class_labels,
        device=device,
        batch_size=prediction_batch_size,
    )
    results = evaluate_model(
        model=classifier,
        data=data,
        report_labels=class_labels,
        target_names=class_names,
    )

    print("PyTorch multilayer perceptron classifier results")
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
    save_pytorch_mlp_checkpoint(
        model=model,
        checkpoint_path=checkpoint_path,
        class_labels=class_labels,
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
    )

    metric_lines = [
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
        f"Iterations without improvement: {n_iter_no_change}",
        f"Tolerance: {tol}",
        f"Random state: {random_state}",
        f"Device: {device}",
        f"Deterministic algorithms: {deterministic}",
        f"PyTorch version: {torch.__version__}",
        f"Classifier classes: {list(classifier.classes_)}",
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
            target_line=None,
            extra_lines=metric_lines,
        ),
    )

    print(checkpoint_path)
    print(data["output_dir"] / "metrics.txt")
    print(predictions)


def save_pytorch_mlp_checkpoint(
    model,
    checkpoint_path,
    class_labels,
    hidden_layer_sizes,
    activation,
):
    """
    Save a PyTorch multilayer perceptron checkpoint.

    Parameters
    ----------
    model : MultilayerPerceptron
        Trained model.
    checkpoint_path : str or pathlib.Path
        Destination checkpoint path.
    class_labels : array-like of int
        Project labels in model-output order.
    hidden_layer_sizes : sequence of int
        Number of neurons in each hidden layer.
    activation : str
        Hidden-layer activation name.
    """

    checkpoint_path = Path(checkpoint_path)
    state_dict = {
        name: value.detach().cpu()
        for name, value in model.state_dict().items()
    }
    checkpoint = {
        "format_version": 1,
        "model_state_dict": state_dict,
        "input_size": model.input_size,
        "hidden_layer_sizes": list(hidden_layer_sizes),
        "n_classes": model.n_classes,
        "activation": activation,
        "class_labels": torch.as_tensor(class_labels, dtype=torch.int64),
    }
    torch.save(checkpoint, checkpoint_path)


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
    PyTorchClassifierAdapter
        Loaded classifier with a NumPy prediction interface.
    """

    checkpoint_path = Path(checkpoint_path)
    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

    _validate_checkpoint(checkpoint)
    input_size = int(checkpoint["input_size"])
    model = MultilayerPerceptron(
        input_size=input_size,
        hidden_layer_sizes=checkpoint["hidden_layer_sizes"],
        n_classes=int(checkpoint["n_classes"]),
        feature_mean=np.zeros(input_size, dtype=np.float32),
        feature_scale=np.ones(input_size, dtype=np.float32),
        activation=checkpoint["activation"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])

    class_labels = checkpoint["class_labels"]
    if isinstance(class_labels, torch.Tensor):
        class_labels = class_labels.cpu().numpy()

    return PyTorchClassifierAdapter(
        model=model,
        class_labels=class_labels,
        device=device,
        batch_size=prediction_batch_size,
    )


def _fit_model(
    model,
    features,
    encoded_targets,
    fit_indices,
    validation_indices,
    device,
    batch_size,
    prediction_batch_size,
    learning_rate_init,
    alpha,
    max_iter,
    tol,
    n_iter_no_change,
    random_state,
):
    dataset = _IndexedFeatureDataset(
        features=features,
        targets=encoded_targets,
        indices=fit_indices,
    )
    generator = torch.Generator()
    generator.manual_seed(random_state)
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
        generator=generator,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate_init,
        weight_decay=0.0,
    )
    loss_function = nn.CrossEntropyLoss()
    best_state_dict = None
    best_validation_accuracy = None
    best_training_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    final_training_loss = float("nan")

    for epoch in range(1, max_iter + 1):
        model.train()
        total_loss = torch.zeros(
            (),
            dtype=torch.float32,
            device=device,
        )
        n_processed = 0

        for feature_batch, target_batch in data_loader:
            feature_batch = feature_batch.to(
                device=device,
                dtype=torch.float32,
                non_blocking=device.type == "cuda",
            )
            target_batch = target_batch.to(
                device=device,
                dtype=torch.int64,
                non_blocking=device.type == "cuda",
            )

            optimizer.zero_grad(set_to_none=True)
            logits = model(feature_batch)
            loss = loss_function(logits, target_batch)
            if alpha > 0.0:
                loss = loss + _l2_penalty(
                    model=model,
                    alpha=alpha,
                    n_samples=len(fit_indices),
                )
            loss.backward()
            optimizer.step()

            batch_sample_count = len(feature_batch)
            total_loss += loss.detach() * batch_sample_count
            n_processed += batch_sample_count

        final_training_loss = float(total_loss.cpu()) / n_processed

        if len(validation_indices) > 0:
            validation_probabilities = _predict_probabilities(
                model=model,
                features=features,
                device=device,
                batch_size=prediction_batch_size,
                indices=validation_indices,
            )
            validation_predictions = np.argmax(
                validation_probabilities,
                axis=1,
            )
            validation_accuracy = float(
                np.mean(
                    validation_predictions
                    == encoded_targets[validation_indices]
                )
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
                best_state_dict = _copy_state_dict_to_cpu(model)

            if significantly_improved:
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
        else:
            significantly_improved = (
                best_training_loss - final_training_loss > tol
            )
            if final_training_loss < best_training_loss:
                best_training_loss = final_training_loss
                best_epoch = epoch

            if significantly_improved:
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

        if epochs_without_improvement >= n_iter_no_change:
            break

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    return {
        "n_epochs": epoch,
        "best_epoch": best_epoch,
        "best_validation_accuracy": best_validation_accuracy,
        "final_training_loss": final_training_loss,
    }


def _predict_probabilities(
    model,
    features,
    device,
    batch_size,
    indices=None,
):
    features = np.asarray(features)
    if features.ndim != 2:
        raise ValueError("Expected a two-dimensional feature matrix")
    if features.shape[1] != model.input_size:
        raise ValueError(
            f"Expected {model.input_size} features, got {features.shape[1]}"
        )

    if indices is None:
        batch_indices = None
        n_samples = len(features)
    else:
        batch_indices = np.asarray(indices, dtype=int)
        n_samples = len(batch_indices)

    probability_batches = []
    model.eval()
    with torch.no_grad():
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            if batch_indices is None:
                feature_batch = features[start:end]
            else:
                feature_batch = features[batch_indices[start:end]]

            feature_tensor = torch.as_tensor(
                feature_batch,
                dtype=torch.float32,
                device=device,
            )
            probabilities = torch.softmax(model(feature_tensor), dim=1)
            probability_batches.append(probabilities.cpu().numpy())

    if not probability_batches:
        return np.empty((0, model.n_classes), dtype=np.float32)

    return np.concatenate(probability_batches, axis=0)


def _create_validation_split(
    encoded_y_train,
    early_stopping,
    validation_fraction,
    random_state,
):
    sample_indices = np.arange(len(encoded_y_train), dtype=int)
    if not early_stopping:
        return sample_indices, np.empty(0, dtype=int)

    try:
        fit_indices, validation_indices = train_test_split(
            sample_indices,
            test_size=validation_fraction,
            random_state=random_state,
            stratify=encoded_y_train,
        )
    except ValueError as error:
        raise ValueError(
            "Could not create a stratified validation split. Check that each "
            "configured class has enough training samples."
        ) from error

    return np.asarray(fit_indices), np.asarray(validation_indices)


def _encode_labels(labels, class_labels):
    label_to_index = {
        int(label): class_index
        for class_index, label in enumerate(class_labels)
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


def _validate_dataset_labels(y_train, y_test, class_labels):
    configured_labels = set(int(label) for label in class_labels)
    observed_labels = set(
        int(label)
        for label in np.concatenate((y_train, y_test))
    )
    unknown_labels = observed_labels - configured_labels
    if unknown_labels:
        raise ValueError(
            f"Dataset labels are missing from config: {sorted(unknown_labels)}"
        )

    missing_training_labels = configured_labels - set(
        int(label) for label in y_train
    )
    if missing_training_labels:
        raise ValueError(
            "Configured classes have no training samples: "
            f"{sorted(missing_training_labels)}"
        )


def _validate_training_config(
    hidden_layer_sizes,
    activation,
    solver,
    alpha,
    learning_rate_init,
    max_iter,
    early_stopping,
    validation_fraction,
    n_iter_no_change,
    tol,
    prediction_batch_size,
):
    if not hidden_layer_sizes or any(size <= 0 for size in hidden_layer_sizes):
        raise ValueError("hidden_layer_sizes must contain positive integers")
    if activation != "relu":
        raise ValueError("PyTorch MLP currently supports only activation='relu'")
    if solver != "adam":
        raise ValueError("PyTorch MLP currently supports only solver='adam'")
    if alpha < 0.0:
        raise ValueError("alpha must be non-negative")
    if learning_rate_init <= 0.0:
        raise ValueError("learning_rate_init must be positive")
    if max_iter <= 0:
        raise ValueError("max_iter must be positive")
    if early_stopping and not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between zero and one")
    if n_iter_no_change <= 0:
        raise ValueError("n_iter_no_change must be positive")
    if tol < 0.0:
        raise ValueError("tol must be non-negative")
    if prediction_batch_size <= 0:
        raise ValueError("prediction_batch_size must be positive")


def _resolve_batch_size(configured_batch_size, n_samples):
    if configured_batch_size == "auto":
        return min(200, n_samples)

    batch_size = int(configured_batch_size)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive or 'auto'")
    return min(batch_size, n_samples)


def _resolve_device(device_name):
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"

    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def _set_random_seeds(random_state, deterministic):
    random.seed(random_state)
    np.random.seed(random_state)
    torch.manual_seed(random_state)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_state)

    if deterministic:
        torch.use_deterministic_algorithms(True)
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True


def _l2_penalty(model, alpha, n_samples):
    squared_weights = [
        layer.weight.square().sum()
        for layer in model.layers
        if isinstance(layer, nn.Linear)
    ]
    return 0.5 * alpha * torch.stack(squared_weights).sum() / n_samples


def _copy_state_dict_to_cpu(model):
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }


def _validate_checkpoint(checkpoint):
    required_keys = {
        "format_version",
        "model_state_dict",
        "input_size",
        "hidden_layer_sizes",
        "n_classes",
        "activation",
        "class_labels",
    }
    missing_keys = required_keys - set(checkpoint)
    if missing_keys:
        raise ValueError(
            f"PyTorch checkpoint is missing keys: {sorted(missing_keys)}"
        )
    if int(checkpoint["format_version"]) != 1:
        raise ValueError(
            "Unsupported PyTorch checkpoint format version: "
            f"{checkpoint['format_version']}"
        )

    class_labels = checkpoint["class_labels"]
    if len(class_labels) != int(checkpoint["n_classes"]):
        raise ValueError(
            "PyTorch checkpoint class-label count does not match the "
            "model output size"
        )
