from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
import os

os.environ["PTNOLATEX"] = "1"

PCA_FILTER_CANDIDATE_COLUMN = "filter_preview_candidate"
PCA_FILTER_METRICS_FILENAME = "pca_filter_metrics.csv"
PCA_FILTER_REMOVED_FILENAME = "pca_filter_removed_samples.csv"
PCA_FILTER_SUMMARY_FILENAME = "pca_filter_summary.txt"
CNN_EMBEDDINGS_FILENAME = "cnn_embeddings.npz"
CNN_EMBEDDING_KNN_METRICS_FILENAME = "cnn_embedding_knn_metrics.csv"
CNN_EMBEDDING_KNN_CANDIDATES_FILENAME = "cnn_embedding_knn_candidates.csv"
CNN_EMBEDDING_KNN_SUMMARY_FILENAME = "cnn_embedding_knn_summary.txt"
CNN_EMBEDDING_KNN_PREVIEW_FILENAME = "cnn_embedding_knn_preview.png"
CNN_EMBEDDING_FILTER_METRICS_FILENAME = "cnn_embedding_knn_filter_metrics.csv"
CNN_EMBEDDING_FILTER_REMOVED_FILENAME = (
    "cnn_embedding_knn_filter_removed_samples.csv"
)
CNN_EMBEDDING_FILTER_SUMMARY_FILENAME = "cnn_embedding_knn_filter_summary.txt"

DEFAULT_DISTANCE_FEATURE_COLUMNS = (
    "distance_to_x_point_re",
    "distance_to_o_point_re",
)
DEFAULT_VECTOR_FEATURE_COLUMNS = (
    "vdf_to_x_point_dx_re",
    "vdf_to_x_point_dz_re",
    "vdf_to_o_point_dx_re",
    "vdf_to_o_point_dz_re",
)

from src.training import (
    create_metrics_text,
    create_predictions,
    evaluate_model,
    load_training_data,
    save_training_artifacts,
)
from src.dataset_pca import plot_dataset_pca


def apply_pytorch_cnn_training_filter_override(
    config,
    training_filter=None,
    dry_run=None,
    run_source_model=None,
    topology_mode=None,
    class_weight=None,
    class_weight_by_class=None,
    sampler=None,
    sampler_weight_by_class=None,
    embedding_k_neighbors=None,
    max_removed_fraction_per_class=None,
    min_point_neighbor_fraction=None,
    min_point_neighbor_fraction_by_class=None,
    max_same_class_fraction=None,
    max_same_class_fraction_by_class=None,
):
    """
    Apply optional command-line training-filter overrides.

    Parameters
    ----------
    config : dict
        CNN training config to update in place.
    training_filter : {"none", "pca", "cnn_embedding_knn"}, optional
        Filter mode override. If omitted, the config is unchanged.
    dry_run : bool, optional
        Optional training-filter dry-run override.
    run_source_model : bool, optional
        Optional CNN embedding source-model override.
    topology_mode : {"none", "input", "auxiliary"}, optional
        Optional topology-mode override. The ``none`` mode also disables the
        distance and vector branches.
    class_weight : str, optional
        Optional model class-weight override.
    class_weight_by_class : sequence of str, optional
        Optional manual class weights as ``class=value``.
    sampler : str, optional
        Optional training sampler override.
    sampler_weight_by_class : sequence of str, optional
        Optional manual sampler weights as ``class=value``.
    embedding_k_neighbors : int, optional
        Optional CNN embedding kNN neighbor-count override.
    max_removed_fraction_per_class : float, optional
        Optional maximum removed fraction per class.
    min_point_neighbor_fraction : float, optional
        Optional point-neighbor fraction threshold for filter candidates.
    min_point_neighbor_fraction_by_class : sequence of str, optional
        Optional class-specific point-neighbor thresholds as ``class=value``.
    max_same_class_fraction : float, optional
        Optional same-class fraction threshold for filter candidates.
    max_same_class_fraction_by_class : sequence of str, optional
        Optional class-specific same-class thresholds as ``class=value``.
    """

    if topology_mode is not None:
        topology_mode = str(topology_mode).strip().lower()
        if topology_mode not in {"none", "input", "auxiliary"}:
            raise ValueError(
                "topology_mode must be one of: none, input, auxiliary"
            )
        model_config = config.setdefault("model", {})
        model_config["topology_mode"] = topology_mode
        if topology_mode == "none":
            model_config.setdefault("distance_branch", {})["enabled"] = False
            model_config.setdefault("vector_branch", {})["enabled"] = False

    if class_weight is not None:
        model_config = config.setdefault("model", {})
        model_config["class_weight"] = class_weight

    if class_weight_by_class:
        model_config = config.setdefault("model", {})
        model_config["class_weights"] = _parse_threshold_overrides(
            class_weight_by_class,
            option_name="class_weight_by_class",
        )

    if sampler is not None:
        sampler_name = str(sampler).lower()
        model_config = config.setdefault("model", {})
        sampler_config = model_config.setdefault("sampler", {})
        if sampler_name in {"none", "false", "off", "disabled"}:
            sampler_config["enabled"] = False
            sampler_config["mode"] = "none"
        elif sampler_name in {"balanced", "sqrt_balanced", "manual"}:
            sampler_config["enabled"] = True
            sampler_config["mode"] = sampler_name
        else:
            raise ValueError(
                "sampler must be one of: none, balanced, "
                "sqrt_balanced, manual"
            )

    if sampler_weight_by_class:
        model_config = config.setdefault("model", {})
        sampler_config = model_config.setdefault("sampler", {})
        sampler_config["class_weights"] = _parse_threshold_overrides(
            sampler_weight_by_class,
            option_name="sampler_weight_by_class",
        )

    if embedding_k_neighbors is not None:
        k_neighbors = int(embedding_k_neighbors)
        if k_neighbors <= 0:
            raise ValueError("embedding_k_neighbors must be positive")
        embedding_config = config.setdefault("cnn_embedding_knn", {})
        embedding_config["k_neighbors"] = k_neighbors
        filter_config = config.setdefault("training_filter", {})
        filter_embedding_config = filter_config.setdefault("cnn_embedding_knn", {})
        filter_embedding_config["k_neighbors"] = k_neighbors

    if training_filter is not None:
        training_filter = str(training_filter).lower()
        filter_config = config.setdefault("training_filter", {})
        if training_filter in {"none", "false", "off", "disabled"}:
            filter_config["enabled"] = False
        elif training_filter == "pca":
            filter_config["enabled"] = True
            filter_config["source"] = "pca"
        elif training_filter in {
            "cnn_embedding_knn",
            "cnn-embedding-knn",
            "cnn_embedding",
            "embedding",
        }:
            filter_config["enabled"] = True
            filter_config["source"] = "cnn_embedding_knn"
        else:
            raise ValueError(
                "training_filter must be one of: none, pca, cnn_embedding_knn"
            )

    if dry_run is not None:
        filter_config = config.setdefault("training_filter", {})
        filter_config["dry_run"] = bool(dry_run)

    if run_source_model is not None:
        filter_config = config.setdefault("training_filter", {})
        embedding_config = filter_config.setdefault("cnn_embedding_knn", {})
        embedding_config["run_source_model"] = bool(run_source_model)

    if max_removed_fraction_per_class is not None:
        filter_config = config.setdefault("training_filter", {})
        filter_config["max_removed_fraction_per_class"] = float(
            max_removed_fraction_per_class
        )

    _apply_filter_threshold_overrides(
        config=config,
        min_point_neighbor_fraction=min_point_neighbor_fraction,
        min_point_neighbor_fraction_by_class=min_point_neighbor_fraction_by_class,
        max_same_class_fraction=max_same_class_fraction,
        max_same_class_fraction_by_class=max_same_class_fraction_by_class,
    )


def _apply_filter_threshold_overrides(
    config,
    min_point_neighbor_fraction=None,
    min_point_neighbor_fraction_by_class=None,
    max_same_class_fraction=None,
    max_same_class_fraction_by_class=None,
):
    """
    Apply optional threshold overrides to PCA and CNN embedding filters.

    Parameters
    ----------
    config : dict
        CNN training config to update in place.
    min_point_neighbor_fraction : float, optional
        Optional point-neighbor fraction threshold.
    min_point_neighbor_fraction_by_class : sequence of str, optional
        Optional class-specific point-neighbor thresholds as ``class=value``.
    max_same_class_fraction : float, optional
        Optional same-class fraction threshold.
    max_same_class_fraction_by_class : sequence of str, optional
        Optional class-specific same-class thresholds as ``class=value``.
    """

    point_thresholds = _parse_threshold_overrides(
        min_point_neighbor_fraction_by_class,
        option_name="min_point_neighbor_fraction_by_class",
    )
    same_thresholds = _parse_threshold_overrides(
        max_same_class_fraction_by_class,
        option_name="max_same_class_fraction_by_class",
    )

    has_overrides = any(
        [
            min_point_neighbor_fraction is not None,
            bool(point_thresholds),
            max_same_class_fraction is not None,
            bool(same_thresholds),
        ]
    )
    if not has_overrides:
        return

    filter_config = config.setdefault("training_filter", {})
    pca_config = filter_config.setdefault("pca", {})
    pca_preview_config = pca_config.setdefault("filter_preview", {})
    embedding_config = filter_config.setdefault("cnn_embedding_knn", {})

    target_configs = [pca_preview_config, embedding_config]
    for target_config in target_configs:
        if min_point_neighbor_fraction is not None:
            target_config["min_point_neighbor_fraction"] = float(
                min_point_neighbor_fraction
            )
        if max_same_class_fraction is not None:
            target_config["max_same_class_fraction"] = float(
                max_same_class_fraction
            )
        if point_thresholds:
            thresholds = dict(
                target_config.get("min_point_neighbor_fraction_by_class", {})
                or {}
            )
            thresholds.update(point_thresholds)
            target_config["min_point_neighbor_fraction_by_class"] = thresholds
        if same_thresholds:
            thresholds = dict(
                target_config.get("max_same_class_fraction_by_class", {}) or {}
            )
            thresholds.update(same_thresholds)
            target_config["max_same_class_fraction_by_class"] = thresholds


def _parse_threshold_overrides(values, option_name):
    """
    Parse class-specific threshold overrides.

    Parameters
    ----------
    values : sequence of str
        Threshold values as ``class=value``.
    option_name : str
        Option name used in error messages.

    Returns
    -------
    dict
        Parsed class-specific thresholds.
    """

    if not values:
        return {}

    thresholds = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"{option_name} values must use class=value")
        class_name, threshold = value.split("=", 1)
        class_name = class_name.strip()
        if not class_name:
            raise ValueError(f"{option_name} class name must not be empty")
        thresholds[class_name] = float(threshold)

    return thresholds


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
    distance_feature_names : sequence of str, optional
        Ordered point-distance feature names.
    distance_hidden_size : int, optional
        Number of neurons in the distance branch.
    distance_loss_weight : float, optional
        Weight of the auxiliary distance-regression loss.
    distance_feature_impute : array-like of float, optional
        Training-only median values used to impute distance features.
    distance_feature_mean : array-like of float, optional
        Training distance-feature means used for standardization.
    distance_feature_scale : array-like of float, optional
        Training distance-feature scales used for standardization.
    vector_feature_names : sequence of str, optional
        Ordered point-vector feature names.
    vector_hidden_size : int, optional
        Number of neurons in the point-vector branch.
    vector_loss_weight : float, optional
        Weight of the auxiliary vector-regression loss.
    vector_feature_impute : array-like of float, optional
        Training-only median values used to impute vector features.
    vector_feature_mean : array-like of float, optional
        Training vector-feature means used for standardization.
    vector_feature_scale : array-like of float, optional
        Training vector-feature scales used for standardization.
    fusion_size : int, optional
        Number of neurons in the fused hidden layer.
    topology_mode : {"input", "auxiliary", "none"}, optional
        Whether topology is a legacy prediction input, an auxiliary training
        target, or disabled.
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
        distance_feature_names=(),
        distance_hidden_size=8,
        distance_loss_weight=0.0,
        distance_feature_impute=None,
        distance_feature_mean=None,
        distance_feature_scale=None,
        vector_feature_names=(),
        vector_hidden_size=8,
        vector_loss_weight=0.0,
        vector_feature_impute=None,
        vector_feature_mean=None,
        vector_feature_scale=None,
        fusion_size=None,
        topology_mode="input",
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
        self.distance_feature_names = tuple(
            str(name) for name in distance_feature_names
        )
        self.vector_feature_names = tuple(
            str(name) for name in vector_feature_names
        )
        self.distance_input_size = len(self.distance_feature_names)
        self.vector_input_size = len(self.vector_feature_names)
        self.distance_hidden_size = int(distance_hidden_size)
        self.vector_hidden_size = int(vector_hidden_size)
        self.distance_loss_weight = float(distance_loss_weight)
        self.vector_loss_weight = float(vector_loss_weight)
        self.topology_enabled = bool(
            self.distance_input_size or self.vector_input_size
        )
        topology_mode = str(topology_mode).strip().lower()
        if topology_mode not in {"input", "auxiliary", "none"}:
            raise ValueError(
                "topology_mode must be 'input', 'auxiliary', or 'none'"
            )
        if self.topology_enabled and topology_mode == "none":
            raise ValueError("topology_mode cannot be none with topology features")
        self.topology_mode = topology_mode if self.topology_enabled else "none"
        self.context_enabled = self.topology_mode == "input"
        self.auxiliary_topology_enabled = self.topology_mode == "auxiliary"
        self.requires_point_context = self.context_enabled
        self.fusion_size = int(
            self.classifier_size if fusion_size is None else fusion_size
        )

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
        if (
            len(self.distance_feature_names) != len(set(self.distance_feature_names))
            or any(not name for name in self.distance_feature_names)
        ):
            raise ValueError("distance_feature_names must be unique and non-empty")
        if (
            len(self.vector_feature_names) != len(set(self.vector_feature_names))
            or any(not name for name in self.vector_feature_names)
        ):
            raise ValueError("vector_feature_names must be unique and non-empty")
        if self.distance_input_size and self.distance_hidden_size <= 0:
            raise ValueError("distance_hidden_size must be positive")
        if self.vector_input_size and self.vector_hidden_size <= 0:
            raise ValueError("vector_hidden_size must be positive")
        if (
            not np.isfinite(self.distance_loss_weight)
            or self.distance_loss_weight < 0.0
        ):
            raise ValueError("distance_loss_weight must be finite and non-negative")
        if (
            not np.isfinite(self.vector_loss_weight)
            or self.vector_loss_weight < 0.0
        ):
            raise ValueError("vector_loss_weight must be finite and non-negative")
        if self.context_enabled and self.fusion_size <= 0:
            raise ValueError("fusion_size must be positive")

        self.register_buffer(
            "feature_mean",
            torch.as_tensor(feature_mean, dtype=torch.float32),
        )
        self.register_buffer(
            "feature_scale",
            torch.as_tensor(feature_scale, dtype=torch.float32),
        )

        if self.distance_input_size:
            self._register_context_preprocessing_buffers(
                prefix="distance",
                impute=distance_feature_impute,
                mean=distance_feature_mean,
                scale=distance_feature_scale,
                input_size=self.distance_input_size,
            )
        else:
            self.distance_feature_impute = None
            self.distance_feature_mean = None
            self.distance_feature_scale = None

        if self.vector_input_size:
            self._register_context_preprocessing_buffers(
                prefix="vector",
                impute=vector_feature_impute,
                mean=vector_feature_mean,
                scale=vector_feature_scale,
                input_size=self.vector_input_size,
            )
        else:
            self.vector_feature_impute = None
            self.vector_feature_mean = None
            self.vector_feature_scale = None

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
        classifier_layers = [
            nn.Flatten(),
            nn.Linear(
                self.channels[-1] * pooled_height * pooled_width,
                self.classifier_size,
            ),
            nn.ReLU(),
        ]
        if not self.context_enabled:
            classifier_layers.extend(
                [
                    nn.Dropout(self.dropout),
                    nn.Linear(self.classifier_size, len(self.classes_)),
                ]
            )
        self.classifier = nn.Sequential(*classifier_layers)

        self.distance_branch = None
        if self.distance_input_size:
            if self.context_enabled:
                self.distance_branch = nn.Sequential(
                    nn.Linear(self.distance_input_size, self.distance_hidden_size),
                    nn.ReLU(),
                )
            else:
                self.distance_branch = nn.Sequential(
                    nn.Linear(self.classifier_size, self.distance_hidden_size),
                    nn.ReLU(),
                    nn.Linear(self.distance_hidden_size, self.distance_input_size),
                )

        self.vector_branch = None
        if self.vector_input_size:
            if self.context_enabled:
                self.vector_branch = nn.Sequential(
                    nn.Linear(self.vector_input_size, self.vector_hidden_size),
                    nn.ReLU(),
                )
            else:
                self.vector_branch = nn.Sequential(
                    nn.Linear(self.classifier_size, self.vector_hidden_size),
                    nn.ReLU(),
                    nn.Linear(self.vector_hidden_size, self.vector_input_size),
                )

        self.fusion = None
        if self.context_enabled:
            fused_input_size = self.classifier_size
            if self.distance_input_size:
                fused_input_size += self.distance_hidden_size
            if self.vector_input_size:
                fused_input_size += self.vector_hidden_size
            self.fusion = nn.Sequential(
                nn.Linear(fused_input_size, self.fusion_size),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.fusion_size, len(self.classes_)),
            )

    def _register_context_preprocessing_buffers(
        self,
        prefix,
        impute,
        mean,
        scale,
        input_size,
    ):
        """
        Register preprocessing values for one optional topology branch.

        Parameters
        ----------
        prefix : str
            Branch name used in the registered buffer names.
        impute : array-like of float or torch.Tensor or None
            Training-only median values used to replace missing features.
        mean : array-like of float or torch.Tensor or None
            Training-feature means used for standardization.
        scale : array-like of float or torch.Tensor or None
            Training-feature scales used for standardization.
        input_size : int
            Number of configured topology values in the branch.
        """

        values_by_name = {
            "impute": impute,
            "mean": mean,
            "scale": scale,
        }
        defaults = {
            "impute": np.zeros(input_size, dtype=np.float32),
            "mean": np.zeros(input_size, dtype=np.float32),
            "scale": np.ones(input_size, dtype=np.float32),
        }
        for value_name, values in values_by_name.items():
            if values is None:
                values = defaults[value_name]
            if isinstance(values, torch.Tensor):
                values = values.detach().cpu().numpy()
            values = np.asarray(values, dtype=np.float32)
            if values.shape != (input_size,):
                raise ValueError(
                    f"{prefix}_feature_{value_name} must have shape "
                    f"({input_size},)"
                )
            if not np.isfinite(values).all():
                raise ValueError(
                    f"{prefix}_feature_{value_name} must be finite"
                )
            if value_name == "scale" and np.any(values <= 0.0):
                raise ValueError(f"{prefix}_feature_scale must be positive")
            self.register_buffer(
                f"{prefix}_feature_{value_name}",
                torch.as_tensor(values, dtype=torch.float32),
            )

    def forward(self, features, distance_features=None, vector_features=None):
        """
        Return unnormalized class scores for a feature batch.

        Parameters
        ----------
        features : torch.Tensor
            Flattened VDF xz-slice features with shape
            ``(n_samples, input_size)``.
        distance_features : torch.Tensor, optional
            Legacy topology-input distances in ``distance_feature_names``
            order. Auxiliary models do not use them for classification.
        vector_features : torch.Tensor, optional
            Legacy topology-input vectors in ``vector_feature_names`` order.
            Auxiliary models do not use them for classification.

        Returns
        -------
        torch.Tensor
            Unnormalized class scores with one row per sample.
        """

        embeddings = self.forward_embeddings(
            features,
            distance_features=distance_features,
            vector_features=vector_features,
        )
        if self.context_enabled:
            embeddings = self.fusion[2](embeddings)
            return self.fusion[3](embeddings)

        embeddings = self.classifier[3](embeddings)
        return self.classifier[4](embeddings)

    def forward_training_outputs(
        self,
        features,
        distance_features=None,
        vector_features=None,
    ):
        """
        Return class scores and optional auxiliary topology predictions.

        Parameters
        ----------
        features : torch.Tensor
            Flattened VDF xz-slice features with shape
            ``(n_samples, input_size)``.
        distance_features : torch.Tensor, optional
            Legacy point-distance inputs for a format-version 2 model.
        vector_features : torch.Tensor, optional
            Legacy point-vector inputs for a format-version 2 model.

        Returns
        -------
        class_scores : torch.Tensor
            Unnormalized class scores with one row per sample.
        distance_predictions : torch.Tensor or None
            Standardized distance predictions for auxiliary training.
        vector_predictions : torch.Tensor or None
            Standardized vector predictions for auxiliary training.
        """

        if not self.auxiliary_topology_enabled:
            class_scores = self(
                features,
                distance_features=distance_features,
                vector_features=vector_features,
            )
            return class_scores, None, None

        embeddings = self.forward_vdf_embeddings(features)
        class_scores = self.classifier[4](self.classifier[3](embeddings))
        distance_predictions = None
        if self.distance_branch is not None:
            distance_predictions = self.distance_branch(embeddings)
        vector_predictions = None
        if self.vector_branch is not None:
            vector_predictions = self.vector_branch(embeddings)
        return class_scores, distance_predictions, vector_predictions

    def forward_embeddings(
        self,
        features,
        distance_features=None,
        vector_features=None,
    ):
        """
        Return the fused or VDF-only hidden-layer embeddings.

        Parameters
        ----------
        features : torch.Tensor
            Flattened VDF xz-slice features with shape
            ``(n_samples, input_size)``.
        distance_features : torch.Tensor, optional
            Legacy topology-input distances. Auxiliary models ignore them.
        vector_features : torch.Tensor, optional
            Legacy topology-input vectors. Auxiliary models ignore them.

        Returns
        -------
        torch.Tensor
            Fused context embeddings when a context branch is enabled,
            otherwise VDF-only embeddings.
        """

        embeddings = [self.forward_vdf_embeddings(features)]
        if self.context_enabled and self.distance_input_size:
            embeddings.append(
                self._forward_context_branch(
                    features=distance_features,
                    branch=self.distance_branch,
                    prefix="distance",
                    input_size=self.distance_input_size,
                )
            )
        if self.context_enabled and self.vector_input_size:
            embeddings.append(
                self._forward_context_branch(
                    features=vector_features,
                    branch=self.vector_branch,
                    prefix="vector",
                    input_size=self.vector_input_size,
                )
            )
        if not self.context_enabled:
            return embeddings[0]

        fused_features = torch.cat(embeddings, dim=1)
        return self.fusion[1](self.fusion[0](fused_features))

    def forward_vdf_embeddings(self, features):
        """
        Return hidden-layer embeddings from the VDF branch.

        Parameters
        ----------
        features : torch.Tensor
            Flattened VDF xz-slice features with shape
            ``(n_samples, input_size)``.

        Returns
        -------
        torch.Tensor
            VDF-branch hidden-layer embeddings.
        """

        features = (features - self.feature_mean) / self.feature_scale
        images = features.reshape(
            -1,
            1,
            self.image_size,
            self.image_size,
        )
        convolution_features = self.convolutions(images)
        flattened_features = self.classifier[0](convolution_features)
        hidden_features = self.classifier[1](flattened_features)
        return self.classifier[2](hidden_features)

    def _forward_context_branch(self, features, branch, prefix, input_size):
        """
        Impute, standardize, and encode one topology-input feature batch.

        Parameters
        ----------
        features : torch.Tensor
            Context features with one row per VDF sample.
        branch : torch.nn.Module
            Linear topology-input branch used to encode standardized values.
        prefix : {"distance", "vector"}
            Branch name used to select its preprocessing buffers.
        input_size : int
            Number of expected input features for the branch.

        Returns
        -------
        torch.Tensor
            Encoded topology-input features.
        """

        if features is None:
            raise ValueError(f"{prefix}_features are required by this model")
        if features.ndim != 2 or features.shape[1] != input_size:
            raise ValueError(
                f"Expected {prefix}_features with shape "
                f"(n_samples, {input_size})"
            )
        impute = getattr(self, f"{prefix}_feature_impute")
        mean = getattr(self, f"{prefix}_feature_mean")
        scale = getattr(self, f"{prefix}_feature_scale")
        features = torch.where(torch.isnan(features), impute, features)
        return branch((features - mean) / scale)

    def predict(self, features, distance_features=None, vector_features=None):
        """
        Predict project class labels.

        Parameters
        ----------
        features : array-like of float
            Flattened VDF xz-slice features.
        distance_features : array-like of float, optional
            Legacy topology-input distances. Auxiliary models ignore them.
        vector_features : array-like of float, optional
            Legacy topology-input vectors. Auxiliary models ignore them.

        Returns
        -------
        numpy.ndarray
            Predicted project class label for each sample.
        """

        class_indices = np.argmax(
            self.predict_proba(
                features,
                distance_features=distance_features,
                vector_features=vector_features,
            ),
            axis=1,
        )
        return self.classes_[class_indices]

    def predict_proba(
        self,
        features,
        distance_features=None,
        vector_features=None,
    ):
        """
        Predict class probabilities in ``classes_`` order.

        Parameters
        ----------
        features : array-like of float
            Flattened VDF xz-slice features.
        distance_features : array-like of float, optional
            Legacy topology-input distances. Auxiliary models ignore them.
        vector_features : array-like of float, optional
            Legacy topology-input vectors. Auxiliary models ignore them.

        Returns
        -------
        numpy.ndarray
            Class probabilities with one row per sample and columns in
            ``classes_`` order.
        """

        features = np.asarray(features, dtype=np.float32)
        if features.ndim != 2 or features.shape[1] != self.input_size:
            raise ValueError(
                "Expected feature matrix with shape "
                f"(n_samples, {self.input_size})"
            )
        if self.requires_point_context:
            distance_features = self._prepare_prediction_context_features(
                features=distance_features,
                prefix="distance",
                input_size=self.distance_input_size,
                n_samples=len(features),
            )
            vector_features = self._prepare_prediction_context_features(
                features=vector_features,
                prefix="vector",
                input_size=self.vector_input_size,
                n_samples=len(features),
            )
        else:
            distance_features = None
            vector_features = None

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
                distance_batch = self._create_prediction_context_batch(
                    features=distance_features,
                    start=start,
                    end=end,
                    device=device,
                )
                vector_batch = self._create_prediction_context_batch(
                    features=vector_features,
                    start=start,
                    end=end,
                    device=device,
                )
                probabilities[start:end] = (
                    torch.softmax(
                        self(
                            feature_batch,
                            distance_features=distance_batch,
                            vector_features=vector_batch,
                        ),
                        dim=1,
                    )
                    .cpu()
                    .numpy()
                )

        if was_training:
            self.train()
        return probabilities

    def _prepare_prediction_context_features(
        self,
        features,
        prefix,
        input_size,
        n_samples,
    ):
        """
        Validate one optional context matrix for prediction.

        Parameters
        ----------
        features : array-like of float or None
            Context feature matrix to validate.
        prefix : {"distance", "vector"}
            Branch name used in validation messages.
        input_size : int
            Number of context features required by the branch.
        n_samples : int
            Number of VDF samples that the context matrix must match.

        Returns
        -------
        numpy.ndarray or None
            Float32 context matrix, or ``None`` when the branch is disabled.
        """

        if input_size == 0:
            if features is not None:
                features = np.asarray(features)
                if features.size:
                    raise ValueError(
                        f"This model does not use {prefix}_features"
                    )
            return None
        if features is None:
            raise ValueError(f"{prefix}_features are required by this model")

        features = np.asarray(features, dtype=np.float32)
        if features.ndim != 2 or features.shape != (n_samples, input_size):
            raise ValueError(
                f"Expected {prefix}_features with shape "
                f"({n_samples}, {input_size})"
            )
        if np.isinf(features).any():
            raise ValueError(f"{prefix}_features must not contain infinity")
        return features

    @staticmethod
    def _create_prediction_context_batch(features, start, end, device):
        """
        Create one optional context tensor for a prediction batch.

        Parameters
        ----------
        features : numpy.ndarray or None
            Full context feature matrix.
        start : int
            Inclusive start row of the prediction batch.
        end : int
            Exclusive end row of the prediction batch.
        device : torch.device
            Device where the prediction tensor is created.

        Returns
        -------
        torch.Tensor or None
            Float32 context tensor for the requested rows, or ``None`` when
            the branch is disabled.
        """

        if features is None:
            return None
        return torch.as_tensor(
            features[start:end],
            dtype=torch.float32,
            device=device,
        )

    def transform_embeddings(
        self,
        features,
        batch_size=None,
        *,
        distance_features=None,
        vector_features=None,
    ):
        """
        Extract CNN hidden-layer embeddings.

        Parameters
        ----------
        features : array-like
            Flattened VDF xz-slice features.
        batch_size : int, optional
            Number of feature rows transformed at once.
        distance_features : array-like, optional
            Legacy topology-input distances. Auxiliary models ignore them.
        vector_features : array-like, optional
            Legacy topology-input vectors. Auxiliary models ignore them.

        Returns
        -------
        numpy.ndarray
            Hidden-layer embeddings.
        """

        features = np.asarray(features, dtype=np.float32)
        if features.ndim != 2 or features.shape[1] != self.input_size:
            raise ValueError(
                "Expected feature matrix with shape "
                f"(n_samples, {self.input_size})"
            )
        if self.requires_point_context:
            distance_features = self._prepare_prediction_context_features(
                features=distance_features,
                prefix="distance",
                input_size=self.distance_input_size,
                n_samples=len(features),
            )
            vector_features = self._prepare_prediction_context_features(
                features=vector_features,
                prefix="vector",
                input_size=self.vector_input_size,
                n_samples=len(features),
            )
        else:
            distance_features = None
            vector_features = None
        if batch_size is None:
            batch_size = self.prediction_batch_size
        batch_size = int(batch_size)
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        device = self.feature_mean.device
        was_training = self.training
        self.eval()
        embeddings = np.empty(
            (
                len(features),
                self.fusion_size if self.context_enabled else self.classifier_size,
            ),
            dtype=np.float32,
        )
        with torch.inference_mode():
            for start in range(0, len(features), batch_size):
                end = start + batch_size
                feature_batch = torch.as_tensor(
                    features[start:end],
                    dtype=torch.float32,
                    device=device,
                )
                distance_batch = self._create_prediction_context_batch(
                    features=distance_features,
                    start=start,
                    end=end,
                    device=device,
                )
                vector_batch = self._create_prediction_context_batch(
                    features=vector_features,
                    start=start,
                    end=end,
                    device=device,
                )
                embeddings[start:end] = (
                    self.forward_embeddings(
                        feature_batch,
                        distance_features=distance_batch,
                        vector_features=vector_batch,
                    )
                    .cpu()
                    .numpy()
                )

        if was_training:
            self.train()
        return embeddings


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
    distance_branch_config = _resolve_context_branch_config(
        model_config=model_config,
        branch_name="distance",
        default_feature_columns=DEFAULT_DISTANCE_FEATURE_COLUMNS,
    )
    vector_branch_config = _resolve_context_branch_config(
        model_config=model_config,
        branch_name="vector",
        default_feature_columns=DEFAULT_VECTOR_FEATURE_COLUMNS,
    )

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
    _add_cnn_context_features(
        data=data,
        distance_branch_config=distance_branch_config,
        vector_branch_config=vector_branch_config,
    )
    embedding_filter_source_result = _run_training_filter_cnn_embedding_knn(
        config=config,
        data=data,
        class_labels=class_labels,
        class_names=class_names,
        class_names_by_label=class_names_by_label,
        channels=channels,
        adaptive_pool_shape=adaptive_pool_shape,
        classifier_size=classifier_size,
        dropout=dropout,
        class_weight=class_weight,
        weight_decay=weight_decay,
        learning_rate=learning_rate,
        max_epochs=max_epochs,
        early_stopping=early_stopping,
        patience=patience,
        tolerance=tolerance,
        random_seed=random_seed,
    )

    training_filter_result = _apply_training_filter(
        data=data,
        config=config.get("training_filter", {}),
    )

    training_state = _train_cnn_model_for_data(
        data=data,
        class_labels=class_labels,
        class_names=class_names,
        channels=channels,
        adaptive_pool_shape=adaptive_pool_shape,
        classifier_size=classifier_size,
        dropout=dropout,
        class_weight=class_weight,
        model_config=model_config,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        max_epochs=max_epochs,
        early_stopping=early_stopping,
        patience=patience,
        tolerance=tolerance,
        random_seed=random_seed,
        stage_name="final",
    )
    model = training_state["model"]
    training_result = training_state["training_result"]
    class_weights = training_state["class_weights"]
    sampler_info = training_state["sampler_info"]
    device = training_state["device"]
    deterministic = training_state["deterministic"]
    model_batch_size = training_state["model_batch_size"]
    prediction_batch_size = training_state["prediction_batch_size"]

    results = evaluate_model(
        model=model,
        data=data,
        report_labels=class_labels,
        target_names=class_names,
        predict_kwargs_by_split=_create_context_kwargs_by_split(model, data),
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
    _add_prediction_probabilities(
        predictions=predictions,
        model=model,
        data=data,
        class_labels=class_labels,
        class_names_by_label=class_names_by_label,
    )

    failure_plot_paths = _plot_failure_cases(
        data=data,
        predictions=predictions,
        class_names_by_label=class_names_by_label,
        plot_config=config.get("failure_plots", {}),
    )
    if (
        embedding_filter_source_result["generated"]
        and bool((config.get("cnn_embedding_knn", {}) or {}).get("enabled", False))
    ):
        embedding_knn_result = {
            "metric_lines": [
                "Final CNN embedding kNN enabled: False",
                "Final CNN embedding kNN skipped: training-filter source "
                "metrics were already saved before filtering",
            ]
        }
    else:
        embedding_knn_result = _run_cnn_embedding_knn_analysis(
            model=model,
            data=data,
            config=config.get("cnn_embedding_knn", {}),
            class_names_by_label=class_names_by_label,
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
        f"Topology mode: {model.topology_mode}",
        f"Distance branch enabled: {model.distance_input_size > 0}",
        f"Distance feature columns: {list(model.distance_feature_names)}",
        f"Distance branch size: {model.distance_hidden_size}",
        f"Distance loss weight: {model.distance_loss_weight}",
        f"Vector branch enabled: {model.vector_input_size > 0}",
        f"Vector feature columns: {list(model.vector_feature_names)}",
        f"Vector branch size: {model.vector_hidden_size}",
        f"Vector loss weight: {model.vector_loss_weight}",
        f"Fusion size: {model.fusion_size if model.context_enabled else 'none'}",
        f"Dropout: {dropout}",
        f"Class weight: {class_weight}",
        f"Sampler enabled: {sampler_info['enabled']}",
        f"Sampler mode: {sampler_info['mode']}",
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
        *embedding_filter_source_result["metric_lines"],
        *training_filter_result["metric_lines"],
        f"Epochs: {training_result['n_epochs']}",
        f"Best epoch: {training_result['best_epoch']}",
        f"Final training loss: {training_result['final_training_loss']}",
        *embedding_knn_result["metric_lines"],
        f"Failure plots saved: {len(failure_plot_paths)}",
    ]
    if failure_plot_paths:
        metric_lines.append(f"Failure plot directory: {data['output_dir'] / 'failure_plots'}")
    if class_weights is not None:
        metric_lines.append(f"Class weights: {class_weights.tolist()}")
    if sampler_info["class_weights"] is not None:
        metric_lines.append(
            f"Sampler class weights: {sampler_info['class_weights'].tolist()}"
        )
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
            "topology_mode": np.asarray(model.topology_mode),
            "distance_feature_names": np.asarray(model.distance_feature_names),
            "distance_loss_weight": model.distance_loss_weight,
            "vector_feature_names": np.asarray(model.vector_feature_names),
            "vector_loss_weight": model.vector_loss_weight,
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


def _add_prediction_probabilities(
    predictions,
    model,
    data,
    class_labels,
    class_names_by_label,
):
    """
    Add class probability columns to the prediction table.

    Parameters
    ----------
    predictions : pandas.DataFrame
        Prediction rows in train, validation, and test order.
    model : PyTorchCNNClassifier
        Trained CNN classifier.
    data : dict
        Training data returned by ``load_training_data``.
    class_labels : numpy.ndarray
        Project labels in model-output order.
    class_names_by_label : dict
        Mapping from integer label to class name.
    """

    probabilities = np.concatenate(
        [
            model.predict_proba(
                data["X_train_features"],
                **_create_context_kwargs(model, data, "train"),
            ),
            model.predict_proba(
                data["X_validation_features"],
                **_create_context_kwargs(model, data, "validation"),
            ),
            model.predict_proba(
                data["X_test_features"],
                **_create_context_kwargs(model, data, "test"),
            ),
        ],
        axis=0,
    )
    if len(probabilities) != len(predictions):
        raise ValueError("Prediction probability count does not match predictions")

    label_to_column = {}
    for column_index, label in enumerate(class_labels):
        class_name = class_names_by_label[int(label)]
        column_name = f"probability_{class_name}"
        predictions[column_name] = probabilities[:, column_index]
        label_to_column[int(label)] = column_index

    true_columns = predictions["true_label"].map(label_to_column).to_numpy(dtype=int)
    predicted_columns = predictions["predicted_label"].map(label_to_column).to_numpy(
        dtype=int
    )
    row_indices = np.arange(len(predictions))
    predictions["true_class_probability"] = probabilities[row_indices, true_columns]
    predictions["predicted_class_probability"] = probabilities[
        row_indices,
        predicted_columns,
    ]
    predictions["max_class_probability"] = probabilities.max(axis=1)


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

    format_version = int(checkpoint["format_version"])
    if format_version not in {1, 2, 3}:
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
    if format_version == 1:
        distance_feature_names = ()
        vector_feature_names = ()
        topology_mode = "none"
    else:
        distance_feature_names = tuple(
            checkpoint.get("distance_feature_names", ())
        )
        vector_feature_names = tuple(checkpoint.get("vector_feature_names", ()))
        if not distance_feature_names and not vector_feature_names:
            raise ValueError(
                f"Format-version {format_version} CNN checkpoint has no "
                "topology features"
            )
        topology_mode = "input" if format_version == 2 else "auxiliary"
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
        distance_feature_names=distance_feature_names,
        distance_hidden_size=int(checkpoint.get("distance_hidden_size", 8)),
        distance_loss_weight=float(
            checkpoint.get("distance_loss_weight", 0.0)
        ),
        distance_feature_impute=checkpoint.get("distance_feature_impute"),
        distance_feature_mean=checkpoint.get("distance_feature_mean"),
        distance_feature_scale=checkpoint.get("distance_feature_scale"),
        vector_feature_names=vector_feature_names,
        vector_hidden_size=int(checkpoint.get("vector_hidden_size", 8)),
        vector_loss_weight=float(checkpoint.get("vector_loss_weight", 0.0)),
        vector_feature_impute=checkpoint.get("vector_feature_impute"),
        vector_feature_mean=checkpoint.get("vector_feature_mean"),
        vector_feature_scale=checkpoint.get("vector_feature_scale"),
        fusion_size=int(
            checkpoint.get("fusion_size", checkpoint["classifier_size"])
        ),
        topology_mode=topology_mode,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.to(device).eval()


def save_pytorch_cnn_checkpoint(model, checkpoint_path):
    """
    Save model weights, preprocessing values, and architecture information.

    Parameters
    ----------
    model : PyTorchCNNClassifier
        Trained CNN classifier to save.
    checkpoint_path : str or pathlib.Path
        Output path for the PyTorch checkpoint.
    """

    if model.context_enabled:
        format_version = 2
    elif model.auxiliary_topology_enabled:
        format_version = 3
    else:
        format_version = 1

    checkpoint = {
        "format_version": format_version,
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
    }
    if model.topology_enabled:
        checkpoint.update(
            {
                "distance_feature_names": list(model.distance_feature_names),
                "distance_hidden_size": model.distance_hidden_size,
                "distance_feature_impute": _cpu_buffer(
                    model.distance_feature_impute
                ),
                "distance_feature_mean": _cpu_buffer(model.distance_feature_mean),
                "distance_feature_scale": _cpu_buffer(
                    model.distance_feature_scale
                ),
                "vector_feature_names": list(model.vector_feature_names),
                "vector_hidden_size": model.vector_hidden_size,
                "vector_feature_impute": _cpu_buffer(model.vector_feature_impute),
                "vector_feature_mean": _cpu_buffer(model.vector_feature_mean),
                "vector_feature_scale": _cpu_buffer(model.vector_feature_scale),
            }
        )
    if model.context_enabled:
        checkpoint["fusion_size"] = model.fusion_size
    if model.auxiliary_topology_enabled:
        checkpoint.update(
            {
                "distance_loss_weight": model.distance_loss_weight,
                "vector_loss_weight": model.vector_loss_weight,
            }
        )

    torch.save(checkpoint, checkpoint_path)


def _cpu_buffer(buffer):
    """
    Return an optional model buffer on the CPU.

    Parameters
    ----------
    buffer : torch.Tensor or None
        Registered model buffer to copy to the CPU.

    Returns
    -------
    torch.Tensor or None
        CPU buffer, or ``None`` when the branch has no buffer.
    """

    if buffer is None:
        return None
    return buffer.detach().cpu()


def _resolve_context_branch_config(
    model_config,
    branch_name,
    default_feature_columns,
):
    """
    Resolve one optional CNN context-branch configuration.

    Parameters
    ----------
    model_config : dict
        CNN model configuration.
    branch_name : {"distance", "vector"}
        Name of the topology branch to resolve.
    default_feature_columns : sequence of str
        Metadata columns used when the branch does not configure columns.

    Returns
    -------
    dict
        Validated branch settings, feature columns, hidden size, and loss
        weight.
    """

    config = dict(model_config.get(f"{branch_name}_branch", {}) or {})
    feature_columns = config.get("feature_columns", default_feature_columns)
    if isinstance(feature_columns, str):
        feature_columns = [feature_columns]
    feature_columns = tuple(str(column) for column in feature_columns)
    if len(feature_columns) != len(set(feature_columns)):
        raise ValueError(
            f"model.{branch_name}_branch.feature_columns contains duplicates"
        )

    hidden_size = int(config.get("hidden_size", 8))
    if hidden_size <= 0:
        raise ValueError(f"model.{branch_name}_branch.hidden_size must be positive")
    loss_weight = float(config.get("loss_weight", 0.0))
    if not np.isfinite(loss_weight) or loss_weight < 0.0:
        raise ValueError(
            f"model.{branch_name}_branch.loss_weight must be finite and "
            "non-negative"
        )

    return {
        "enabled": bool(config.get("enabled", False)),
        "feature_columns": feature_columns,
        "hidden_size": hidden_size,
        "loss_weight": loss_weight,
    }


def _add_cnn_context_features(
    data,
    distance_branch_config,
    vector_branch_config,
):
    """
    Extract configured distance and vector matrices from dataset metadata.

    Parameters
    ----------
    data : dict
        Training data and metadata returned by ``load_training_data``. The
        dictionary is updated in place with topology arrays for each split.
    distance_branch_config : dict
        Resolved distance-branch configuration.
    vector_branch_config : dict
        Resolved vector-branch configuration.
    """

    metadata = data["metadata"]
    data["distance_branch_config"] = distance_branch_config
    data["vector_branch_config"] = vector_branch_config

    for branch_name, branch_config in [
        ("distance", distance_branch_config),
        ("vector", vector_branch_config),
    ]:
        if not branch_config["enabled"]:
            data[f"{branch_name}_feature_names"] = ()
            for split_name in ["train", "validation", "test"]:
                data[f"X_{split_name}_{branch_name}_features"] = None
            continue

        feature_names = list(branch_config["feature_columns"])
        missing_columns = [
            column for column in feature_names if column not in metadata.columns
        ]
        if missing_columns:
            raise ValueError(
                f"CNN {branch_name} branch metadata is missing columns: "
                f"{missing_columns}"
            )
        if not feature_names:
            raise ValueError(
                f"CNN {branch_name} branch must configure at least one feature"
            )

        feature_names = tuple(feature_names)
        data[f"{branch_name}_feature_names"] = feature_names
        for split_name in ["train", "validation", "test"]:
            indices = data[f"{split_name}_indices"]
            branch_features = (
                metadata.iloc[indices]
                .loc[:, list(feature_names)]
                .to_numpy(dtype=np.float32, copy=True)
            )
            if np.isinf(branch_features).any():
                raise ValueError(
                    f"CNN {branch_name} metadata features must not contain "
                    "infinity"
                )
            data[f"X_{split_name}_{branch_name}_features"] = branch_features


def _fit_context_preprocessing(features, feature_names, branch_name):
    """
    Fit train-only median imputation and standardization values.

    Parameters
    ----------
    features : array-like of float or None
        Training topology matrix in Earth radii.
    feature_names : sequence of str
        Ordered metadata column names represented by ``features``.
    branch_name : {"distance", "vector"}
        Branch name used in validation messages.

    Returns
    -------
    dict
        Median imputation values, feature means, and feature scales fitted
        from the training split only.
    """

    feature_names = tuple(feature_names)
    if not feature_names:
        return {"impute": None, "mean": None, "scale": None}

    features = np.asarray(features, dtype=np.float32)
    if features.ndim != 2 or features.shape[1] != len(feature_names):
        raise ValueError(
            f"CNN {branch_name} feature matrix does not match configured columns"
        )
    if np.isinf(features).any():
        raise ValueError(
            f"CNN {branch_name} training features must not contain infinity"
        )
    finite = np.isfinite(features)
    all_missing = np.flatnonzero(~finite.any(axis=0))
    if len(all_missing):
        missing_names = [feature_names[index] for index in all_missing]
        raise ValueError(
            f"CNN {branch_name} training columns are entirely nonfinite: "
            f"{missing_names}"
        )

    finite_or_nan = np.where(finite, features, np.nan)
    impute = np.nanmedian(finite_or_nan, axis=0).astype(np.float32, copy=False)
    imputed_features = np.where(finite, features, impute)
    scaler = StandardScaler().fit(imputed_features)
    return {
        "impute": impute,
        "mean": np.asarray(scaler.mean_, dtype=np.float32),
        "scale": np.asarray(scaler.scale_, dtype=np.float32),
    }


def _create_context_kwargs(model, data, split_name):
    """
    Create optional CNN prediction arguments for one dataset split.

    Parameters
    ----------
    model : PyTorchCNNClassifier
        CNN whose prediction-input mode determines whether context is used.
    data : dict
        Training data containing optional distance and vector arrays.
    split_name : {"train", "validation", "test"}
        Dataset split used to select topology-input arrays when required.

    Returns
    -------
    dict
        Optional ``distance_features`` and ``vector_features`` keyword
        arguments accepted by the CNN prediction methods.
    """

    if not model.requires_point_context:
        return {}

    kwargs = {}
    distance_features = data.get(f"X_{split_name}_distance_features")
    if distance_features is not None:
        kwargs["distance_features"] = distance_features
    vector_features = data.get(f"X_{split_name}_vector_features")
    if vector_features is not None:
        kwargs["vector_features"] = vector_features
    return kwargs


def _create_context_kwargs_by_split(model, data):
    """
    Create optional CNN prediction arguments for all dataset splits.

    Parameters
    ----------
    model : PyTorchCNNClassifier
        CNN whose prediction-input mode determines whether context is used.
    data : dict
        Training data containing optional distance and vector arrays.

    Returns
    -------
    dict
        Prediction keyword arguments keyed by train, validation, and test
        split names.
    """

    return {
        split_name: _create_context_kwargs(model, data, split_name)
        for split_name in ["train", "validation", "test"]
    }


def _train_cnn_model_for_data(
    data,
    class_labels,
    class_names,
    channels,
    adaptive_pool_shape,
    classifier_size,
    dropout,
    class_weight,
    model_config,
    learning_rate,
    weight_decay,
    max_epochs,
    early_stopping,
    patience,
    tolerance,
    random_seed,
    stage_name,
):
    """
    Train a CNN model for the current in-memory training arrays.

    Parameters
    ----------
    data : dict
        Training data returned by ``load_training_data``.
    class_labels : numpy.ndarray
        Project labels in model-output order.
    class_names : list of str
        Class names in model-output order.
    channels : tuple of int
        Number of convolution channels.
    adaptive_pool_shape : tuple of int
        Spatial shape after adaptive pooling.
    classifier_size : int
        Number of hidden-layer neurons.
    dropout : float
        Dropout probability.
    class_weight : str
        Class weighting mode.
    model_config : dict
        CNN model configuration.
    learning_rate : float
        AdamW learning rate.
    weight_decay : float
        AdamW weight decay.
    max_epochs : int
        Maximum number of training epochs.
    early_stopping : bool
        Whether to use validation macro-F1 early stopping.
    patience : int
        Early-stopping patience.
    tolerance : float
        Minimum validation macro-F1 improvement.
    random_seed : int
        Random seed.
    stage_name : str
        Name printed before training.

    Returns
    -------
    dict
        Trained model and training settings.
    """

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
        class_names=class_names,
        configured_class_weights=model_config.get("class_weights", {}),
    )
    sampler, sampler_info = _create_training_sampler(
        targets=y_train,
        class_names=class_names,
        sampler_config=model_config.get("sampler", {}),
        random_seed=random_seed,
    )

    scaler = StandardScaler().fit(data["X_train_features"])
    distance_preprocessing = _fit_context_preprocessing(
        features=data.get("X_train_distance_features"),
        feature_names=data.get("distance_feature_names", ()),
        branch_name="distance",
    )
    vector_preprocessing = _fit_context_preprocessing(
        features=data.get("X_train_vector_features"),
        feature_names=data.get("vector_feature_names", ()),
        branch_name="vector",
    )
    distance_branch_config = data["distance_branch_config"]
    vector_branch_config = data["vector_branch_config"]
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
        distance_feature_names=data["distance_feature_names"],
        distance_hidden_size=distance_branch_config["hidden_size"],
        distance_loss_weight=distance_branch_config["loss_weight"],
        distance_feature_impute=distance_preprocessing["impute"],
        distance_feature_mean=distance_preprocessing["mean"],
        distance_feature_scale=distance_preprocessing["scale"],
        vector_feature_names=data["vector_feature_names"],
        vector_hidden_size=vector_branch_config["hidden_size"],
        vector_loss_weight=vector_branch_config["loss_weight"],
        vector_feature_impute=vector_preprocessing["impute"],
        vector_feature_mean=vector_preprocessing["mean"],
        vector_feature_scale=vector_preprocessing["scale"],
        fusion_size=int(model_config.get("fusion_size", classifier_size)),
        topology_mode=model_config.get("topology_mode", "input"),
    ).to(device)

    print("Configured classes:")
    for label, class_name in zip(class_labels, class_names):
        print(f"  {label}: {class_name}")
    print(f"CNN stage: {stage_name}")
    print(f"CNN input image: {model.image_size} x {model.image_size}")
    print(f"Training device: {device}")

    training_result = _fit_model(
        model=model,
        features=data["X_train_features"],
        distance_features=data.get("X_train_distance_features"),
        vector_features=data.get("X_train_vector_features"),
        targets=y_train,
        validation_features=data["X_validation_features"],
        validation_distance_features=data.get(
            "X_validation_distance_features"
        ),
        validation_vector_features=data.get("X_validation_vector_features"),
        validation_targets=y_validation,
        class_weights=class_weights,
        sampler=sampler,
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

    return {
        "model": model,
        "training_result": training_result,
        "class_weights": class_weights,
        "sampler_info": sampler_info,
        "device": device,
        "deterministic": deterministic,
        "model_batch_size": model_batch_size,
        "prediction_batch_size": prediction_batch_size,
    }


def _fit_model(
    model,
    features,
    distance_features,
    vector_features,
    targets,
    validation_features,
    validation_distance_features,
    validation_vector_features,
    validation_targets,
    class_weights,
    sampler,
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
    """
    Fit a CNN with optional topology-input or auxiliary topology branches.

    Parameters
    ----------
    model : PyTorchCNNClassifier
        CNN classifier to train.
    features : array-like of float
        Flattened training VDF xz-slice features.
    distance_features : array-like of float or None
        Training point distances in Earth radii in the model's configured
        order.
    vector_features : array-like of float or None
        Training VDF-to-point vector components in Earth radii in the model's
        configured order.
    targets : numpy.ndarray
        Encoded training class indices.
    validation_features : array-like of float
        Flattened validation VDF xz-slice features.
    validation_distance_features : array-like of float or None
        Validation distance features in the model's configured order.
    validation_vector_features : array-like of float or None
        Validation vector features in the model's configured order.
    validation_targets : numpy.ndarray
        Encoded validation class indices.
    class_weights : numpy.ndarray or None
        Optional class weights in model-output order.
    sampler : torch.utils.data.Sampler or None
        Optional sampler for selecting training rows.
    device : torch.device
        Device used for model fitting.
    batch_size : int
        Number of training samples in each optimization batch.
    learning_rate : float
        AdamW learning rate.
    weight_decay : float
        AdamW weight-decay coefficient.
    max_epochs : int
        Maximum number of training epochs.
    early_stopping : bool
        Whether to stop using validation macro F1.
    tolerance : float
        Minimum score improvement that resets early-stopping patience.
    patience : int
        Number of epochs without improvement before stopping.
    random_seed : int
        Seed used for shuffled training batches.

    Returns
    -------
    dict
        Epoch count, best epoch and validation score, and final training loss.
    """

    features = np.asarray(features, dtype=np.float32)
    distance_features = model._prepare_prediction_context_features(
        features=distance_features,
        prefix="distance",
        input_size=model.distance_input_size,
        n_samples=len(features),
    )
    vector_features = model._prepare_prediction_context_features(
        features=vector_features,
        prefix="vector",
        input_size=model.vector_input_size,
        n_samples=len(features),
    )
    dataset_tensors = [torch.from_numpy(features)]
    if distance_features is not None:
        dataset_tensors.append(torch.from_numpy(distance_features))
    if vector_features is not None:
        dataset_tensors.append(torch.from_numpy(vector_features))
    dataset_tensors.append(torch.from_numpy(targets))
    dataset = TensorDataset(*dataset_tensors)
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=0,
        pin_memory=device.type == "cuda",
        generator=None if sampler is not None else torch.Generator().manual_seed(
            random_seed
        ),
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

        for batch in data_loader:
            feature_batch = batch[0]
            batch_position = 1
            distance_batch = None
            if model.distance_input_size:
                distance_batch = batch[batch_position]
                batch_position += 1
            vector_batch = None
            if model.vector_input_size:
                vector_batch = batch[batch_position]
            target_batch = batch[-1]
            feature_batch = feature_batch.to(
                device,
                non_blocking=device.type == "cuda",
            )
            if distance_batch is not None:
                distance_batch = distance_batch.to(
                    device,
                    non_blocking=device.type == "cuda",
                )
            if vector_batch is not None:
                vector_batch = vector_batch.to(
                    device,
                    non_blocking=device.type == "cuda",
                )
            target_batch = target_batch.to(
                device,
                non_blocking=device.type == "cuda",
            )
            optimizer.zero_grad(set_to_none=True)
            (
                class_scores,
                distance_predictions,
                vector_predictions,
            ) = model.forward_training_outputs(
                feature_batch,
                distance_features=distance_batch,
                vector_features=vector_batch,
            )
            loss = loss_function(class_scores, target_batch)
            if distance_predictions is not None:
                loss = loss + _create_auxiliary_topology_loss(
                    model=model,
                    predictions=distance_predictions,
                    targets=distance_batch,
                    prefix="distance",
                )
            if vector_predictions is not None:
                loss = loss + _create_auxiliary_topology_loss(
                    model=model,
                    predictions=vector_predictions,
                    targets=vector_batch,
                    prefix="vector",
                )
            loss.backward()
            optimizer.step()
            total_loss += loss.detach() * len(feature_batch)

        training_loss = float(total_loss.cpu()) / len(dataset)
        if early_stopping:
            validation_context = {}
            if model.requires_point_context:
                validation_context = {
                    "distance_features": validation_distance_features,
                    "vector_features": validation_vector_features,
                }
            validation_predictions = np.argmax(
                model.predict_proba(
                    validation_features,
                    **validation_context,
                ),
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


def _create_auxiliary_topology_loss(model, predictions, targets, prefix):
    """
    Return one weighted auxiliary topology regression loss.

    Parameters
    ----------
    model : PyTorchCNNClassifier
        CNN containing training-only topology preprocessing values.
    predictions : torch.Tensor
        Predicted standardized topology values.
    targets : torch.Tensor
        Raw topology values in Earth radii, with missing values as NaN.
    prefix : {"distance", "vector"}
        Topology branch name.

    Returns
    -------
    torch.Tensor
        Weighted Smooth L1 loss for the topology branch.
    """

    impute = getattr(model, f"{prefix}_feature_impute")
    mean = getattr(model, f"{prefix}_feature_mean")
    scale = getattr(model, f"{prefix}_feature_scale")
    loss_weight = getattr(model, f"{prefix}_loss_weight")
    targets = torch.where(torch.isnan(targets), impute, targets)
    standardized_targets = (targets - mean) / scale
    return loss_weight * F.smooth_l1_loss(
        predictions,
        standardized_targets,
    )


def _run_training_filter_cnn_embedding_knn(
    config,
    data,
    class_labels,
    class_names,
    class_names_by_label,
    channels,
    adaptive_pool_shape,
    classifier_size,
    dropout,
    class_weight,
    weight_decay,
    learning_rate,
    max_epochs,
    early_stopping,
    patience,
    tolerance,
    random_seed,
):
    """
    Train a source CNN and save embedding kNN metrics for filtering.

    Parameters
    ----------
    config : dict
        Full CNN training config.
    data : dict
        Training data returned by ``load_training_data``.
    class_labels : numpy.ndarray
        Project labels in model-output order.
    class_names : list of str
        Class names in model-output order.
    class_names_by_label : dict
        Mapping from integer label to class name.
    channels : tuple of int
        Number of convolution channels.
    adaptive_pool_shape : tuple of int
        Spatial shape after adaptive pooling.
    classifier_size : int
        Number of hidden-layer neurons.
    dropout : float
        Dropout probability.
    class_weight : str
        Class weighting mode.
    weight_decay : float
        AdamW weight decay.
    learning_rate : float
        AdamW learning rate.
    max_epochs : int
        Maximum number of training epochs.
    early_stopping : bool
        Whether to use validation macro-F1 early stopping.
    patience : int
        Early-stopping patience.
    tolerance : float
        Minimum validation macro-F1 improvement.
    random_seed : int
        Random seed.

    Returns
    -------
    dict
        Metric lines describing source embedding generation.
    """

    filter_config = config.get("training_filter", {}) or {}
    source = str(filter_config.get("source", "pca")).lower()
    if (
        not bool(filter_config.get("enabled", False))
        or not _is_cnn_embedding_filter_source(source)
    ):
        return {"metric_lines": [], "generated": False}

    embedding_filter_config = filter_config.get("cnn_embedding_knn", {}) or {}
    run_source_model = bool(embedding_filter_config.get("run_source_model", True))
    if not run_source_model:
        return {
            "metric_lines": [
                "CNN embedding filter source model enabled: False",
                "CNN embedding filter source metrics: existing file required",
            ],
            "generated": False,
        }

    print("Training CNN embedding kNN filter source model")
    source_state = _train_cnn_model_for_data(
        data=data,
        class_labels=class_labels,
        class_names=class_names,
        channels=channels,
        adaptive_pool_shape=adaptive_pool_shape,
        classifier_size=classifier_size,
        dropout=dropout,
        class_weight=class_weight,
        model_config=config["model"],
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        max_epochs=max_epochs,
        early_stopping=early_stopping,
        patience=patience,
        tolerance=tolerance,
        random_seed=random_seed,
        stage_name="embedding_filter_source",
    )
    source_embedding_config = _create_training_filter_embedding_knn_config(
        config=config,
        filter_config=filter_config,
    )
    source_analysis_result = _run_cnn_embedding_knn_analysis(
        model=source_state["model"],
        data=data,
        config=source_embedding_config,
        class_names_by_label=class_names_by_label,
    )

    source_device = source_state["device"]
    source_training_result = source_state["training_result"]
    del source_state
    if source_device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "generated": True,
        "metric_lines": [
            "CNN embedding filter source model enabled: True",
            "CNN embedding filter source model trained before final filter: True",
            "CNN embedding filter source epochs: "
            f"{source_training_result['n_epochs']}",
            "CNN embedding filter source best epoch: "
            f"{source_training_result['best_epoch']}",
            "CNN embedding filter source final training loss: "
            f"{source_training_result['final_training_loss']}",
            *source_analysis_result["metric_lines"],
        ],
    }


def _create_training_filter_embedding_knn_config(config, filter_config):
    """
    Create embedding kNN config for the training-filter source model.

    Parameters
    ----------
    config : dict
        Full CNN training config.
    filter_config : dict
        Training filter config.

    Returns
    -------
    dict
        Config accepted by ``_run_cnn_embedding_knn_analysis``.
    """

    embedding_config = dict(config.get("cnn_embedding_knn", {}) or {})
    embedding_filter_config = dict(filter_config.get("cnn_embedding_knn", {}) or {})
    embedding_filter_config.pop("run_source_model", None)
    embedding_config.update(embedding_filter_config)
    embedding_config["enabled"] = True

    for key in [
        "candidate_classes",
        "point_neighbor_classes",
        "protected_classes",
    ]:
        if key in filter_config:
            embedding_config[key] = filter_config[key]
    embedding_config.setdefault("apply_splits", ["train"])

    return embedding_config


def _run_cnn_embedding_knn_analysis(model, data, config, class_names_by_label):
    """
    Save CNN embedding kNN metrics for filter inspection.

    Parameters
    ----------
    model : PyTorchCNNClassifier
        Trained CNN classifier.
    data : dict
        Training data returned by ``load_training_data``.
    config : dict
        CNN embedding kNN configuration.
    class_names_by_label : dict
        Mapping from integer label to class name.

    Returns
    -------
    dict
        Summary lines to include in model metrics.
    """

    config = _resolve_cnn_embedding_knn_config(config)
    if not config["enabled"]:
        return {"metric_lines": ["CNN embedding kNN enabled: False"]}

    embeddings_by_split = _extract_cnn_embeddings_by_split(
        model=model,
        data=data,
        batch_size=config["embedding_batch_size"],
    )
    metrics = _create_cnn_embedding_metrics(
        data=data,
        embeddings_by_split=embeddings_by_split,
        class_names_by_label=class_names_by_label,
    )
    _add_cnn_embedding_neighbor_metrics(
        metrics=metrics,
        embeddings_by_split=embeddings_by_split,
        data=data,
        class_names_by_label=class_names_by_label,
        config=config,
    )
    _add_cnn_embedding_filter_preview(metrics=metrics, config=config)
    _add_cnn_embedding_plot_coordinates(
        metrics=metrics,
        embeddings_by_split=embeddings_by_split,
    )

    output_dir = data["output_dir"]
    embeddings_path = output_dir / CNN_EMBEDDINGS_FILENAME
    metrics_path = output_dir / CNN_EMBEDDING_KNN_METRICS_FILENAME
    candidates_path = output_dir / CNN_EMBEDDING_KNN_CANDIDATES_FILENAME
    summary_path = output_dir / CNN_EMBEDDING_KNN_SUMMARY_FILENAME
    preview_path = output_dir / CNN_EMBEDDING_KNN_PREVIEW_FILENAME

    _save_cnn_embeddings(
        embeddings_by_split=embeddings_by_split,
        data=data,
        output_path=embeddings_path,
    )
    metrics.to_csv(metrics_path, index=False)
    metrics[metrics["embedding_filter_candidate"]].to_csv(
        candidates_path,
        index=False,
    )
    summary_text = _create_cnn_embedding_knn_summary_text(
        metrics=metrics,
        config=config,
        embeddings_path=embeddings_path,
        metrics_path=metrics_path,
        candidates_path=candidates_path,
        preview_path=preview_path,
    )
    with open(summary_path, "w") as summary_file:
        summary_file.write(summary_text)

    preview_saved = False
    if config["plot_enabled"]:
        preview_saved = _plot_cnn_embedding_knn_preview(
            metrics=metrics,
            output_path=preview_path,
            config=config,
        )

    candidate_count = int(metrics["embedding_filter_candidate"].sum())
    print("CNN embedding kNN")
    print(f"  Metrics: {metrics_path}")
    print(f"  Candidates: {candidate_count}")
    if preview_saved:
        print(f"  Preview: {preview_path}")

    return {
        "metric_lines": _create_cnn_embedding_knn_metric_lines(
            metrics=metrics,
            config=config,
            embeddings_path=embeddings_path,
            metrics_path=metrics_path,
            candidates_path=candidates_path,
            summary_path=summary_path,
            preview_path=preview_path if preview_saved else None,
        )
    }


def _resolve_cnn_embedding_knn_config(config):
    """
    Resolve and validate CNN-embedding k-nearest-neighbor settings.

    Parameters
    ----------
    config : dict or None
        CNN-embedding analysis configuration.

    Returns
    -------
    dict
        Validated neighbor, filtering, batching, and plotting settings.
    """

    config = config or {}
    candidate_classes = _resolve_training_filter_classes(
        config.get("candidate_classes", ["exhaust", "dayside"]),
        "cnn_embedding_knn.candidate_classes",
        require_nonempty=True,
    )
    point_neighbor_classes = _resolve_training_filter_classes(
        config.get("point_neighbor_classes", ["x_point", "o_point"]),
        "cnn_embedding_knn.point_neighbor_classes",
        require_nonempty=True,
    )
    protected_classes = _resolve_training_filter_classes(
        config.get("protected_classes", point_neighbor_classes),
        "cnn_embedding_knn.protected_classes",
        require_nonempty=False,
    )
    apply_splits = _resolve_training_filter_classes(
        config.get("apply_splits", ["train"]),
        "cnn_embedding_knn.apply_splits",
        require_nonempty=True,
    )
    min_point_neighbor_fraction = float(
        config.get("min_point_neighbor_fraction", 0.5)
    )
    min_point_neighbor_fraction_by_class = _resolve_class_thresholds(
        config.get("min_point_neighbor_fraction_by_class", {})
    )
    max_same_class_fraction = config.get("max_same_class_fraction")
    if max_same_class_fraction is not None:
        max_same_class_fraction = float(max_same_class_fraction)
    max_same_class_fraction_by_class = _resolve_class_thresholds(
        config.get("max_same_class_fraction_by_class", {})
    )
    k_neighbors = int(config.get("k_neighbors", 25))
    embedding_batch_size = int(config.get("embedding_batch_size", 512))
    neighbor_batch_size = int(config.get("neighbor_batch_size", 4096))
    plot_config = config.get("plot", {}) or {}
    max_points_per_class = int(plot_config.get("max_points_per_class", 5000))
    alpha = float(plot_config.get("alpha", 0.6))
    point_size = float(plot_config.get("point_size", 8.0))

    if k_neighbors <= 0:
        raise ValueError("cnn_embedding_knn.k_neighbors must be positive")
    if embedding_batch_size <= 0:
        raise ValueError("cnn_embedding_knn.embedding_batch_size must be positive")
    if neighbor_batch_size <= 0:
        raise ValueError("cnn_embedding_knn.neighbor_batch_size must be positive")
    if not 0.0 <= min_point_neighbor_fraction <= 1.0:
        raise ValueError(
            "cnn_embedding_knn.min_point_neighbor_fraction must be between zero and one"
        )
    if max_same_class_fraction is not None and not (
        0.0 <= max_same_class_fraction <= 1.0
    ):
        raise ValueError(
            "cnn_embedding_knn.max_same_class_fraction must be between zero and one"
        )
    if max_points_per_class <= 0:
        raise ValueError(
            "cnn_embedding_knn.plot.max_points_per_class must be positive"
        )
    if not 0.0 < alpha <= 1.0:
        raise ValueError("cnn_embedding_knn.plot.alpha must be between zero and one")
    if point_size <= 0.0:
        raise ValueError("cnn_embedding_knn.plot.point_size must be positive")

    return {
        "enabled": bool(config.get("enabled", False)),
        "k_neighbors": k_neighbors,
        "embedding_batch_size": embedding_batch_size,
        "neighbor_batch_size": neighbor_batch_size,
        "candidate_classes": candidate_classes,
        "point_neighbor_classes": point_neighbor_classes,
        "protected_classes": protected_classes,
        "apply_splits": apply_splits,
        "min_point_neighbor_fraction": min_point_neighbor_fraction,
        "min_point_neighbor_fraction_by_class": (
            min_point_neighbor_fraction_by_class
        ),
        "max_same_class_fraction": max_same_class_fraction,
        "max_same_class_fraction_by_class": max_same_class_fraction_by_class,
        "plot_enabled": bool(plot_config.get("enabled", True)),
        "max_points_per_class": max_points_per_class,
        "plot_random_state": int(plot_config.get("random_state", 1234)),
        "alpha": alpha,
        "point_size": point_size,
    }


def _extract_cnn_embeddings_by_split(model, data, batch_size):
    """
    Extract CNN embeddings for the train, validation, and test splits.

    Parameters
    ----------
    model : PyTorchCNNClassifier
        Trained CNN classifier used to extract embeddings.
    data : dict
        Training data containing VDF features and optional topology arrays.
    batch_size : int
        Number of samples transformed at once.

    Returns
    -------
    dict
        Embedding matrices keyed by train, validation, and test split names.
    """

    return {
        "train": model.transform_embeddings(
            data["X_train_features"],
            batch_size=batch_size,
            **_create_context_kwargs(model, data, "train"),
        ),
        "validation": model.transform_embeddings(
            data["X_validation_features"],
            batch_size=batch_size,
            **_create_context_kwargs(model, data, "validation"),
        ),
        "test": model.transform_embeddings(
            data["X_test_features"],
            batch_size=batch_size,
            **_create_context_kwargs(model, data, "test"),
        ),
    }


def _create_cnn_embedding_metrics(
    data,
    embeddings_by_split,
    class_names_by_label,
):
    rows = []
    split_items = [
        ("train", data["train_indices"], data["y_train"]),
        ("validation", data["validation_indices"], data["y_validation"]),
        ("test", data["test_indices"], data["y_test"]),
    ]
    for split_name, indices, labels in split_items:
        split_rows = data["metadata"].iloc[indices].copy()
        if "sample_index" not in split_rows.columns:
            split_rows.insert(0, "sample_index", np.asarray(indices, dtype=int))
        split_rows["split"] = split_name
        split_rows["label"] = np.asarray(labels, dtype=int)
        if "class_name" not in split_rows.columns:
            split_rows["class_name"] = split_rows["label"].map(class_names_by_label)
        split_rows["embedding_norm"] = np.linalg.norm(
            embeddings_by_split[split_name],
            axis=1,
        )
        rows.append(split_rows)

    return pd.concat(rows, ignore_index=True)


def _add_cnn_embedding_neighbor_metrics(
    metrics,
    embeddings_by_split,
    data,
    class_names_by_label,
    config,
):
    train_embeddings = embeddings_by_split["train"]
    train_labels = np.asarray(data["y_train"], dtype=int)
    metrics["same_class_fraction"] = np.nan
    metrics["point_neighbor_fraction"] = np.nan
    metrics["embedding_borderline_score"] = np.nan
    metrics["nearest_other_class"] = ""
    metrics["nearest_point_class"] = ""

    if len(train_embeddings) < 2:
        return

    point_labels = _labels_for_class_names(
        class_names_by_label=class_names_by_label,
        class_names=config["point_neighbor_classes"],
    )
    k_neighbors = min(config["k_neighbors"], len(train_embeddings) - 1)
    neighbor_model = NearestNeighbors(n_neighbors=k_neighbors + 1)
    neighbor_model.fit(train_embeddings)

    split_offsets = _create_split_offsets(metrics)
    for split_name, embeddings in embeddings_by_split.items():
        start, stop = split_offsets[split_name]
        query_labels = metrics.loc[start:stop - 1, "label"].to_numpy(dtype=int)
        query_train_positions = None
        if split_name == "train":
            query_train_positions = np.arange(len(embeddings), dtype=int)

        (
            same_class_fraction,
            point_neighbor_fraction,
            nearest_other_class,
            nearest_point_class,
        ) = _compute_embedding_knn_metrics(
            neighbor_model=neighbor_model,
            query_embeddings=embeddings,
            query_labels=query_labels,
            train_labels=train_labels,
            point_labels=point_labels,
            class_names_by_label=class_names_by_label,
            k_neighbors=k_neighbors,
            batch_size=config["neighbor_batch_size"],
            query_train_positions=query_train_positions,
        )

        metrics.loc[start:stop - 1, "same_class_fraction"] = same_class_fraction
        metrics.loc[start:stop - 1, "point_neighbor_fraction"] = (
            point_neighbor_fraction
        )
        metrics.loc[start:stop - 1, "embedding_borderline_score"] = (
            point_neighbor_fraction - same_class_fraction
        )
        metrics.loc[start:stop - 1, "nearest_other_class"] = nearest_other_class
        metrics.loc[start:stop - 1, "nearest_point_class"] = nearest_point_class


def _compute_embedding_knn_metrics(
    neighbor_model,
    query_embeddings,
    query_labels,
    train_labels,
    point_labels,
    class_names_by_label,
    k_neighbors,
    batch_size,
    query_train_positions=None,
):
    same_class_fraction = np.empty(len(query_embeddings), dtype=np.float32)
    point_neighbor_fraction = np.empty(len(query_embeddings), dtype=np.float32)
    nearest_other_class = np.empty(len(query_embeddings), dtype=object)
    nearest_point_class = np.empty(len(query_embeddings), dtype=object)
    point_labels = set(int(label) for label in point_labels)

    for batch_start in range(0, len(query_embeddings), batch_size):
        batch_end = min(batch_start + batch_size, len(query_embeddings))
        _, neighbor_indices = neighbor_model.kneighbors(
            query_embeddings[batch_start:batch_end],
            return_distance=True,
        )
        for local_index, raw_neighbors in enumerate(neighbor_indices):
            query_index = batch_start + local_index
            neighbors = raw_neighbors
            if query_train_positions is not None:
                train_position = int(query_train_positions[query_index])
                neighbors = neighbors[neighbors != train_position]
            neighbors = neighbors[:k_neighbors]
            neighbor_labels = train_labels[neighbors]
            query_label = int(query_labels[query_index])
            point_mask = np.isin(neighbor_labels, list(point_labels))

            same_class_fraction[query_index] = np.mean(
                neighbor_labels == query_label
            )
            point_neighbor_fraction[query_index] = np.mean(point_mask)
            nearest_other_class[query_index] = _nearest_class_name(
                labels=neighbor_labels[neighbor_labels != query_label],
                class_names_by_label=class_names_by_label,
            )
            nearest_point_class[query_index] = _nearest_class_name(
                labels=neighbor_labels[point_mask],
                class_names_by_label=class_names_by_label,
            )

    return (
        same_class_fraction,
        point_neighbor_fraction,
        nearest_other_class,
        nearest_point_class,
    )


def _add_cnn_embedding_filter_preview(metrics, config):
    point_thresholds = _thresholds_for_rows(
        rows=metrics,
        default_threshold=config["min_point_neighbor_fraction"],
        class_thresholds=config["min_point_neighbor_fraction_by_class"],
    )
    candidate_mask = (
        metrics["split"].isin(config["apply_splits"])
        & metrics["class_name"].isin(config["candidate_classes"])
        & ~metrics["class_name"].isin(config["protected_classes"])
        & metrics["point_neighbor_fraction"].ge(point_thresholds)
    )

    if _has_embedding_same_class_filter(config):
        same_thresholds = _thresholds_for_rows(
            rows=metrics,
            default_threshold=config["max_same_class_fraction"],
            class_thresholds=config["max_same_class_fraction_by_class"],
        )
        candidate_mask = candidate_mask & metrics["same_class_fraction"].le(
            same_thresholds
        )

    metrics["embedding_filter_candidate"] = candidate_mask.to_numpy(dtype=bool)
    metrics["embedding_filter_reason"] = ""
    if metrics["embedding_filter_candidate"].any():
        metrics.loc[
            metrics["embedding_filter_candidate"],
            "embedding_filter_reason",
        ] = _create_embedding_filter_reasons(
            rows=metrics[metrics["embedding_filter_candidate"]],
            config=config,
        )


def _add_cnn_embedding_plot_coordinates(metrics, embeddings_by_split):
    embeddings = np.concatenate(
        [embeddings_by_split[split_name] for split_name in embeddings_by_split],
        axis=0,
    )
    if embeddings.shape[1] >= 2 and len(embeddings) >= 2:
        coordinates = PCA(n_components=2).fit_transform(embeddings)
    else:
        coordinates = np.zeros((len(embeddings), 2), dtype=np.float32)

    metrics["embedding_plot_0"] = coordinates[:, 0].astype(np.float32, copy=False)
    metrics["embedding_plot_1"] = coordinates[:, 1].astype(np.float32, copy=False)


def _save_cnn_embeddings(embeddings_by_split, data, output_path):
    embeddings = np.concatenate(
        [embeddings_by_split[split_name] for split_name in embeddings_by_split],
        axis=0,
    )
    sample_indices = np.concatenate(
        [
            np.asarray(data[f"{split_name}_indices"], dtype=int)
            for split_name in embeddings_by_split
        ]
    )
    labels = np.concatenate(
        [
            np.asarray(data[f"y_{split_name}"], dtype=int)
            for split_name in embeddings_by_split
        ]
    )
    splits = np.concatenate(
        [
            np.full(len(embeddings_by_split[split_name]), split_name)
            for split_name in embeddings_by_split
        ]
    )
    np.savez(
        output_path,
        embeddings=embeddings.astype(np.float32, copy=False),
        sample_indices=sample_indices,
        labels=labels,
        splits=splits,
    )


def _plot_cnn_embedding_knn_preview(metrics, output_path, config):
    plot_rows = _subsample_embedding_plot_rows(
        metrics=metrics,
        max_points_per_class=config["max_points_per_class"],
        random_state=config["plot_random_state"],
    )
    candidate_rows = metrics[metrics["embedding_filter_candidate"]]
    plot_rows = pd.concat([plot_rows, candidate_rows], ignore_index=True)
    plot_rows = plot_rows.drop_duplicates(subset=["sample_index", "split"])
    if plot_rows.empty:
        return False

    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 7))
    for class_name, class_rows in plot_rows.groupby("class_name", sort=True):
        ax.scatter(
            class_rows["embedding_plot_0"],
            class_rows["embedding_plot_1"],
            s=config["point_size"],
            alpha=config["alpha"],
            label=class_name,
        )

    point_rows = plot_rows[
        plot_rows["class_name"].isin(config["point_neighbor_classes"])
    ]
    if not point_rows.empty:
        ax.scatter(
            point_rows["embedding_plot_0"],
            point_rows["embedding_plot_1"],
            s=config["point_size"] * 4,
            facecolors="none",
            edgecolors="black",
            linewidths=0.8,
            label="point-neighbor classes",
        )

    candidate_rows = plot_rows[plot_rows["embedding_filter_candidate"]]
    if not candidate_rows.empty:
        ax.scatter(
            candidate_rows["embedding_plot_0"],
            candidate_rows["embedding_plot_1"],
            s=config["point_size"] * 7,
            marker="x",
            c="red",
            linewidths=1.0,
            label="embedding preview candidates",
        )

    ax.set_title("CNN embedding kNN preview")
    ax.set_xlabel("Embedding PCA 1")
    ax.set_ylabel("Embedding PCA 2")
    ax.legend(loc="best", fontsize="small")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return True


def _subsample_embedding_plot_rows(metrics, max_points_per_class, random_state):
    sampled_rows = []
    rng = np.random.default_rng(random_state)
    for _, class_rows in metrics.groupby("class_name", sort=True):
        if len(class_rows) <= max_points_per_class:
            sampled_rows.append(class_rows)
            continue
        selected_positions = rng.choice(
            len(class_rows),
            size=max_points_per_class,
            replace=False,
        )
        sampled_rows.append(class_rows.iloc[np.sort(selected_positions)])
    if not sampled_rows:
        return metrics.iloc[0:0].copy()
    return pd.concat(sampled_rows, ignore_index=True)


def _create_cnn_embedding_knn_metric_lines(
    metrics,
    config,
    embeddings_path,
    metrics_path,
    candidates_path,
    summary_path,
    preview_path,
):
    lines = [
        "CNN embedding kNN enabled: True",
        f"CNN embedding kNN neighbors: {config['k_neighbors']}",
        f"CNN embedding candidate classes: {', '.join(config['candidate_classes'])}",
        "CNN embedding point-neighbor classes: "
        f"{', '.join(config['point_neighbor_classes'])}",
        f"CNN embedding protected classes: {', '.join(config['protected_classes'])}",
        f"CNN embedding apply splits: {', '.join(config['apply_splits'])}",
        "CNN embedding candidates: "
        f"{int(metrics['embedding_filter_candidate'].sum())}",
        f"CNN embeddings: {embeddings_path}",
        f"CNN embedding kNN metrics: {metrics_path}",
        f"CNN embedding kNN candidates: {candidates_path}",
        f"CNN embedding kNN summary: {summary_path}",
    ]
    if preview_path is not None:
        lines.append(f"CNN embedding kNN preview: {preview_path}")
    for _, row in _summarize_embedding_candidate_counts(metrics).iterrows():
        lines.append(
            "CNN embedding kNN class "
            f"{row['split']} {row['class_name']}: "
            f"count={int(row['count'])}, "
            f"candidates={int(row['candidates'])}, "
            f"mean_same={row['mean_same_class_fraction']:.6g}, "
            f"mean_point={row['mean_point_neighbor_fraction']:.6g}"
        )
    return lines


def _create_cnn_embedding_knn_summary_text(
    metrics,
    config,
    embeddings_path,
    metrics_path,
    candidates_path,
    preview_path,
):
    summary_counts = _summarize_embedding_candidate_counts(metrics)
    lines = [
        "CNN embedding kNN metrics",
        "=" * 70,
        "Preview only: these candidates are not removed by this stage.",
        f"Neighbors: {config['k_neighbors']}",
        f"Candidate classes: {', '.join(config['candidate_classes'])}",
        f"Point-neighbor classes: {', '.join(config['point_neighbor_classes'])}",
        f"Protected classes: {', '.join(config['protected_classes'])}",
        f"Apply splits: {', '.join(config['apply_splits'])}",
        f"Minimum point-neighbor fraction: {config['min_point_neighbor_fraction']}",
        "Class-specific point thresholds: "
        f"{config['min_point_neighbor_fraction_by_class']}",
        f"Maximum same-class fraction: {config['max_same_class_fraction']}",
        "Class-specific same-class thresholds: "
        f"{config['max_same_class_fraction_by_class']}",
        f"Embeddings: {embeddings_path}",
        f"Metrics: {metrics_path}",
        f"Candidates: {candidates_path}",
        f"Preview plot: {preview_path}",
        "",
        "Counts by split and class",
        "=" * 70,
        summary_counts.to_string(index=False),
        "",
    ]
    return "\n".join(lines)


def _summarize_embedding_candidate_counts(metrics):
    return (
        metrics.groupby(["split", "class_name"], sort=True)
        .agg(
            count=("sample_index", "size"),
            candidates=("embedding_filter_candidate", "sum"),
            mean_same_class_fraction=("same_class_fraction", "mean"),
            mean_point_neighbor_fraction=("point_neighbor_fraction", "mean"),
        )
        .reset_index()
    )


def _labels_for_class_names(class_names_by_label, class_names):
    name_to_label = {
        str(class_name): int(label)
        for label, class_name in class_names_by_label.items()
    }
    return [
        name_to_label[class_name]
        for class_name in class_names
        if class_name in name_to_label
    ]


def _nearest_class_name(labels, class_names_by_label):
    if len(labels) == 0:
        return ""
    return class_names_by_label.get(int(labels[0]), "")


def _create_split_offsets(rows):
    offsets = {}
    start = 0
    for split_name, split_rows in rows.groupby("split", sort=False):
        stop = start + len(split_rows)
        offsets[str(split_name)] = (start, stop)
        start = stop
    return offsets


def _thresholds_for_rows(rows, default_threshold, class_thresholds):
    if default_threshold is None:
        thresholds = np.ones(len(rows), dtype=np.float32)
    else:
        thresholds = np.full(len(rows), float(default_threshold), dtype=np.float32)

    for class_name, threshold in class_thresholds.items():
        thresholds[rows["class_name"].to_numpy() == class_name] = float(threshold)
    return thresholds


def _has_embedding_same_class_filter(config):
    return (
        config["max_same_class_fraction"] is not None
        or bool(config["max_same_class_fraction_by_class"])
    )


def _create_embedding_filter_reasons(rows, config):
    reasons = []
    for _, row in rows.iterrows():
        class_name = row["class_name"]
        point_threshold = config["min_point_neighbor_fraction_by_class"].get(
            class_name,
            config["min_point_neighbor_fraction"],
        )
        parts = [f"point_neighbor_fraction>={point_threshold:g}"]
        if _has_embedding_same_class_filter(config):
            same_threshold = config["max_same_class_fraction_by_class"].get(
                class_name,
                config["max_same_class_fraction"],
            )
            if same_threshold is not None:
                parts.append(f"same_class_fraction<={same_threshold:g}")
        reasons.append(";".join(parts))
    return reasons


def _resolve_class_thresholds(thresholds):
    if thresholds is None:
        return {}
    if not isinstance(thresholds, dict):
        raise ValueError("class-specific thresholds must be a mapping")
    parsed_thresholds = {
        str(class_name): float(threshold)
        for class_name, threshold in thresholds.items()
    }
    if any(threshold < 0.0 or threshold > 1.0 for threshold in parsed_thresholds.values()):
        raise ValueError("class-specific threshold values must be between zero and one")
    return parsed_thresholds


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
        return

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
            filter_config.get("protected_classes", ["x_point", "o_point"]),
        ),
    )
    pca_filter_preview_config.setdefault(
        "protected_classes",
        filter_config.get("protected_classes", ["x_point", "o_point"]),
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
        "spatial_features": pca_filter_config.get(
            "spatial_features",
            {},
        ),
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
    if _is_cnn_embedding_filter_source(source):
        return _apply_cnn_embedding_knn_training_filter(
            data=data,
            config=config,
        )
    if source != "pca":
        raise ValueError(
            "training_filter.source must be 'pca' or 'cnn_embedding_knn'"
        )

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


def _is_cnn_embedding_filter_source(source):
    return str(source).lower() in {
        "cnn_embedding_knn",
        "cnn_embedding",
        "embedding",
    }


def _apply_cnn_embedding_knn_training_filter(data, config):
    """
    Apply a CNN embedding kNN candidate file as a training filter.

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

    embedding_metrics_path = data["output_dir"] / CNN_EMBEDDING_KNN_METRICS_FILENAME
    if not embedding_metrics_path.exists():
        raise FileNotFoundError(
            "CNN embedding kNN training filter requires metrics at "
            f"{embedding_metrics_path}. Set "
            "training_filter.cnn_embedding_knn.run_source_model: true to "
            "create it in the same run, or create it in an earlier run."
        )

    embedding_metrics = pd.read_csv(embedding_metrics_path)
    filter_metrics = _create_cnn_embedding_training_filter_metrics(
        embedding_metrics=embedding_metrics,
        data=data,
        candidate_classes=candidate_classes,
        protected_classes=protected_classes,
        max_removed_fraction=max_removed_fraction,
        dry_run=dry_run,
    )

    selected_sample_indices = filter_metrics.loc[
        filter_metrics["cnn_embedding_filter_selected"],
        "sample_index",
    ].to_numpy(dtype=int)

    if not dry_run and len(selected_sample_indices) > 0:
        _remove_training_samples(data, selected_sample_indices)

    output_dir = data["output_dir"]
    metrics_path = output_dir / CNN_EMBEDDING_FILTER_METRICS_FILENAME
    removed_path = output_dir / CNN_EMBEDDING_FILTER_REMOVED_FILENAME
    summary_path = output_dir / CNN_EMBEDDING_FILTER_SUMMARY_FILENAME

    filter_metrics.to_csv(metrics_path, index=False)
    filter_metrics[filter_metrics["cnn_embedding_filter_selected"]].to_csv(
        removed_path,
        index=False,
    )

    summary_text = _create_cnn_embedding_filter_summary_text(
        embedding_metrics_path=embedding_metrics_path,
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

    selected_count = int(filter_metrics["cnn_embedding_filter_selected"].sum())
    removed_count = int(filter_metrics["cnn_embedding_filter_removed"].sum())
    print("CNN embedding kNN training filter")
    print(f"  Metrics: {metrics_path}")
    print(f"  Selected samples: {selected_count}")
    print(f"  Removed samples: {removed_count}")
    if dry_run:
        print("  Dry run: training samples were not removed")

    return {
        "metric_lines": _create_cnn_embedding_filter_metric_lines(
            embedding_metrics_path=embedding_metrics_path,
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


def _create_cnn_embedding_training_filter_metrics(
    embedding_metrics,
    data,
    candidate_classes,
    protected_classes,
    max_removed_fraction,
    dry_run,
):
    """
    Create per-training-sample CNN embedding filter decisions.

    Parameters
    ----------
    embedding_metrics : pandas.DataFrame
        CNN embedding per-sample metrics.
    data : dict
        Training data returned by ``load_training_data``.
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
        Embedding metrics with CNN training-filter decision columns.
    """

    required_columns = {
        "sample_index",
        "split",
        "class_name",
        "embedding_filter_candidate",
    }
    missing_columns = sorted(required_columns - set(embedding_metrics.columns))
    if missing_columns:
        raise ValueError(
            "CNN embedding kNN metrics are missing required columns: "
            f"{missing_columns}"
        )
    if embedding_metrics["sample_index"].duplicated().any():
        raise ValueError(
            "CNN embedding kNN metrics contain duplicate sample_index values"
        )

    train_indices = np.asarray(data["train_indices"], dtype=int)
    train_order = pd.DataFrame(
        {
            "sample_index": train_indices,
            "cnn_train_order": np.arange(len(train_indices), dtype=int),
            "fallback_class_name": data["metadata"]
            .iloc[train_indices]["class_name"]
            .to_numpy(),
        }
    )
    filter_metrics = train_order.merge(
        embedding_metrics,
        on="sample_index",
        how="left",
        validate="one_to_one",
    )
    missing_rows = filter_metrics["split"].isna()
    non_train_rows = (~missing_rows) & (filter_metrics["split"] != "train")
    if non_train_rows.any():
        bad_sample_indices = filter_metrics.loc[
            non_train_rows,
            "sample_index",
        ].head(10).tolist()
        raise ValueError(
            "CNN embedding kNN metrics do not use the same train split as "
            "the CNN run. First mismatched sample indices: "
            f"{bad_sample_indices}"
        )

    filter_metrics["cnn_embedding_filter_source_missing"] = (
        missing_rows.to_numpy(dtype=bool)
    )
    filter_metrics.loc[missing_rows, "split"] = "train"
    filter_metrics["class_name"] = filter_metrics["class_name"].fillna(
        filter_metrics["fallback_class_name"]
    )
    filter_metrics["embedding_filter_candidate"] = filter_metrics[
        "embedding_filter_candidate"
    ].fillna(False)

    embedding_candidate = _coerce_boolean_column(
        filter_metrics["embedding_filter_candidate"],
        "embedding_filter_candidate",
    )
    candidate_class_mask = filter_metrics["class_name"].isin(
        candidate_classes
    ).to_numpy(dtype=bool)
    protected_class_mask = filter_metrics["class_name"].isin(
        protected_classes
    ).to_numpy(dtype=bool)
    eligible_mask = (
        embedding_candidate
        & candidate_class_mask
        & ~protected_class_mask
        & ~filter_metrics["cnn_embedding_filter_source_missing"].to_numpy(
            dtype=bool
        )
    )

    filter_metrics["cnn_embedding_filter_candidate"] = embedding_candidate
    filter_metrics["cnn_embedding_filter_eligible"] = eligible_mask
    filter_metrics["cnn_embedding_filter_selected"] = False
    selected_indices = _select_pca_filter_candidates(
        filter_metrics=filter_metrics,
        eligible_mask=eligible_mask,
        candidate_classes=candidate_classes,
        max_removed_fraction=max_removed_fraction,
    )
    if selected_indices:
        filter_metrics.loc[
            selected_indices,
            "cnn_embedding_filter_selected",
        ] = True

    filter_metrics["cnn_embedding_filter_removed"] = (
        filter_metrics["cnn_embedding_filter_selected"] & (not dry_run)
    )
    filter_metrics["cnn_embedding_filter_dry_run"] = dry_run
    filter_metrics["cnn_embedding_filter_reason"] = (
        _create_cnn_embedding_training_filter_reasons(
            filter_metrics=filter_metrics,
            embedding_candidate=embedding_candidate,
            candidate_class_mask=candidate_class_mask,
            protected_class_mask=protected_class_mask,
        )
    )
    filter_metrics = filter_metrics.drop(columns=["fallback_class_name"])
    return filter_metrics


def _create_cnn_embedding_training_filter_reasons(
    filter_metrics,
    embedding_candidate,
    candidate_class_mask,
    protected_class_mask,
):
    reasons = np.full(len(filter_metrics), "", dtype=object)
    selected = filter_metrics["cnn_embedding_filter_selected"].to_numpy(dtype=bool)
    eligible = filter_metrics["cnn_embedding_filter_eligible"].to_numpy(dtype=bool)
    source_missing = filter_metrics[
        "cnn_embedding_filter_source_missing"
    ].to_numpy(dtype=bool)

    reasons[source_missing] = "missing_embedding_metrics"
    reasons[embedding_candidate & ~candidate_class_mask] = "not_candidate_class"
    reasons[embedding_candidate & protected_class_mask] = "protected_class"
    reasons[eligible & ~selected] = "class_fraction_cap"
    reasons[selected] = "selected_by_cnn_embedding_filter"
    return reasons


def _create_cnn_embedding_filter_metric_lines(
    embedding_metrics_path,
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
    summary_counts = _summarize_cnn_embedding_filter_counts(filter_metrics)
    lines = [
        "Training filter enabled: True",
        "Training filter source: cnn_embedding_knn",
        f"Training filter dry run: {dry_run}",
        f"CNN embedding filter source metrics: {embedding_metrics_path}",
        "CNN embedding filter candidate column: embedding_filter_candidate",
        f"CNN embedding filter candidate classes: {', '.join(candidate_classes)}",
        f"CNN embedding filter protected classes: {', '.join(protected_classes)}",
        "CNN embedding filter max removed fraction per class: "
        f"{max_removed_fraction}",
        f"CNN embedding filter train samples before filter: {len(filter_metrics)}",
        "CNN embedding filter selected samples: "
        f"{int(filter_metrics['cnn_embedding_filter_selected'].sum())}",
        "CNN embedding filter removed samples: "
        f"{int(filter_metrics['cnn_embedding_filter_removed'].sum())}",
        "CNN embedding filter missing source metrics: "
        f"{int(filter_metrics['cnn_embedding_filter_source_missing'].sum())}",
        f"CNN embedding filter train samples after filter: {train_samples_after_filter}",
        f"CNN embedding filter metrics: {metrics_path}",
        f"CNN embedding filter selected-sample file: {removed_path}",
        f"CNN embedding filter summary: {summary_path}",
    ]
    for _, row in summary_counts.iterrows():
        lines.append(
            "CNN embedding filter class "
            f"{row['class_name']}: train={int(row['train_samples'])}, "
            f"candidate={int(row['embedding_candidates'])}, "
            f"eligible={int(row['eligible_candidates'])}, "
            f"selected={int(row['selected_samples'])}, "
            f"removed={int(row['removed_samples'])}, "
            f"missing_source={int(row['missing_source_metrics'])}"
        )
    return lines


def _create_cnn_embedding_filter_summary_text(
    embedding_metrics_path,
    metrics_path,
    removed_path,
    dry_run,
    candidate_classes,
    protected_classes,
    max_removed_fraction,
    filter_metrics,
    train_samples_after_filter,
):
    summary_counts = _summarize_cnn_embedding_filter_counts(filter_metrics)
    lines = [
        "CNN embedding kNN training filter summary",
        "=" * 70,
        f"Source CNN embedding metrics: {embedding_metrics_path}",
        f"Output metrics: {metrics_path}",
        f"Selected-sample file: {removed_path}",
        "Candidate column: embedding_filter_candidate",
        f"Dry run: {dry_run}",
        f"Candidate classes: {', '.join(candidate_classes)}",
        f"Protected classes: {', '.join(protected_classes)}",
        f"Max removed fraction per class: {max_removed_fraction}",
        f"Train samples before filter: {len(filter_metrics)}",
        "Selected samples: "
        f"{int(filter_metrics['cnn_embedding_filter_selected'].sum())}",
        "Removed samples: "
        f"{int(filter_metrics['cnn_embedding_filter_removed'].sum())}",
        "Missing source metrics: "
        f"{int(filter_metrics['cnn_embedding_filter_source_missing'].sum())}",
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


def _summarize_cnn_embedding_filter_counts(filter_metrics):
    return (
        filter_metrics.groupby("class_name", sort=True)
        .agg(
            train_samples=("sample_index", "size"),
            embedding_candidates=("cnn_embedding_filter_candidate", "sum"),
            eligible_candidates=("cnn_embedding_filter_eligible", "sum"),
            selected_samples=("cnn_embedding_filter_selected", "sum"),
            removed_samples=("cnn_embedding_filter_removed", "sum"),
            missing_source_metrics=(
                "cnn_embedding_filter_source_missing",
                "sum",
            ),
        )
        .reset_index()
    )


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
    """
    Remove selected rows from all in-memory training arrays.

    Parameters
    ----------
    data : dict
        Training data updated in place, including optional topology arrays.
    selected_sample_indices : array-like of int
        Original metadata row indices to remove from the training split.
    """

    selected_sample_indices = np.asarray(selected_sample_indices, dtype=int)
    keep_mask = ~np.isin(
        np.asarray(data["train_indices"], dtype=int),
        selected_sample_indices,
    )
    data["X_train_features"] = data["X_train_features"][keep_mask]
    for branch_name in ["distance", "vector"]:
        key = f"X_train_{branch_name}_features"
        if data.get(key) is not None:
            data[key] = data[key][keep_mask]
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
    if class_weight in {"manual", "custom"}:
        return "manual"

    raise ValueError(
        "class_weight must be 'none', 'sqrt_balanced', 'balanced', or 'manual'"
    )


def _create_class_weights(
    targets,
    n_classes,
    class_weight,
    class_names,
    configured_class_weights,
):
    if class_weight == "none":
        return None

    class_counts = np.bincount(targets, minlength=n_classes).astype(float)
    if np.any(class_counts == 0.0):
        raise ValueError("Cannot create class weights for empty classes")

    if class_weight == "manual":
        weights = _create_manual_class_weight_array(
            class_names=class_names,
            configured_class_weights=configured_class_weights,
            config_name="model.class_weights",
        )
    else:
        weights = len(targets) / (n_classes * class_counts)
        if class_weight == "sqrt_balanced":
            weights = np.sqrt(weights)

    return weights.astype(np.float32)


def _create_training_sampler(targets, class_names, sampler_config, random_seed):
    sampler_config = sampler_config or {}
    enabled = bool(sampler_config.get("enabled", False))
    mode = str(sampler_config.get("mode", "none")).lower()
    if not enabled or mode in {"none", "false", "off", "disabled"}:
        return None, {
            "enabled": False,
            "mode": "none",
            "class_weights": None,
        }

    if mode not in {"balanced", "sqrt_balanced", "manual"}:
        raise ValueError(
            "model.sampler.mode must be 'balanced', 'sqrt_balanced', or 'manual'"
        )

    class_counts = np.bincount(targets, minlength=len(class_names)).astype(float)
    if np.any(class_counts == 0.0):
        raise ValueError("Cannot create sampler weights for empty classes")

    if mode == "manual":
        class_weights = _create_manual_class_weight_array(
            class_names=class_names,
            configured_class_weights=sampler_config.get("class_weights", {}),
            config_name="model.sampler.class_weights",
        )
    else:
        class_weights = len(targets) / (len(class_names) * class_counts)
        if mode == "sqrt_balanced":
            class_weights = np.sqrt(class_weights)

    sample_weights = class_weights[np.asarray(targets, dtype=int)]
    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(targets),
        replacement=bool(sampler_config.get("replacement", True)),
        generator=torch.Generator().manual_seed(random_seed),
    )
    return sampler, {
        "enabled": True,
        "mode": mode,
        "class_weights": class_weights.astype(np.float32),
    }


def _create_manual_class_weight_array(
    class_names,
    configured_class_weights,
    config_name,
):
    if not configured_class_weights:
        raise ValueError(f"{config_name} must be set when using manual weights")

    configured_class_weights = {
        str(class_name): float(weight)
        for class_name, weight in configured_class_weights.items()
    }
    missing_classes = [
        class_name
        for class_name in class_names
        if class_name not in configured_class_weights
    ]
    if missing_classes:
        raise ValueError(
            f"{config_name} missing weights for classes: {missing_classes}"
        )

    weights = np.asarray(
        [configured_class_weights[class_name] for class_name in class_names],
        dtype=float,
    )
    if np.any(weights <= 0.0):
        raise ValueError(f"{config_name} values must be positive")

    return weights


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
