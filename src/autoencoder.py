from copy import deepcopy
from pathlib import Path
import os
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

os.environ["PTNOLATEX"] = "1"

from src.autoencoder_data import (
    create_log_slice,
    extract_plot_xz_slice_from_dataset,
    load_autoencoder_training_data,
    normalize_log_slice,
    normalized_log_slice_to_physical,
)
from src.autoencoder_plot import (
    create_reconstruction_filename,
    plot_latent_pca,
    plot_reconstruction_pair,
    resolve_vdf_plot_parameters,
)


LOG_10 = float(np.log(10.0))


class VdfSliceAutoencoder2D(nn.Module):
    """
    Deterministic convolutional autoencoder for 2D VDF xz slices.

    Parameters
    ----------
    input_shape : tuple of int
        Input tensor shape ``(channels, height, width)``.
    channels : sequence of int
        Output channels for each encoder convolution block.
    latent_dim : int
        Size of the latent vector.
    encoder_pooling : {"average", "stride"}, optional
        Downsampling method used in the encoder.
    encoder_adaptive_pool_shape : tuple of int or None, optional
        Optional adaptive average pooling output shape after convolution blocks.
    """

    def __init__(
        self,
        input_shape,
        channels=(16, 32, 64),
        latent_dim=64,
        encoder_pooling="average",
        encoder_adaptive_pool_shape=(4, 4),
    ):
        super().__init__()

        self.input_shape = tuple(int(value) for value in input_shape)
        self.channels = tuple(int(channel) for channel in channels)
        self.latent_dim = int(latent_dim)
        self.encoder_pooling = str(encoder_pooling)
        self.encoder_adaptive_pool_shape = resolve_optional_pool_shape(
            encoder_adaptive_pool_shape
        )

        if len(self.input_shape) != 3:
            raise ValueError("input_shape must be (channels, height, width)")
        if self.input_shape[0] != 1:
            raise ValueError("VdfSliceAutoencoder2D expects one input channel")
        if not self.channels or any(channel <= 0 for channel in self.channels):
            raise ValueError("channels must contain positive integers")
        if self.latent_dim <= 0:
            raise ValueError("latent_dim must be positive")
        if self.encoder_pooling not in {"average", "stride"}:
            raise ValueError("encoder_pooling must be 'average' or 'stride'")

        encoder_layers = []
        encoder_sizes = [self.input_shape[1:]]
        input_channels = self.input_shape[0]
        height, width = self.input_shape[1:]

        for output_channels in self.channels:
            convolution_stride = 2 if self.encoder_pooling == "stride" else 1
            encoder_layers.extend(
                [
                    nn.Conv2d(
                        input_channels,
                        output_channels,
                        kernel_size=3,
                        stride=convolution_stride,
                        padding=1,
                    ),
                    nn.ReLU(),
                ]
            )
            if self.encoder_pooling == "average":
                encoder_layers.append(nn.AvgPool2d(kernel_size=2))
                height = _pool_stride2_output_size(height)
                width = _pool_stride2_output_size(width)
            else:
                height = _conv_stride2_output_size(height)
                width = _conv_stride2_output_size(width)
            encoder_sizes.append((height, width))
            input_channels = output_channels

        if self.encoder_adaptive_pool_shape is not None:
            encoder_layers.append(
                nn.AdaptiveAvgPool2d(self.encoder_adaptive_pool_shape)
            )
            height, width = self.encoder_adaptive_pool_shape

        self.encoder = nn.Sequential(*encoder_layers)
        self.encoder_shape = (self.channels[-1], height, width)
        self.encoder_sizes = tuple(encoder_sizes)
        self.flattened_encoder_size = int(np.prod(self.encoder_shape))
        self.to_latent = nn.Linear(
            self.flattened_encoder_size,
            self.latent_dim,
        )
        self.from_latent = nn.Linear(
            self.latent_dim,
            self.flattened_encoder_size,
        )

        decoder_layers = []
        decoder_input_channels = self.channels[-1]
        decoder_targets, decoder_channels = create_decoder_plan(
            input_channels=self.input_shape[0],
            channels=self.channels,
            encoder_sizes=self.encoder_sizes,
            has_adaptive_pool=self.encoder_adaptive_pool_shape is not None,
        )

        for layer_index, (target_size, output_channels) in enumerate(
            zip(decoder_targets, decoder_channels)
        ):

            decoder_layers.append(
                nn.Upsample(
                    size=target_size,
                    mode="bilinear",
                    align_corners=False,
                )
            )
            decoder_layers.append(
                nn.Conv2d(
                    decoder_input_channels,
                    output_channels,
                    kernel_size=3,
                    padding=1,
                )
            )
            if layer_index < len(decoder_channels) - 1:
                decoder_layers.append(nn.ReLU())

            decoder_input_channels = output_channels

        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, inputs):
        """
        Encode normalized log VDF slices.

        Parameters
        ----------
        inputs : torch.Tensor
            Input tensor with shape ``(batch, 1, height, width)``.

        Returns
        -------
        torch.Tensor
            Latent vectors.
        """

        features = self.encoder(inputs)
        features = features.reshape(features.shape[0], -1)

        return self.to_latent(features)

    def decode(self, latents):
        """
        Decode latent vectors into normalized log VDF slices.

        Parameters
        ----------
        latents : torch.Tensor
            Latent vectors.

        Returns
        -------
        torch.Tensor
            Reconstructed normalized log VDF slices.
        """

        features = self.from_latent(latents)
        features = features.reshape(-1, *self.encoder_shape)

        return self.decoder(features)

    def forward(self, inputs):
        """Return reconstructed normalized log VDF slices."""

        return self.decode(self.encode(inputs))


class VdfAutoencoderLoss(nn.Module):
    """
    Reconstruction loss with optional high-signal and physical terms.

    Parameters
    ----------
    mean : float
        Training-set mean used to normalize log-scaled VDF slices.
    std : float
        Training-set standard deviation used to normalize log-scaled VDF slices.
    high_signal_loss_weight : float, optional
        Weight for the high-signal reconstruction term.
    high_signal_quantile : float, optional
        Target-slice quantile used to define high-signal pixels.
    mass_loss_weight : float, optional
        Weight for the log10 slice-mass loss.
    peak_loss_weight : float, optional
        Weight for the log10 peak-value loss.
    """

    def __init__(
        self,
        mean,
        std,
        high_signal_loss_weight=0.0,
        high_signal_quantile=0.95,
        mass_loss_weight=0.0,
        peak_loss_weight=0.0,
    ):
        super().__init__()

        self.mean = float(mean)
        self.std = float(std)
        self.high_signal_loss_weight = float(high_signal_loss_weight)
        self.high_signal_quantile = float(high_signal_quantile)
        self.mass_loss_weight = float(mass_loss_weight)
        self.peak_loss_weight = float(peak_loss_weight)

        if self.std <= 0.0:
            raise ValueError("Loss normalization std must be positive")
        if self.high_signal_loss_weight < 0.0:
            raise ValueError("loss.high_signal_loss_weight must be non-negative")
        if not 0.0 <= self.high_signal_quantile < 1.0:
            raise ValueError("loss.high_signal_quantile must be in [0, 1)")
        if self.mass_loss_weight < 0.0:
            raise ValueError("loss.mass_loss_weight must be non-negative")
        if self.peak_loss_weight < 0.0:
            raise ValueError("loss.peak_loss_weight must be non-negative")

    def forward(self, reconstruction, target):
        """Return weighted autoencoder loss."""

        reconstruction_loss = torch.mean(torch.square(reconstruction - target))
        total_loss = reconstruction_loss

        if self.high_signal_loss_weight > 0.0:
            total_loss = (
                total_loss
                + self.high_signal_loss_weight
                * high_signal_reconstruction_loss(
                    reconstruction=reconstruction,
                    target=target,
                    quantile=self.high_signal_quantile,
                )
            )
        if self.mass_loss_weight > 0.0:
            total_loss = total_loss + self.mass_loss_weight * torch.mean(
                torch.square(
                    normalized_log_slice_mass(reconstruction, self.mean, self.std)
                    - normalized_log_slice_mass(target, self.mean, self.std)
                )
            )
        if self.peak_loss_weight > 0.0:
            total_loss = total_loss + self.peak_loss_weight * torch.mean(
                torch.square(
                    normalized_log_slice_peak(reconstruction, self.mean, self.std)
                    - normalized_log_slice_peak(target, self.mean, self.std)
                )
            )

        return total_loss


def maybe_wrap_data_parallel(model, device, multi_gpu=True, device_ids="all"):
    """
    Wrap a model with ``torch.nn.DataParallel`` when multiple GPUs are visible.

    Parameters
    ----------
    model : torch.nn.Module
        Model to wrap.
    device : torch.device
        Primary training device.
    multi_gpu : bool, optional
        Whether to use more than one CUDA device when available.
    device_ids : {"all"} or sequence of int, optional
        CUDA device IDs visible to PyTorch.

    Returns
    -------
    torch.nn.Module
        Original model or DataParallel-wrapped model.
    """

    if device.type != "cuda" or not bool(multi_gpu):
        return model

    available_gpus = torch.cuda.device_count()
    resolved_device_ids = resolve_data_parallel_device_ids(
        device_ids=device_ids,
        available_gpus=available_gpus,
    )
    if len(resolved_device_ids) <= 1:
        return model

    return nn.DataParallel(model, device_ids=resolved_device_ids)


def resolve_data_parallel_device_ids(device_ids, available_gpus):
    """
    Resolve configured CUDA IDs for DataParallel.

    Parameters
    ----------
    device_ids : {"all"} or sequence of int
        Configured device IDs.
    available_gpus : int
        Number of CUDA devices visible to PyTorch.

    Returns
    -------
    list of int
        Valid CUDA device IDs.
    """

    available_gpus = int(available_gpus)
    if available_gpus <= 0:
        return []

    if device_ids in (None, "all", "auto"):
        resolved_device_ids = list(range(available_gpus))
    elif isinstance(device_ids, str):
        resolved_device_ids = [
            int(value.strip())
            for value in device_ids.split(",")
            if value.strip()
        ]
    else:
        resolved_device_ids = [int(value) for value in device_ids]

    invalid_device_ids = [
        device_id
        for device_id in resolved_device_ids
        if device_id < 0 or device_id >= available_gpus
    ]
    if invalid_device_ids:
        raise ValueError(
            "Configured CUDA device IDs are not visible to PyTorch: "
            f"{invalid_device_ids}. Visible device count: {available_gpus}"
        )

    return resolved_device_ids


def get_base_model(model):
    """
    Return the underlying autoencoder model.

    Parameters
    ----------
    model : torch.nn.Module
        Autoencoder or DataParallel-wrapped autoencoder.

    Returns
    -------
    torch.nn.Module
        Unwrapped autoencoder model.
    """

    if isinstance(model, nn.DataParallel):
        return model.module

    return model


def get_model_device_ids(model):
    """
    Return CUDA device IDs used by a model.

    Parameters
    ----------
    model : torch.nn.Module
        Autoencoder or DataParallel-wrapped autoencoder.

    Returns
    -------
    list of int
        CUDA device IDs, or an empty list for non-parallel models.
    """

    if isinstance(model, nn.DataParallel):
        return [int(device_id) for device_id in model.device_ids]

    return []


def encode_model(model, inputs):
    """
    Encode inputs with an autoencoder, handling DataParallel wrapping.

    Parameters
    ----------
    model : torch.nn.Module
        Autoencoder or DataParallel-wrapped autoencoder.
    inputs : torch.Tensor
        Input tensor.

    Returns
    -------
    torch.Tensor
        Latent vectors.
    """

    return get_base_model(model).encode(inputs)


def resolve_optional_pool_shape(pool_shape):
    """
    Resolve an optional two-dimensional pooling shape.

    Parameters
    ----------
    pool_shape : sequence of int or None
        Pooling shape, or ``None`` to disable adaptive pooling.

    Returns
    -------
    tuple of int or None
        Validated pooling shape.
    """

    if pool_shape is None:
        return None

    if isinstance(pool_shape, str):
        if pool_shape.lower() in {"none", "false", "off"}:
            return None
        values = [int(value.strip()) for value in pool_shape.split(",")]
    else:
        values = [int(value) for value in pool_shape]

    if len(values) != 2 or any(value <= 0 for value in values):
        raise ValueError("encoder_adaptive_pool_shape must contain two positives")

    return tuple(values)


def create_decoder_plan(input_channels, channels, encoder_sizes, has_adaptive_pool):
    """
    Create decoder target sizes and channels.

    Parameters
    ----------
    input_channels : int
        Number of model input channels.
    channels : sequence of int
        Encoder channel sizes.
    encoder_sizes : sequence of tuple of int
        Spatial sizes before and after each encoder downsampling block.
    has_adaptive_pool : bool
        Whether the encoder ends with adaptive pooling.

    Returns
    -------
    decoder_targets : list of tuple of int
        Spatial sizes used by decoder upsampling blocks.
    decoder_channels : list of int
        Output channels for each decoder convolution.
    """

    if has_adaptive_pool:
        decoder_targets = list(reversed(encoder_sizes))
        decoder_channels = [channels[-1], *reversed(channels[:-1]), input_channels]
    else:
        decoder_targets = list(reversed(encoder_sizes[:-1]))
        decoder_channels = [*reversed(channels[:-1]), input_channels]

    return decoder_targets, decoder_channels


def normalized_log_slice_mass(normalized_slice, mean, std):
    """
    Return log10 slice mass from normalized log-scaled VDF slices.

    Parameters
    ----------
    normalized_slice : torch.Tensor
        Normalized log-scaled VDF slices with shape ``(batch, 1, height, width)``.
    mean : float
        Training-set mean used for normalization.
    std : float
        Training-set standard deviation used for normalization.

    Returns
    -------
    torch.Tensor
        Approximate ``log10(sum(f))`` for each slice.
    """

    log_values = normalized_slice * float(std) + float(mean)
    return torch.logsumexp(log_values * LOG_10, dim=(1, 2, 3)) / LOG_10


def normalized_log_slice_peak(normalized_slice, mean, std):
    """
    Return log10 peak value from normalized log-scaled VDF slices.

    Parameters
    ----------
    normalized_slice : torch.Tensor
        Normalized log-scaled VDF slices with shape ``(batch, 1, height, width)``.
    mean : float
        Training-set mean used for normalization.
    std : float
        Training-set standard deviation used for normalization.

    Returns
    -------
    torch.Tensor
        Approximate ``log10(max(f))`` for each slice.
    """

    log_values = normalized_slice * float(std) + float(mean)
    return torch.amax(log_values, dim=(1, 2, 3))


def high_signal_reconstruction_loss(reconstruction, target, quantile):
    """
    Return MSE over the highest-signal pixels in each target slice.

    Parameters
    ----------
    reconstruction : torch.Tensor
        Reconstructed normalized log VDF slices.
    target : torch.Tensor
        Target normalized log VDF slices.
    quantile : float
        Target-slice quantile used to select high-signal pixels.

    Returns
    -------
    torch.Tensor
        Mean squared reconstruction error over high-signal pixels.
    """

    return torch.mean(
        per_sample_high_signal_reconstruction_loss(
            reconstruction=reconstruction,
            target=target,
            quantile=quantile,
        )
    )


def per_sample_high_signal_reconstruction_loss(reconstruction, target, quantile):
    """
    Return per-sample MSE over high-signal target pixels.

    Parameters
    ----------
    reconstruction : torch.Tensor
        Reconstructed normalized log VDF slices.
    target : torch.Tensor
        Target normalized log VDF slices.
    quantile : float
        Target-slice quantile used to select high-signal pixels.

    Returns
    -------
    torch.Tensor
        Per-sample high-signal reconstruction losses.
    """

    flattened_target = target.reshape(target.shape[0], -1)
    flattened_error = torch.square(
        reconstruction.reshape(reconstruction.shape[0], -1) - flattened_target
    )
    n_pixels = flattened_target.shape[1]
    n_high_signal = max(1, int(np.ceil((1.0 - float(quantile)) * n_pixels)))
    high_signal_indices = torch.topk(
        flattened_target,
        k=n_high_signal,
        dim=1,
        largest=True,
        sorted=False,
    ).indices

    return torch.mean(
        torch.gather(flattened_error, dim=1, index=high_signal_indices),
        dim=1,
    )


def resolve_autoencoder_loss_config(config):
    """
    Resolve autoencoder loss settings.

    Parameters
    ----------
    config : dict
        Loss configuration.

    Returns
    -------
    dict
        Validated loss settings.
    """

    if config is None:
        config = {}

    loss_config = {
        "high_signal_loss_weight": float(
            config.get("high_signal_loss_weight", 0.0)
        ),
        "high_signal_quantile": float(config.get("high_signal_quantile", 0.95)),
        "mass_loss_weight": float(config.get("mass_loss_weight", 0.0)),
        "peak_loss_weight": float(config.get("peak_loss_weight", 0.0)),
    }

    if loss_config["high_signal_loss_weight"] < 0.0:
        raise ValueError("loss.high_signal_loss_weight must be non-negative")
    if not 0.0 <= loss_config["high_signal_quantile"] < 1.0:
        raise ValueError("loss.high_signal_quantile must be in [0, 1)")
    if loss_config["mass_loss_weight"] < 0.0:
        raise ValueError("loss.mass_loss_weight must be non-negative")
    if loss_config["peak_loss_weight"] < 0.0:
        raise ValueError("loss.peak_loss_weight must be non-negative")

    return loss_config


def train_autoencoder(config, dataset_id, model_id):
    """
    Train and save a deterministic VDF slice autoencoder.

    Parameters
    ----------
    config : dict
        Autoencoder training config.
    dataset_id : str
        Dataset identifier.
    model_id : str
        Model identifier.
    """

    load_start = time.perf_counter()
    data = load_autoencoder_training_data(
        config=config,
        dataset_id=dataset_id,
        model_id=model_id,
    )
    load_elapsed = time.perf_counter() - load_start
    model_config = config.get("model", {})
    loss_config = resolve_autoencoder_loss_config(config.get("loss", {}))
    training_config = config.get("training", {})

    channels = tuple(
        int(channel)
        for channel in model_config.get("channels", [16, 32, 64])
    )
    latent_dim = int(model_config.get("latent_dim", 64))
    encoder_pooling = str(model_config.get("encoder_pooling", "average"))
    encoder_adaptive_pool_shape = resolve_optional_pool_shape(
        model_config.get("encoder_adaptive_pool_shape", [4, 4])
    )
    learning_rate = float(training_config.get("learning_rate", 0.0003))
    weight_decay = float(training_config.get("weight_decay", 0.0001))
    max_epochs = int(training_config.get("max_epochs", 200))
    early_stopping = bool(training_config.get("early_stopping", True))
    patience = int(training_config.get("patience", 20))
    tolerance = float(training_config.get("tolerance", 1e-5))
    random_seed = int(training_config.get("random_state", 1234))
    deterministic = bool(training_config.get("deterministic", False))
    batch_size = int(training_config.get("batch_size", 64))
    num_workers = int(training_config.get("num_workers", 0))
    prefetch_factor = int(training_config.get("prefetch_factor", 2))
    persistent_workers = bool(
        training_config.get("persistent_workers", num_workers > 0)
    )
    multi_gpu = bool(training_config.get("multi_gpu", True))
    device_ids = training_config.get("device_ids", "all")

    if learning_rate <= 0.0:
        raise ValueError("training.learning_rate must be positive")
    if weight_decay < 0.0:
        raise ValueError("training.weight_decay must be non-negative")
    if max_epochs <= 0:
        raise ValueError("training.max_epochs must be positive")
    if patience <= 0:
        raise ValueError("training.patience must be positive")
    if batch_size <= 0:
        raise ValueError("training.batch_size must be positive")
    if num_workers < 0:
        raise ValueError("training.num_workers must be non-negative")
    if prefetch_factor <= 0:
        raise ValueError("training.prefetch_factor must be positive")

    _set_random_seed(random_seed, deterministic)
    device = _resolve_device(training_config.get("device", "auto"))
    input_shape = (1, *data["slice_shape"])

    model = VdfSliceAutoencoder2D(
        input_shape=input_shape,
        channels=channels,
        latent_dim=latent_dim,
        encoder_pooling=encoder_pooling,
        encoder_adaptive_pool_shape=encoder_adaptive_pool_shape,
    ).to(device)
    model = maybe_wrap_data_parallel(
        model=model,
        device=device,
        multi_gpu=multi_gpu,
        device_ids=device_ids,
    )
    loss_function = VdfAutoencoderLoss(
        mean=data["normalization"]["mean"],
        std=data["normalization"]["std"],
        high_signal_loss_weight=loss_config["high_signal_loss_weight"],
        high_signal_quantile=loss_config["high_signal_quantile"],
        mass_loss_weight=loss_config["mass_loss_weight"],
        peak_loss_weight=loss_config["peak_loss_weight"],
    )

    train_loader = create_data_loader(
        dataset=data["train_dataset"],
        batch_size=batch_size,
        shuffle=True,
        device=device,
        random_seed=random_seed,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )
    validation_loader = create_data_loader(
        dataset=data["validation_dataset"],
        batch_size=batch_size,
        shuffle=False,
        device=device,
        random_seed=random_seed,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )
    test_loader = create_data_loader(
        dataset=data["test_dataset"],
        batch_size=batch_size,
        shuffle=False,
        device=device,
        random_seed=random_seed,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )

    base_model = get_base_model(model)
    print(f"Dataset directory: {data['dataset_dir']}")
    print(f"Output directory: {data['output_dir']}")
    print(f"Raw dataset shape: {data['X'].shape}")
    print(f"Input slice shape: {input_shape}")
    print(f"Training device: {device}")
    print(f"Training module: {model.__class__.__name__}")
    print(f"CUDA devices used: {get_model_device_ids(model)}")
    print(f"Data loader workers: {num_workers}")
    print(f"Persistent workers: {persistent_workers and num_workers > 0}")
    print(f"Prefetch factor: {prefetch_factor if num_workers > 0 else 'disabled'}")
    print(f"Latent dimension: {base_model.latent_dim}")
    print(f"Data loading/preprocessing setup: {load_elapsed:.2f} s")

    fit_start = time.perf_counter()
    training_result = fit_autoencoder(
        model=model,
        train_loader=train_loader,
        validation_loader=validation_loader,
        loss_function=loss_function,
        device=device,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        max_epochs=max_epochs,
        early_stopping=early_stopping,
        patience=patience,
        tolerance=tolerance,
    )
    training_result["fit_elapsed_seconds"] = time.perf_counter() - fit_start

    model.eval()
    output_dir = data["output_dir"]
    checkpoint_path = output_dir / "autoencoder.pt"
    save_autoencoder_checkpoint(model=model, checkpoint_path=checkpoint_path)

    training_history = pd.DataFrame(training_result["history"])
    training_history_path = output_dir / "training_history.csv"
    training_history.to_csv(training_history_path, index=False)

    metrics_start = time.perf_counter()
    reconstruction_metrics = create_reconstruction_metrics(
        model=model,
        loaders={
            "train": train_loader,
            "validation": validation_loader,
            "test": test_loader,
        },
        data=data,
        loss_config=loss_config,
        device=device,
    )
    reconstruction_metrics_path = output_dir / "reconstruction_metrics.csv"
    reconstruction_metrics.to_csv(reconstruction_metrics_path, index=False)
    reconstruction_metrics_elapsed = time.perf_counter() - metrics_start

    latent_start = time.perf_counter()
    latent_embeddings, latent_metadata = create_latent_outputs(
        model=model,
        loaders={
            "train": train_loader,
            "validation": validation_loader,
            "test": test_loader,
        },
        data=data,
        device=device,
    )
    latent_path = output_dir / "latent_embeddings.npz"
    np.savez(
        latent_path,
        embeddings=latent_embeddings,
        sample_indices=latent_metadata["sample_index"].to_numpy(dtype=int),
        labels=latent_metadata["label"].to_numpy(dtype=int),
        splits=latent_metadata["split"].to_numpy(dtype=str),
    )
    latent_metadata_path = output_dir / "latent_metadata.csv"
    latent_metadata.to_csv(latent_metadata_path, index=False)
    plot_latent_pca(
        embeddings=latent_embeddings,
        metadata=latent_metadata,
        output_path=output_dir / "latent_pca.png",
    )
    latent_elapsed = time.perf_counter() - latent_start

    plot_start = time.perf_counter()
    reconstruction_plot_paths = save_reconstruction_plots(
        model=model,
        data=data,
        reconstruction_metrics=reconstruction_metrics,
        config=config,
        device=device,
    )
    plot_elapsed = time.perf_counter() - plot_start

    preprocessing_path = output_dir / "preprocessing.npz"
    save_preprocessing(
        preprocessing_path=preprocessing_path,
        data=data,
        dataset_id=dataset_id,
        model_id=model_id,
        channels=channels,
        latent_dim=latent_dim,
        encoder_pooling=encoder_pooling,
        encoder_adaptive_pool_shape=encoder_adaptive_pool_shape,
        loss_config=loss_config,
        batch_size=batch_size,
    )

    metrics_text = create_autoencoder_metrics_text(
        dataset_id=dataset_id,
        model_id=model_id,
        data=data,
        model=model,
        loss_config=loss_config,
        training_result=training_result,
        reconstruction_metrics=reconstruction_metrics,
        checkpoint_path=checkpoint_path,
        preprocessing_path=preprocessing_path,
        training_history_path=training_history_path,
        reconstruction_metrics_path=reconstruction_metrics_path,
        latent_path=latent_path,
        latent_metadata_path=latent_metadata_path,
        reconstruction_plot_paths=reconstruction_plot_paths,
        device=device,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        max_epochs=max_epochs,
        early_stopping=early_stopping,
        patience=patience,
        tolerance=tolerance,
        random_seed=random_seed,
        deterministic=deterministic,
        load_elapsed=load_elapsed,
        reconstruction_metrics_elapsed=reconstruction_metrics_elapsed,
        latent_elapsed=latent_elapsed,
        plot_elapsed=plot_elapsed,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )
    metrics_path = output_dir / "metrics.txt"
    with open(metrics_path, "w") as metrics_file:
        metrics_file.write(metrics_text)

    print(checkpoint_path)
    print(preprocessing_path)
    print(training_history_path)
    print(reconstruction_metrics_path)
    print(latent_path)
    print(latent_metadata_path)
    print(metrics_path)
    print(f"Saved reconstruction plots: {len(reconstruction_plot_paths)}")
    print(f"Training fit time: {training_result['fit_elapsed_seconds']:.2f} s")
    print(f"Reconstruction metrics time: {reconstruction_metrics_elapsed:.2f} s")
    print(f"Latent export time: {latent_elapsed:.2f} s")
    print(f"Reconstruction plot time: {plot_elapsed:.2f} s")


def create_data_loader(
    dataset,
    batch_size,
    shuffle,
    device,
    random_seed,
    num_workers=0,
    persistent_workers=False,
    prefetch_factor=2,
):
    """
    Create a PyTorch data loader for VDF slices.

    Parameters
    ----------
    dataset : torch.utils.data.Dataset
        Dataset to load.
    batch_size : int
        Number of samples per batch.
    shuffle : bool
        Whether to shuffle samples.
    device : torch.device
        Training device.
    random_seed : int
        Random seed used by the data loader.
    num_workers : int, optional
        Number of worker processes.
    persistent_workers : bool, optional
        Whether worker processes stay alive between epochs.
    prefetch_factor : int, optional
        Number of batches loaded in advance by each worker.

    Returns
    -------
    torch.utils.data.DataLoader
        Configured data loader.
    """

    batch_size = min(int(batch_size), len(dataset))
    if batch_size <= 0:
        raise ValueError("Cannot create a DataLoader for an empty dataset")

    num_workers = int(num_workers)
    data_loader_kwargs = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": bool(shuffle),
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
        "generator": torch.Generator().manual_seed(int(random_seed)),
    }

    if num_workers > 0:
        data_loader_kwargs["persistent_workers"] = bool(persistent_workers)
        data_loader_kwargs["prefetch_factor"] = int(prefetch_factor)

    return DataLoader(**data_loader_kwargs)


def fit_autoencoder(
    model,
    train_loader,
    validation_loader,
    loss_function,
    device,
    learning_rate,
    weight_decay,
    max_epochs,
    early_stopping,
    patience,
    tolerance,
):
    """
    Fit an autoencoder with validation-based early stopping.

    Parameters
    ----------
    model : VdfSliceAutoencoder2D
        Autoencoder model.
    train_loader : torch.utils.data.DataLoader
        Training data loader.
    validation_loader : torch.utils.data.DataLoader
        Validation data loader.
    loss_function : torch.nn.Module
        Training and validation loss.
    device : torch.device
        Training device.
    learning_rate : float
        AdamW learning rate.
    weight_decay : float
        AdamW weight decay.
    max_epochs : int
        Maximum number of epochs.
    early_stopping : bool
        Whether to restore the best validation checkpoint.
    patience : int
        Epochs without validation improvement before stopping.
    tolerance : float
        Minimum improvement that resets patience.

    Returns
    -------
    dict
        Training history and best validation epoch.
    """

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(learning_rate),
        weight_decay=float(weight_decay),
    )
    history = []
    best_validation_loss = None
    best_epoch = 0
    best_state_dict = None
    epochs_without_improvement = 0

    for epoch in range(1, int(max_epochs) + 1):
        train_loss = run_autoencoder_epoch(
            model=model,
            data_loader=train_loader,
            loss_function=loss_function,
            optimizer=optimizer,
            device=device,
        )
        validation_loss = evaluate_autoencoder_loss(
            model=model,
            data_loader=validation_loader,
            loss_function=loss_function,
            device=device,
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
            }
        )

        significantly_improved = (
            best_validation_loss is None
            or validation_loss < best_validation_loss - float(tolerance)
        )
        improved = (
            best_validation_loss is None
            or validation_loss < best_validation_loss
        )

        if improved:
            best_validation_loss = validation_loss
            best_epoch = epoch
            if early_stopping:
                best_state_dict = deepcopy(
                    {
                        name: value.detach().cpu()
                        for name, value in model.state_dict().items()
                    }
                )

        if significantly_improved:
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        print(
            f"Epoch {epoch}: "
            f"train_loss={train_loss:.6g}, "
            f"validation_loss={validation_loss:.6g}"
        )

        if early_stopping and epochs_without_improvement >= int(patience):
            break

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    return {
        "history": history,
        "n_epochs": int(history[-1]["epoch"]),
        "best_epoch": int(best_epoch),
        "best_validation_loss": float(best_validation_loss),
        "final_train_loss": float(history[-1]["train_loss"]),
        "final_validation_loss": float(history[-1]["validation_loss"]),
    }


def run_autoencoder_epoch(
    model,
    data_loader,
    loss_function,
    optimizer,
    device,
):
    """Run one training epoch and return mean batch loss."""

    model.train()
    total_loss = 0.0
    total_samples = 0

    for batch in data_loader:
        inputs = batch["input"].to(device, non_blocking=device.type == "cuda")
        targets = batch["target"].to(device, non_blocking=device.type == "cuda")

        optimizer.zero_grad(set_to_none=True)
        reconstruction = model(inputs)
        loss = loss_function(reconstruction, targets)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.detach().cpu()) * len(inputs)
        total_samples += len(inputs)

    return total_loss / total_samples


def evaluate_autoencoder_loss(model, data_loader, loss_function, device):
    """Return mean reconstruction loss for a data loader."""

    model.eval()
    total_loss = 0.0
    total_samples = 0

    with torch.inference_mode():
        for batch in data_loader:
            inputs = batch["input"].to(
                device,
                non_blocking=device.type == "cuda",
            )
            targets = batch["target"].to(
                device,
                non_blocking=device.type == "cuda",
            )
            loss = loss_function(model(inputs), targets)
            total_loss += float(loss.cpu()) * len(inputs)
            total_samples += len(inputs)

    return total_loss / total_samples


def create_reconstruction_metrics(model, loaders, data, loss_config, device):
    """
    Create per-sample reconstruction metrics for all splits.

    Parameters
    ----------
    model : VdfSliceAutoencoder2D
        Trained autoencoder.
    loaders : dict
        Mapping from split name to data loader.
    data : dict
        Autoencoder training data.
    loss_config : dict
        Autoencoder loss settings.
    device : torch.device
        Inference device.

    Returns
    -------
    pandas.DataFrame
        Per-sample reconstruction metrics.
    """

    metric_frames = []
    normalization = data["normalization"]
    input_config = data["input_config"]
    high_signal_quantile = float(loss_config.get("high_signal_quantile", 0.95))

    model.eval()
    with torch.inference_mode():
        for split_name, data_loader in loaders.items():
            split_rows = []

            for batch in data_loader:
                inputs = batch["input"].to(
                    device,
                    non_blocking=device.type == "cuda",
                )
                targets = batch["target"].to(
                    device,
                    non_blocking=device.type == "cuda",
                )
                reconstruction = model(inputs)
                losses = torch.mean(
                    torch.square(reconstruction - targets),
                    dim=(1, 2, 3),
                ).cpu().numpy()
                high_signal_losses = (
                    per_sample_high_signal_reconstruction_loss(
                        reconstruction=reconstruction,
                        target=targets,
                        quantile=high_signal_quantile,
                    )
                    .cpu()
                    .numpy()
                )

                original = normalized_log_slice_to_physical(
                    normalized_slice=targets.cpu().numpy()[:, 0, :, :],
                    mean=normalization["mean"],
                    std=normalization["std"],
                    log_eps=input_config["log_eps"],
                )
                reconstructed = normalized_log_slice_to_physical(
                    normalized_slice=reconstruction.cpu().numpy()[:, 0, :, :],
                    mean=normalization["mean"],
                    std=normalization["std"],
                    log_eps=input_config["log_eps"],
                )
                original_mass = original.sum(axis=(1, 2))
                reconstructed_mass = reconstructed.sum(axis=(1, 2))
                original_peak = original.max(axis=(1, 2))
                reconstructed_peak = reconstructed.max(axis=(1, 2))
                original_mass_log10 = _log10_with_eps(
                    original_mass,
                    input_config["log_eps"],
                )
                reconstructed_mass_log10 = _log10_with_eps(
                    reconstructed_mass,
                    input_config["log_eps"],
                )
                original_peak_log10 = _log10_with_eps(
                    original_peak,
                    input_config["log_eps"],
                )
                reconstructed_peak_log10 = _log10_with_eps(
                    reconstructed_peak,
                    input_config["log_eps"],
                )

                for row_index, sample_index in enumerate(
                    batch["sample_index"].cpu().numpy()
                ):
                    sample_index = int(sample_index)
                    metadata_row = data["metadata"].iloc[sample_index].to_dict()
                    mass_error = (
                        reconstructed_mass[row_index] - original_mass[row_index]
                    )
                    peak_error = (
                        reconstructed_peak[row_index] - original_peak[row_index]
                    )
                    mass_relative_error = _safe_relative_error(
                        error=mass_error,
                        reference=original_mass[row_index],
                    )
                    peak_relative_error = _safe_relative_error(
                        error=peak_error,
                        reference=original_peak[row_index],
                    )
                    mass_log10_error = (
                        reconstructed_mass_log10[row_index]
                        - original_mass_log10[row_index]
                    )
                    peak_log10_error = (
                        reconstructed_peak_log10[row_index]
                        - original_peak_log10[row_index]
                    )
                    split_rows.append(
                        {
                            "split": split_name,
                            "sample_index": sample_index,
                            "label": int(batch["label"][row_index]),
                            "class_name": metadata_row.get("class_name"),
                            "timestep": metadata_row.get("timestep"),
                            "cid": metadata_row.get("cid"),
                            "reconstruction_loss": float(losses[row_index]),
                            "high_signal_reconstruction_loss": float(
                                high_signal_losses[row_index]
                            ),
                            "slice_mass": float(original_mass[row_index]),
                            "reconstructed_slice_mass": float(
                                reconstructed_mass[row_index]
                            ),
                            "slice_mass_error": float(mass_error),
                            "slice_mass_relative_error": mass_relative_error,
                            "slice_mass_absolute_relative_error": abs(
                                mass_relative_error
                            ),
                            "slice_mass_log10": float(original_mass_log10[row_index]),
                            "reconstructed_slice_mass_log10": float(
                                reconstructed_mass_log10[row_index]
                            ),
                            "slice_mass_log10_error": float(mass_log10_error),
                            "slice_mass_log10_absolute_error": abs(
                                float(mass_log10_error)
                            ),
                            "peak_value": float(original_peak[row_index]),
                            "reconstructed_peak_value": float(
                                reconstructed_peak[row_index]
                            ),
                            "peak_value_error": float(peak_error),
                            "peak_value_relative_error": peak_relative_error,
                            "peak_value_absolute_relative_error": abs(
                                peak_relative_error
                            ),
                            "peak_value_log10": float(original_peak_log10[row_index]),
                            "reconstructed_peak_value_log10": float(
                                reconstructed_peak_log10[row_index]
                            ),
                            "peak_value_log10_error": float(peak_log10_error),
                            "peak_value_log10_absolute_error": abs(
                                float(peak_log10_error)
                            ),
                        }
                    )

            metric_frames.append(pd.DataFrame(split_rows))

    return pd.concat(metric_frames, ignore_index=True)


def create_latent_outputs(model, loaders, data, device):
    """
    Encode all splits and return latent arrays with metadata.

    Parameters
    ----------
    model : VdfSliceAutoencoder2D
        Trained autoencoder.
    loaders : dict
        Mapping from split name to data loader.
    data : dict
        Autoencoder training data.
    device : torch.device
        Inference device.

    Returns
    -------
    embeddings : numpy.ndarray
        Latent vectors.
    latent_metadata : pandas.DataFrame
        Metadata rows with split and latent columns.
    """

    embeddings = []
    metadata_rows = []

    model.eval()
    with torch.inference_mode():
        for split_name, data_loader in loaders.items():
            for batch in data_loader:
                inputs = batch["input"].to(
                    device,
                    non_blocking=device.type == "cuda",
                )
                latents = encode_model(model, inputs).cpu().numpy()
                sample_indices = batch["sample_index"].cpu().numpy()

                for row_index, sample_index in enumerate(sample_indices):
                    sample_index = int(sample_index)
                    metadata_row = data["metadata"].iloc[sample_index].to_dict()
                    metadata_row["split"] = split_name
                    metadata_row["sample_index"] = sample_index
                    metadata_row["label"] = int(batch["label"][row_index])
                    for latent_index, latent_value in enumerate(latents[row_index]):
                        metadata_row[f"latent_{latent_index:03d}"] = float(
                            latent_value
                        )
                    metadata_rows.append(metadata_row)

                embeddings.append(latents)

    return np.concatenate(embeddings, axis=0), pd.DataFrame(metadata_rows)


def save_reconstruction_plots(model, data, reconstruction_metrics, config, device):
    """
    Save side-by-side original and reconstructed VDF plots.

    Parameters
    ----------
    model : VdfSliceAutoencoder2D
        Trained autoencoder.
    data : dict
        Autoencoder training data.
    reconstruction_metrics : pandas.DataFrame
        Per-sample reconstruction metrics.
    config : dict
        Autoencoder training config.
    device : torch.device
        Inference device.

    Returns
    -------
    list of pathlib.Path
        Paths of saved reconstruction plots.
    """

    plot_config = config.get("plot", {})
    if not bool(plot_config.get("enabled", True)):
        return []

    splits = plot_config.get("splits", ["train", "validation", "test"])
    max_per_class = int(plot_config.get("max_per_class", 3))
    vdflim = float(plot_config.get("vdflim", 2e6))
    selected_rows = select_reconstruction_plot_rows(
        reconstruction_metrics=reconstruction_metrics,
        splits=splits,
        max_per_class=max_per_class,
    )

    plot_paths = []
    plot_parameter_cache = {}
    model.eval()

    with torch.inference_mode():
        for _, row in selected_rows.iterrows():
            sample_index = int(row["sample_index"])
            split_name = str(row["split"])
            metadata_row = data["metadata"].iloc[sample_index].to_dict()
            original_slice, reconstructed_slice = reconstruct_physical_slice(
                model=model,
                X=data["X"],
                sample_index=sample_index,
                normalization=data["normalization"],
                input_config=data["input_config"],
                device=device,
            )
            extent, dv, threshold = resolve_vdf_plot_parameters(
                metadata_row=metadata_row,
                vdf_shape=tuple(data["X"].shape[1:]),
                vdflim=vdflim,
                cache=plot_parameter_cache,
            )
            output_path = (
                data["output_dir"]
                / "reconstructions"
                / split_name
                / str(metadata_row.get("class_name", "unknown"))
                / create_reconstruction_filename(
                    metadata_row=metadata_row,
                    split_name=split_name,
                    sample_index=sample_index,
                )
            )
            saved = plot_reconstruction_pair(
                original_slice=original_slice,
                reconstructed_slice=reconstructed_slice,
                metadata_row=metadata_row,
                output_path=output_path,
                extent=extent,
                dv=dv,
                threshold=threshold,
                vdflim=vdflim,
                split_name=split_name,
                sample_index=sample_index,
            )
            if saved:
                plot_paths.append(output_path)

    return plot_paths


def select_reconstruction_plot_rows(reconstruction_metrics, splits, max_per_class):
    """
    Select reconstruction examples to plot by split and class.

    Parameters
    ----------
    reconstruction_metrics : pandas.DataFrame
        Per-sample reconstruction metrics.
    splits : sequence of str
        Split names to include.
    max_per_class : int
        Maximum number of samples per class and split.

    Returns
    -------
    pandas.DataFrame
        Selected metric rows.
    """

    max_per_class = int(max_per_class)
    if max_per_class <= 0:
        raise ValueError("plot.max_per_class must be positive")

    selected = []
    for split_name in splits:
        split_rows = reconstruction_metrics[
            reconstruction_metrics["split"] == str(split_name)
        ]
        if split_rows.empty:
            continue

        if split_name in {"validation", "test"}:
            split_rows = split_rows.sort_values(
                "reconstruction_loss",
                ascending=False,
            )
        else:
            split_rows = split_rows.sort_values("sample_index")

        for _, class_rows in split_rows.groupby("class_name", sort=True):
            selected.append(class_rows.head(max_per_class))

    if not selected:
        return reconstruction_metrics.head(0)

    return pd.concat(selected, ignore_index=True)


def reconstruct_physical_slice(
    model,
    X,
    sample_index,
    normalization,
    input_config,
    device,
):
    """
    Reconstruct one VDF sample as physical xz slices.

    Parameters
    ----------
    model : VdfSliceAutoencoder2D
        Trained autoencoder.
    X : numpy.ndarray
        VDF samples with shape ``(n_samples, vx, vy, vz)``.
    sample_index : int
        Dataset sample index.
    normalization : dict
        Log-slice normalization values.
    input_config : dict
        VDF preprocessing settings.
    device : torch.device
        Inference device.

    Returns
    -------
    original_slice : numpy.ndarray
        Original physical VDF xz slice.
    reconstructed_slice : numpy.ndarray
        Reconstructed physical VDF xz slice.
    """

    original_slice = extract_plot_xz_slice_from_dataset(
        X=X,
        sample_index=sample_index,
    )
    slice_log = create_log_slice(
        physical_slice=original_slice,
        log_eps=input_config["log_eps"],
        clip_negative_to_zero=input_config["clip_negative_to_zero"],
    )
    normalized_slice = normalize_log_slice(
        slice_log=slice_log,
        mean=normalization["mean"],
        std=normalization["std"],
    )
    inputs = torch.from_numpy(normalized_slice[None, None, ...]).to(device)
    reconstruction = model(inputs).cpu().numpy()[0, 0]
    reconstructed_slice = normalized_log_slice_to_physical(
        normalized_slice=reconstruction,
        mean=normalization["mean"],
        std=normalization["std"],
        log_eps=input_config["log_eps"],
    )

    return np.maximum(original_slice, 0.0), reconstructed_slice


def save_preprocessing(
    preprocessing_path,
    data,
    dataset_id,
    model_id,
    channels,
    latent_dim,
    encoder_pooling,
    encoder_adaptive_pool_shape,
    loss_config,
    batch_size,
):
    """
    Save autoencoder preprocessing and split metadata.

    Parameters
    ----------
    preprocessing_path : str or pathlib.Path
        Path to saved preprocessing file.
    data : dict
        Autoencoder training data.
    dataset_id : str
        Dataset identifier.
    model_id : str
        Model identifier.
    channels : sequence of int
        Autoencoder channel configuration.
    latent_dim : int
        Latent vector size.
    encoder_pooling : str
        Encoder downsampling method.
    encoder_adaptive_pool_shape : tuple of int or None
        Encoder adaptive pooling output shape.
    loss_config : dict
        Autoencoder loss settings.
    batch_size : int
        Training batch size.
    """

    if encoder_adaptive_pool_shape is None:
        adaptive_pool_shape = np.asarray([], dtype=int)
    else:
        adaptive_pool_shape = np.asarray(encoder_adaptive_pool_shape, dtype=int)

    np.savez(
        preprocessing_path,
        dataset_id=np.asarray(dataset_id),
        model_id=np.asarray(model_id),
        input_slice=np.asarray(data["input_config"]["slice"]),
        input_orientation=np.asarray(data["input_config"]["orientation"]),
        input_normalization=np.asarray(data["input_config"]["normalization"]),
        log_eps=np.asarray(data["input_config"]["log_eps"]),
        clip_negative_to_zero=np.asarray(
            data["input_config"]["clip_negative_to_zero"]
        ),
        input_mean=np.asarray(data["normalization"]["mean"]),
        input_std=np.asarray(data["normalization"]["std"]),
        input_log_min=np.asarray(data["normalization"]["log_min"]),
        input_log_max=np.asarray(data["normalization"]["log_max"]),
        normalization_n_samples=np.asarray(data["normalization"]["n_samples"]),
        normalization_n_values=np.asarray(data["normalization"]["n_values"]),
        cache_enabled=np.asarray(data["cache_metadata"].get("enabled", False)),
        cache_path=np.asarray(data["cache_metadata"].get("cache_path", "")),
        cache_metadata_path=np.asarray(
            data["cache_metadata"].get("metadata_path", "")
        ),
        slice_shape=np.asarray(data["slice_shape"], dtype=int),
        raw_vdf_shape=np.asarray(data["X"].shape[1:], dtype=int),
        train_fraction=np.asarray(data["train_fraction"]),
        validation_fraction=np.asarray(data["validation_fraction"]),
        gap_timesteps=np.asarray(data["gap_timesteps"]),
        train_timesteps=data["train_timesteps"],
        validation_timesteps=data["validation_timesteps"],
        test_timesteps=data["test_timesteps"],
        train_validation_gap=data["train_validation_gap"],
        validation_test_gap=data["validation_test_gap"],
        channels=np.asarray(channels, dtype=int),
        latent_dim=np.asarray(latent_dim),
        encoder_pooling=np.asarray(encoder_pooling),
        encoder_adaptive_pool_shape=adaptive_pool_shape,
        high_signal_loss_weight=np.asarray(
            loss_config["high_signal_loss_weight"]
        ),
        high_signal_quantile=np.asarray(loss_config["high_signal_quantile"]),
        mass_loss_weight=np.asarray(loss_config["mass_loss_weight"]),
        peak_loss_weight=np.asarray(loss_config["peak_loss_weight"]),
        batch_size=np.asarray(batch_size),
    )


def create_autoencoder_metrics_text(
    dataset_id,
    model_id,
    data,
    model,
    loss_config,
    training_result,
    reconstruction_metrics,
    checkpoint_path,
    preprocessing_path,
    training_history_path,
    reconstruction_metrics_path,
    latent_path,
    latent_metadata_path,
    reconstruction_plot_paths,
    device,
    batch_size,
    learning_rate,
    weight_decay,
    max_epochs,
    early_stopping,
    patience,
    tolerance,
    random_seed,
    deterministic,
    load_elapsed,
    reconstruction_metrics_elapsed,
    latent_elapsed,
    plot_elapsed,
    num_workers,
    persistent_workers,
    prefetch_factor,
):
    """
    Create a text summary of autoencoder training and evaluation.

    Parameters
    ----------
    dataset_id : str
        Dataset identifier.
    model_id : str
        Model identifier.
    data : dict
        Autoencoder training data.
    model : VdfSliceAutoencoder2D
        Trained model.
    loss_config : dict
        Autoencoder loss settings.
    training_result : dict
        Training history summary.
    reconstruction_metrics : pandas.DataFrame
        Per-sample reconstruction metrics.
    checkpoint_path : pathlib.Path
        Saved checkpoint path.
    preprocessing_path : pathlib.Path
        Saved preprocessing path.
    training_history_path : pathlib.Path
        Saved training history path.
    reconstruction_metrics_path : pathlib.Path
        Saved reconstruction metrics path.
    latent_path : pathlib.Path
        Saved latent embedding path.
    latent_metadata_path : pathlib.Path
        Saved latent metadata path.
    reconstruction_plot_paths : list of pathlib.Path
        Saved reconstruction plot paths.
    device : torch.device
        Training device.
    batch_size : int
        Training batch size.
    learning_rate : float
        AdamW learning rate.
    weight_decay : float
        AdamW weight decay.
    max_epochs : int
        Maximum epochs.
    early_stopping : bool
        Whether early stopping was enabled.
    patience : int
        Early stopping patience.
    tolerance : float
        Early stopping tolerance.
    random_seed : int
        Random seed.
    deterministic : bool
        Whether deterministic algorithms were requested.

    Returns
    -------
    str
        Metrics summary.
    """

    base_model = get_base_model(model)
    split_summary = summarize_split_losses(reconstruction_metrics)
    class_summary = summarize_class_losses(reconstruction_metrics, "validation")
    worst_validation = (
        reconstruction_metrics[reconstruction_metrics["split"] == "validation"]
        .sort_values("reconstruction_loss", ascending=False)
        .head(12)
    )

    lines = [
        "VDF slice autoencoder evaluation",
        "=" * 70,
        f"Dataset ID: {dataset_id}",
        f"Model ID: {model_id}",
        f"Dataset directory: {data['dataset_dir']}",
        f"Output directory: {data['output_dir']}",
        f"Raw dataset shape: {data['X'].shape}",
        f"Input slice shape: {(1, *data['slice_shape'])}",
        f"Train samples: {len(data['train_indices'])}",
        f"Validation samples: {len(data['validation_indices'])}",
        f"Test samples: {len(data['test_indices'])}",
        "Train timesteps: "
        f"{format_timestep_range(data['train_timesteps'])}",
        "Train-validation gap: "
        f"{format_timestep_range(data['train_validation_gap'])}",
        "Validation timesteps: "
        f"{format_timestep_range(data['validation_timesteps'])}",
        "Validation-test gap: "
        f"{format_timestep_range(data['validation_test_gap'])}",
        "Test timesteps: "
        f"{format_timestep_range(data['test_timesteps'])}",
        f"Input transform: log10(f + {data['input_config']['log_eps']})",
        "Input normalization: train split global mean/std",
        f"Input mean: {data['normalization']['mean']}",
        f"Input std: {data['normalization']['std']}",
        f"Input log min: {data['normalization']['log_min']}",
        f"Input log max: {data['normalization']['log_max']}",
        f"Encoder channels: {base_model.channels}",
        f"Encoder pooling: {base_model.encoder_pooling}",
        "Encoder adaptive pool shape: "
        f"{base_model.encoder_adaptive_pool_shape}",
        f"Latent dimension: {base_model.latent_dim}",
        f"Training module: {model.__class__.__name__}",
        f"CUDA devices used: {get_model_device_ids(model)}",
        "Activation: relu",
        "Optimizer: AdamW",
        "Loss reconstruction term: normalized log-space MSE",
        f"Loss high-signal weight: {loss_config['high_signal_loss_weight']}",
        f"Loss high-signal quantile: {loss_config['high_signal_quantile']}",
        f"Loss log-mass weight: {loss_config['mass_loss_weight']}",
        f"Loss log-peak weight: {loss_config['peak_loss_weight']}",
        f"Batch size: {batch_size}",
        f"Data loader workers: {num_workers}",
        f"Persistent workers: {persistent_workers and num_workers > 0}",
        f"Prefetch factor: {prefetch_factor if num_workers > 0 else 'disabled'}",
        f"Learning rate: {learning_rate}",
        f"Weight decay: {weight_decay}",
        f"Max epochs: {max_epochs}",
        f"Early stopping: {early_stopping}",
        f"Patience: {patience}",
        f"Tolerance: {tolerance}",
        f"Random seed: {random_seed}",
        f"Device: {device}",
        f"Deterministic algorithms: {deterministic}",
        f"PyTorch version: {torch.__version__}",
        f"Epochs: {training_result['n_epochs']}",
        f"Best epoch: {training_result['best_epoch']}",
        f"Best validation loss: {training_result['best_validation_loss']}",
        f"Final train loss: {training_result['final_train_loss']}",
        f"Final validation loss: {training_result['final_validation_loss']}",
        f"Data loading/preprocessing setup seconds: {load_elapsed}",
        f"Fit seconds: {training_result['fit_elapsed_seconds']}",
        f"Reconstruction metrics seconds: {reconstruction_metrics_elapsed}",
        f"Latent export seconds: {latent_elapsed}",
        f"Reconstruction plot seconds: {plot_elapsed}",
        f"Cache enabled: {data['cache_metadata'].get('enabled', False)}",
        f"Cache path: {data['cache_metadata'].get('cache_path', 'none')}",
        "",
        "Split reconstruction losses",
        "=" * 70,
        split_summary.to_string(index=False),
        "",
        "Validation class reconstruction losses",
        "=" * 70,
        class_summary.to_string(index=False),
        "",
        "Worst validation reconstructions",
        "=" * 70,
        worst_validation[
            [
                "sample_index",
                "timestep",
                "cid",
                "class_name",
                "reconstruction_loss",
                "high_signal_reconstruction_loss",
                "slice_mass_relative_error",
                "peak_value_relative_error",
                "slice_mass_log10_error",
                "peak_value_log10_error",
            ]
        ].to_string(index=False),
        "",
        "Saved outputs",
        "=" * 70,
        f"Checkpoint: {checkpoint_path}",
        f"Preprocessing: {preprocessing_path}",
        f"Training history: {training_history_path}",
        f"Reconstruction metrics: {reconstruction_metrics_path}",
        f"Latent embeddings: {latent_path}",
        f"Latent metadata: {latent_metadata_path}",
        f"Latent PCA: {data['output_dir'] / 'latent_pca.png'}",
        f"Side-by-side reconstruction plots: {len(reconstruction_plot_paths)}",
    ]

    return "\n".join(lines)


def summarize_split_losses(reconstruction_metrics):
    """
    Summarize reconstruction losses by split.

    Parameters
    ----------
    reconstruction_metrics : pandas.DataFrame
        Per-sample reconstruction metrics.

    Returns
    -------
    pandas.DataFrame
        Split-level summary.
    """

    return (
        reconstruction_metrics.groupby("split", sort=True)
        .agg(
            n_samples=("reconstruction_loss", "size"),
            mean_loss=("reconstruction_loss", "mean"),
            median_loss=("reconstruction_loss", "median"),
            max_loss=("reconstruction_loss", "max"),
            mean_high_signal_loss=("high_signal_reconstruction_loss", "mean"),
            median_high_signal_loss=("high_signal_reconstruction_loss", "median"),
            max_high_signal_loss=("high_signal_reconstruction_loss", "max"),
            mean_mass_relative_error=("slice_mass_relative_error", "mean"),
            median_abs_mass_rel_error=(
                "slice_mass_absolute_relative_error",
                "median",
            ),
            p90_abs_mass_rel_error=(
                "slice_mass_absolute_relative_error",
                _quantile_90,
            ),
            p99_abs_mass_rel_error=(
                "slice_mass_absolute_relative_error",
                _quantile_99,
            ),
            median_abs_mass_log10_error=(
                "slice_mass_log10_absolute_error",
                "median",
            ),
            p90_abs_mass_log10_error=(
                "slice_mass_log10_absolute_error",
                _quantile_90,
            ),
            p99_abs_mass_log10_error=(
                "slice_mass_log10_absolute_error",
                _quantile_99,
            ),
            mean_peak_relative_error=("peak_value_relative_error", "mean"),
            median_abs_peak_rel_error=(
                "peak_value_absolute_relative_error",
                "median",
            ),
            p90_abs_peak_rel_error=(
                "peak_value_absolute_relative_error",
                _quantile_90,
            ),
            p99_abs_peak_rel_error=(
                "peak_value_absolute_relative_error",
                _quantile_99,
            ),
            median_abs_peak_log10_error=(
                "peak_value_log10_absolute_error",
                "median",
            ),
            p90_abs_peak_log10_error=(
                "peak_value_log10_absolute_error",
                _quantile_90,
            ),
            p99_abs_peak_log10_error=(
                "peak_value_log10_absolute_error",
                _quantile_99,
            ),
        )
        .reset_index()
    )


def summarize_class_losses(reconstruction_metrics, split_name):
    """
    Summarize reconstruction losses by class for one split.

    Parameters
    ----------
    reconstruction_metrics : pandas.DataFrame
        Per-sample reconstruction metrics.
    split_name : str
        Split name to summarize.

    Returns
    -------
    pandas.DataFrame
        Class-level summary.
    """

    split_rows = reconstruction_metrics[
        reconstruction_metrics["split"] == split_name
    ]
    if split_rows.empty:
        return pd.DataFrame(
            columns=[
                "class_name",
                "n_samples",
                "mean_loss",
                "median_loss",
                "max_loss",
                "mean_high_signal_loss",
                "median_high_signal_loss",
                "max_high_signal_loss",
                "mean_mass_relative_error",
                "median_abs_mass_rel_error",
                "p90_abs_mass_rel_error",
                "p99_abs_mass_rel_error",
                "median_abs_mass_log10_error",
                "p90_abs_mass_log10_error",
                "p99_abs_mass_log10_error",
                "mean_peak_relative_error",
                "median_abs_peak_rel_error",
                "p90_abs_peak_rel_error",
                "p99_abs_peak_rel_error",
                "median_abs_peak_log10_error",
                "p90_abs_peak_log10_error",
                "p99_abs_peak_log10_error",
            ]
        )

    return (
        split_rows.groupby("class_name", sort=True)
        .agg(
            n_samples=("reconstruction_loss", "size"),
            mean_loss=("reconstruction_loss", "mean"),
            median_loss=("reconstruction_loss", "median"),
            max_loss=("reconstruction_loss", "max"),
            mean_high_signal_loss=("high_signal_reconstruction_loss", "mean"),
            median_high_signal_loss=("high_signal_reconstruction_loss", "median"),
            max_high_signal_loss=("high_signal_reconstruction_loss", "max"),
            mean_mass_relative_error=("slice_mass_relative_error", "mean"),
            median_abs_mass_rel_error=(
                "slice_mass_absolute_relative_error",
                "median",
            ),
            p90_abs_mass_rel_error=(
                "slice_mass_absolute_relative_error",
                _quantile_90,
            ),
            p99_abs_mass_rel_error=(
                "slice_mass_absolute_relative_error",
                _quantile_99,
            ),
            median_abs_mass_log10_error=(
                "slice_mass_log10_absolute_error",
                "median",
            ),
            p90_abs_mass_log10_error=(
                "slice_mass_log10_absolute_error",
                _quantile_90,
            ),
            p99_abs_mass_log10_error=(
                "slice_mass_log10_absolute_error",
                _quantile_99,
            ),
            mean_peak_relative_error=("peak_value_relative_error", "mean"),
            median_abs_peak_rel_error=(
                "peak_value_absolute_relative_error",
                "median",
            ),
            p90_abs_peak_rel_error=(
                "peak_value_absolute_relative_error",
                _quantile_90,
            ),
            p99_abs_peak_rel_error=(
                "peak_value_absolute_relative_error",
                _quantile_99,
            ),
            median_abs_peak_log10_error=(
                "peak_value_log10_absolute_error",
                "median",
            ),
            p90_abs_peak_log10_error=(
                "peak_value_log10_absolute_error",
                _quantile_90,
            ),
            p99_abs_peak_log10_error=(
                "peak_value_log10_absolute_error",
                _quantile_99,
            ),
        )
        .reset_index()
    )


def save_autoencoder_checkpoint(model, checkpoint_path):
    """
    Save autoencoder weights and architecture metadata.

    Parameters
    ----------
    model : VdfSliceAutoencoder2D
        Trained model.
    checkpoint_path : str or pathlib.Path
        Output checkpoint path.
    """

    base_model = get_base_model(model)
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": 1,
            "model_state_dict": {
                name: value.detach().cpu()
                for name, value in base_model.state_dict().items()
            },
            "input_shape": base_model.input_shape,
            "channels": list(base_model.channels),
            "latent_dim": base_model.latent_dim,
            "encoder_pooling": base_model.encoder_pooling,
            "encoder_adaptive_pool_shape": base_model.encoder_adaptive_pool_shape,
        },
        checkpoint_path,
    )


def load_autoencoder_checkpoint(checkpoint_path, device="cpu"):
    """
    Load a saved VDF slice autoencoder checkpoint.

    Parameters
    ----------
    checkpoint_path : str or pathlib.Path
        Saved checkpoint path.
    device : str or torch.device, optional
        Device where the model is loaded.

    Returns
    -------
    VdfSliceAutoencoder2D
        Loaded autoencoder.
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
            "Unsupported autoencoder checkpoint format version: "
            f"{checkpoint['format_version']}"
        )

    model = VdfSliceAutoencoder2D(
        input_shape=checkpoint["input_shape"],
        channels=checkpoint["channels"],
        latent_dim=checkpoint["latent_dim"],
        encoder_pooling=checkpoint.get("encoder_pooling", "stride"),
        encoder_adaptive_pool_shape=checkpoint.get(
            "encoder_adaptive_pool_shape",
            None,
        ),
    )
    model.load_state_dict(checkpoint["model_state_dict"])

    return model.to(device).eval()


def format_timestep_range(timesteps):
    """
    Format a timestep array as a compact range.

    Parameters
    ----------
    timesteps : array-like of int
        Timesteps.

    Returns
    -------
    str
        Compact range string.
    """

    if len(timesteps) == 0:
        return "none"
    if len(timesteps) == 1:
        return str(timesteps[0])

    return f"{timesteps[0]} ... {timesteps[-1]}"


def _conv_stride2_output_size(size):
    return int(np.floor((int(size) + 2 - 2 - 1) / 2 + 1))


def _pool_stride2_output_size(size):
    return int(np.floor((int(size) - 2) / 2 + 1))


def _log10_with_eps(values, log_eps):
    values = np.asarray(values, dtype=np.float64)
    return np.log10(np.maximum(values, 0.0) + float(log_eps))


def _quantile_90(values):
    return values.quantile(0.90)


def _quantile_99(values):
    return values.quantile(0.99)


def _safe_relative_error(error, reference):
    reference = float(reference)
    if reference == 0.0:
        return float("nan")

    return float(error) / reference


def _resolve_device(device_name):
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    return device


def _set_random_seed(random_seed, deterministic):
    torch.manual_seed(int(random_seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(random_seed))

    torch.use_deterministic_algorithms(bool(deterministic))
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = bool(deterministic)
