"""Stage 4: build and place the full-volume multitask autoencoder.

Both raw ``(vx, vy, vz)`` VDFs and complete Hermite coefficient cubes use
the same Conv3d encoder and decoder. A compact latent vector feeds both the
full-volume reconstruction branch and a six-value auxiliary topology head.

One Python process owns the complete model and optimizer. Meaningful
consecutive encoder, bottleneck, decoder, and output stages can be placed on
several visible CUDA devices. Activations move only when the next stage has
a different owner; CPU and one-GPU execution use this same placement path.
"""

from math import prod

import torch
from torch import nn
from torch.nn import functional

from src.data.metadata_columns import TOPOLOGY_TARGET_COLUMNS


DEFAULT_CHANNELS = (8, 16, 32)
DEFAULT_BOTTLENECK_SHAPE = (4, 4, 4)


class VdfAutoencoder(nn.Module):
    """Reconstruct complete three-dimensional VDF representations.

    The autoencoder accepts either a full raw velocity distribution with
    saved axis order ``(vx, vy, vz)`` or a complete Hermite coefficient cube
    in unrotated ``(n_x, n_y, n_z)`` or rotated
    ``(n_parallel, n_perp1, n_perp2)`` order. Both representations are
    processed by Conv3d encoder and decoder blocks; training never extracts
    a velocity-space plane or downsamples the saved sample.

    The encoder spatially reduces the complete volume before adaptive pooling
    to the configured per-axis bottleneck maximum and projection to one
    ``(batch_size, latent_size)`` vector. The effective bottleneck never
    exceeds the encoded shape. The decoder expands that spatial bottleneck
    and reconstructs the normalized three-dimensional input, while an
    auxiliary head predicts six scaled Earth-radius distances and x-z
    displacement components to the nearest physical X and O points. Topology
    values supervise the latent space but are never model inputs.

    Runtime model parallelism places consecutive encoder, bottleneck,
    decoder, reconstruction, and topology stages across CUDA devices in one
    Python process. Activations move only when ownership changes. The
    reconstruction, topology predictions, and exposed latent embedding all
    finish on :attr:`output_device`.

    Parameters
    ----------
    representation : {"raw", "hermite"}
        Complete physical representation reconstructed by the model.
    input_shape : sequence of int
        Full spatial shape ``(depth_1, depth_2, depth_3)`` without batch or
        channel dimensions.
    input_mean : array-like
        Training-only per-voxel or per-coefficient mean.
    input_scale : array-like
        Training-only population standard deviation with ``input_shape``.
    channels : sequence of int, optional
        Ordered Conv3d encoder widths. The decoder reverses these widths.
    latent_size : int, optional
        Width of the shared reconstruction/topology embedding.
    pooling : str, optional
        Spatial reduction convention, currently average pooling.
    topology_hidden_size : int, optional
        Hidden width of the auxiliary topology head.
    topology_target_names : sequence of str, optional
        Fixed six-target order, with all values measured in Earth radii.
    bottleneck_shape : sequence of int, optional
        Maximum retained spatial cells along the three model axes after the
        encoder. Raw axes follow ``(vx, vy, vz)``; Hermite axes follow their
        saved coefficient order. The effective shape is capped by the encoded
        volume on each axis.

    Returns
    -------
    torch.nn.Module
        Conv3d autoencoder whose forward result contains complete
        reconstruction, six topology predictions, and latent embedding.

    Notes
    -----
    The target order is ``distance_to_x_point_re``,
    ``distance_to_o_point_re``, ``vdf_to_x_point_dx_re``,
    ``vdf_to_x_point_dz_re``, ``vdf_to_o_point_dx_re``, and
    ``vdf_to_o_point_dz_re``. Missing-value masks and training-only topology
    scaling belong to the loss workflow, not this input-only forward method.
    """

    def __init__(
        self,
        representation,
        input_shape,
        input_mean,
        input_scale,
        channels=DEFAULT_CHANNELS,
        latent_size=64,
        pooling="average",
        topology_hidden_size=64,
        topology_target_names=TOPOLOGY_TARGET_COLUMNS,
        bottleneck_shape=DEFAULT_BOTTLENECK_SHAPE,
    ):
        """Initialize the Conv3d encoder, bottleneck, decoder, and head."""

        super().__init__()
        self.representation = str(representation).strip().lower()
        self.input_shape = tuple(int(value) for value in input_shape)
        self.channels = tuple(int(value) for value in channels)
        self.latent_size = int(latent_size)
        self.pooling = str(pooling).strip().lower()
        self.topology_hidden_size = int(topology_hidden_size)
        self.topology_target_names = tuple(
            str(name) for name in topology_target_names
        )
        self.configured_bottleneck_shape = tuple(
            int(value) for value in bottleneck_shape
        )
        mean = torch.as_tensor(input_mean, dtype=torch.float32)
        scale = torch.as_tensor(input_scale, dtype=torch.float32)
        self.register_buffer(
            "input_mean",
            mean.reshape(1, 1, *self.input_shape),
        )
        self.register_buffer(
            "input_scale",
            scale.reshape(1, 1, *self.input_shape),
        )

        (
            self.encoder_blocks,
            self._pre_pool_shapes,
            self._pooled_stages,
            self.encoded_shape,
        ) = _create_encoder_blocks(self.input_shape, self.channels)
        self.bottleneck_shape = tuple(
            min(limit, size)
            for limit, size in zip(
                self.configured_bottleneck_shape,
                self.encoded_shape,
                strict=True,
            )
        )
        self.bottleneck_pool = nn.AdaptiveAvgPool3d(self.bottleneck_shape)
        bottleneck_count = int(
            self.channels[-1] * prod(self.bottleneck_shape)
        )
        self.to_latent = nn.Linear(bottleneck_count, self.latent_size)
        self.from_latent = nn.Linear(self.latent_size, bottleneck_count)
        self.decoder_blocks = _create_decoder_blocks(self.channels)
        self.reconstruction_head = nn.Conv3d(
            self.channels[0],
            1,
            kernel_size=3,
            padding=1,
        )
        self.topology_head = nn.Sequential(
            nn.Linear(self.latent_size, self.topology_hidden_size),
            nn.ReLU(),
            nn.Linear(
                self.topology_hidden_size,
                len(self.topology_target_names),
            ),
        )
        self._stage_names = _create_stage_names(len(self.channels))
        self._stage_devices = tuple(
            torch.device("cpu") for _ in self._stage_names
        )
        self.requested_model_parallel_gpus = 1
        self.effective_model_parallel_gpus = 1

    def forward(self, inputs):
        """Reconstruct a full volume and predict auxiliary topology.

        Parameters
        ----------
        inputs : torch.Tensor
            Unstandardized raw or Hermite samples with shape
            ``(batch_size, 1, depth_1, depth_2, depth_3)``.

        Returns
        -------
        dict
            ``reconstruction`` is the complete normalized input shape,
            ``topology_predictions`` has shape ``(batch_size, 6)``, and
            ``latent_embedding`` has shape ``(batch_size, latent_size)``.

        Notes
        -----
        The latent vector is not detached. Reconstruction and topology
        gradients therefore both reach the encoder. Topology targets,
        validity masks, and physical classes are never forward arguments.
        """

        return self.forward_from_normalized(self.normalize_inputs(inputs))

    def forward_from_normalized(self, normalized):
        """Run placed Conv3d stages for an already normalized batch.

        Training and evaluation normalize each complete volume once so the
        same tensor can serve as the reconstruction target. The public
        :meth:`forward` remains the input-only inference interface and calls
        this operation after applying its checkpointed normalization.

        Parameters
        ----------
        normalized : torch.Tensor
            Complete normalized batch on :attr:`input_device` with shape
            ``(batch_size, 1, depth_1, depth_2, depth_3)``.

        Returns
        -------
        dict
            Complete reconstruction, six topology predictions, and shared
            latent embedding, all on :attr:`output_device`.
        """

        encoded = normalized
        for block, stage_device in zip(
            self.encoder_blocks,
            self._stage_devices[: len(self.encoder_blocks)],
            strict=True,
        ):
            encoded = block(encoded.to(stage_device))

        bottleneck_index = len(self.encoder_blocks)
        bottleneck_device = self._stage_devices[bottleneck_index]
        pooled = self.bottleneck_pool(encoded.to(bottleneck_device))
        latent = self.to_latent(torch.flatten(pooled, start_dim=1))
        decoded = self.from_latent(latent).reshape(
            len(latent),
            self.channels[-1],
            *self.bottleneck_shape,
        )
        decoded = functional.interpolate(
            decoded,
            size=self.encoded_shape,
            mode="trilinear",
            align_corners=False,
        )

        decoder_start = bottleneck_index + 1
        stage_indices = range(len(self.channels) - 1, 0, -1)
        for offset, (stage_index, block) in enumerate(
            zip(stage_indices, self.decoder_blocks, strict=True)
        ):
            stage_device = self._stage_devices[decoder_start + offset]
            decoded = decoded.to(stage_device)
            if self._pooled_stages[stage_index]:
                decoded = functional.interpolate(
                    decoded,
                    size=self._pre_pool_shapes[stage_index],
                    mode="trilinear",
                    align_corners=False,
                )
            decoded = block(decoded)

        decoded = decoded.to(self.output_device)
        if self._pooled_stages[0]:
            decoded = functional.interpolate(
                decoded,
                size=self._pre_pool_shapes[0],
                mode="trilinear",
                align_corners=False,
            )
        reconstruction = self.reconstruction_head(decoded)
        latent_embedding = latent.to(self.output_device)
        topology_predictions = self.topology_head(latent_embedding)
        return {
            "reconstruction": reconstruction,
            "topology_predictions": topology_predictions,
            "latent_embedding": latent_embedding,
        }

    def normalize_inputs(self, inputs):
        """Apply stored training-only full-volume normalization once.

        Parameters
        ----------
        inputs : torch.Tensor
            Complete unstandardized raw or Hermite batch.

        Returns
        -------
        torch.Tensor
            Float32 normalized batch on :attr:`input_device`.
        """

        inputs = inputs.to(
            device=self.input_device,
            dtype=self.input_mean.dtype,
        )
        return (inputs - self.input_mean) / self.input_scale

    def denormalize_reconstruction(self, reconstruction):
        """Return complete reconstructions to representation units.

        Parameters
        ----------
        reconstruction : torch.Tensor
            Normalized complete reconstruction on any runtime device.

        Returns
        -------
        torch.Tensor
            Raw logarithmic volume or signed Hermite cube with the same
            shape and device as ``reconstruction``.
        """

        scale = self.input_scale.to(reconstruction.device)
        mean = self.input_mean.to(reconstruction.device)
        return reconstruction * scale + mean

    @property
    def input_device(self):
        """Return the device owning normalization and the first encoder.

        Returns
        -------
        torch.device
            First occupied model-stage device.
        """

        return self.input_mean.device

    @property
    def output_device(self):
        """Return the device owning reconstruction and topology outputs.

        Returns
        -------
        torch.device
            Final occupied model-stage device.
        """

        return self.reconstruction_head.weight.device

    @property
    def stage_names(self):
        """Return meaningful stages in exact forward order.

        Returns
        -------
        tuple of str
            Encoder, bottleneck, decoder, and combined output stage names.
        """

        return self._stage_names

    @property
    def stage_devices(self):
        """Return runtime owners aligned with :attr:`stage_names`.

        Returns
        -------
        tuple of torch.device
            Device for every meaningful consecutive forward stage.
        """

        return self._stage_devices

    def place_model_parallel(self, device, model_parallel_gpus=1):
        """Place consecutive autoencoder stages across CUDA devices.

        One Python process owns the complete Conv3d model and one optimizer.
        Input normalization and early encoder blocks begin on the first
        device, the bottleneck and later stages continue across the requested
        visible devices, and decoder/output stages finish on the final
        device. Activations move only when ownership changes between
        consecutive stages.

        The reconstruction output, exposed latent vector, and auxiliary
        topology output are produced on the final device. Runtime CUDA
        identifiers and stage mapping are not checkpoint architecture and
        are reconstructed whenever a model is loaded.

        Parameters
        ----------
        device : str or torch.device
            CPU or first CUDA device for runtime placement.
        model_parallel_gpus : int, optional
            Requested visible CUDA-device count. CPU always occupies one
            device; CUDA use is capped by visible devices and meaningful
            stages.

        Returns
        -------
        VdfAutoencoder
            This same model with consecutive stage ownership configured.

        Notes
        -----
        This is layer model parallelism rather than data parallelism. Each
        Conv3d stage and its local backward activations must fit on its owner.
        A single optimizer owns parameters across all occupied devices.
        """

        device = torch.device(device)
        requested = int(model_parallel_gpus)
        if device.type == "cuda":
            effective = min(
                requested,
                torch.cuda.device_count(),
                len(self._stage_names),
            )
            devices = (
                (device,)
                if effective == 1
                else tuple(
                    torch.device(f"cuda:{index}")
                    for index in range(effective)
                )
            )
        else:
            effective = 1
            devices = (device,)
        assignments = calculate_contiguous_stage_assignment(
            len(self._stage_names),
            effective,
        )
        stage_devices = tuple(devices[index] for index in assignments)

        self._buffers["input_mean"] = self.input_mean.to(stage_devices[0])
        self._buffers["input_scale"] = self.input_scale.to(stage_devices[0])
        for block, stage_device in zip(
            self.encoder_blocks,
            stage_devices[: len(self.encoder_blocks)],
            strict=True,
        ):
            block.to(stage_device)
        bottleneck_index = len(self.encoder_blocks)
        bottleneck_device = stage_devices[bottleneck_index]
        self.bottleneck_pool.to(bottleneck_device)
        self.to_latent.to(bottleneck_device)
        self.from_latent.to(bottleneck_device)
        decoder_start = bottleneck_index + 1
        for offset, block in enumerate(self.decoder_blocks):
            block.to(stage_devices[decoder_start + offset])
        self.reconstruction_head.to(stage_devices[-1])
        self.topology_head.to(stage_devices[-1])
        self._stage_devices = stage_devices
        self.requested_model_parallel_gpus = requested
        self.effective_model_parallel_gpus = effective
        return self

    def constructor_config(self):
        """Return device-independent checkpoint architecture fields.

        Primitive values are separated from runtime placement so the same
        state dictionary can reconstruct meaningful stages on CPU or a new
        visible CUDA-device count.

        Returns
        -------
        dict
            Representation, complete input shape, Conv3d widths, latent
            size, configured bottleneck maximum, pooling, topology head
            width, and fixed target order.

        Notes
        -----
        Runtime CUDA identifiers, stage devices, and requested GPU count are
        intentionally absent so one checkpoint loads on CPU or another GPU
        count.
        """

        return {
            "representation": self.representation,
            "input_shape": list(self.input_shape),
            "channels": list(self.channels),
            "latent_size": self.latent_size,
            "bottleneck_shape": list(
                self.configured_bottleneck_shape
            ),
            "pooling": self.pooling,
            "topology_hidden_size": self.topology_hidden_size,
            "topology_target_names": list(self.topology_target_names),
        }


def build_autoencoder(data, input_scaler, model_config, topology_config):
    """Construct the shared Conv3d raw/Hermite autoencoder.

    This factory joins the memory-mapped representation description,
    training-only input normalization, common Conv3d widths, and auxiliary
    head width without selecting a representation-specific convolution family.

    Parameters
    ----------
    data : AutoencoderTrainingData
        On-demand memory-mapped representation and three-dimensional shape.
    input_scaler : InputFeatureScaler
        Per-voxel or per-coefficient normalization fitted from training rows.
    model_config : mapping
        Conv3d channels, latent size, per-axis bottleneck maximum, and pooling
        convention.
    topology_config : mapping
        Auxiliary hidden width and loss weight. Only the hidden width affects
        architecture; the weight belongs to the combined loss.

    Returns
    -------
    VdfAutoencoder
        Untrained model on its default CPU stage placement.
    """

    return VdfAutoencoder(
        representation=data.representation,
        input_shape=data.input_shape,
        input_mean=input_scaler.mean,
        input_scale=input_scaler.scale,
        channels=model_config["channels"],
        latent_size=model_config["latent_size"],
        bottleneck_shape=model_config["bottleneck_shape"],
        pooling=model_config["pooling"],
        topology_hidden_size=topology_config["hidden_size"],
        topology_target_names=TOPOLOGY_TARGET_COLUMNS,
    )


def calculate_contiguous_stage_assignment(stage_count, device_count):
    """Map meaningful consecutive stages to consecutive devices.

    Parameters
    ----------
    stage_count : int
        Number of nonempty stages in exact forward order.
    device_count : int
        Effective occupied device count after runtime capping.

    Returns
    -------
    tuple of int
        Device index for each stage using
        ``stage_index * device_count // stage_count``.
    """

    stage_count = int(stage_count)
    device_count = min(int(device_count), stage_count)
    return tuple(
        stage_index * device_count // stage_count
        for stage_index in range(stage_count)
    )


def _create_encoder_blocks(input_shape, channels):
    """Create Conv3d encoder blocks and record exact decoder geometry.

    Parameters
    ----------
    input_shape : sequence of int
        Complete three-dimensional raw or Hermite spatial shape.
    channels : sequence of int
        Successive Conv3d output widths.

    Returns
    -------
    blocks : torch.nn.ModuleList
        Consecutive Conv3d, ReLU, and optional AvgPool3d blocks.
    pre_pool_shapes : tuple of tuple
        Spatial shape before each possible reduction.
    pooled_stages : tuple of bool
        Whether each block reduced every axis by two.
    encoded_shape : tuple of int
        Spatial shape passed to adaptive bottleneck pooling.
    """

    blocks = []
    pre_pool_shapes = []
    pooled_stages = []
    current_channels = 1
    current_shape = tuple(int(value) for value in input_shape)
    for output_channels in channels:
        layers = [
            nn.Conv3d(
                current_channels,
                output_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
        ]
        pre_pool_shapes.append(current_shape)
        should_pool = all(size >= 2 for size in current_shape)
        pooled_stages.append(should_pool)
        if should_pool:
            layers.append(nn.AvgPool3d(kernel_size=2))
            current_shape = tuple(size // 2 for size in current_shape)
        blocks.append(nn.Sequential(*layers))
        current_channels = output_channels
    return (
        nn.ModuleList(blocks),
        tuple(pre_pool_shapes),
        tuple(pooled_stages),
        current_shape,
    )


def _create_decoder_blocks(channels):
    """Create reversed Conv3d decoder blocks before reconstruction output.

    Parameters
    ----------
    channels : sequence of int
        Encoder widths in forward order.

    Returns
    -------
    torch.nn.ModuleList
        Deep-to-shallow Conv3d/ReLU blocks. The separate reconstruction head
        converts the final shallow activation to one channel.
    """

    return nn.ModuleList(
        nn.Sequential(
            nn.Conv3d(
                channels[stage_index],
                channels[stage_index - 1],
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
        )
        for stage_index in range(len(channels) - 1, 0, -1)
    )


def _create_stage_names(channel_count):
    """Name nonempty model stages in exact forward order.

    Parameters
    ----------
    channel_count : int
        Number of encoder widths and blocks.

    Returns
    -------
    tuple of str
        Encoder stages, bottleneck, reversed decoder stages, and output.
    """

    return (
        *(f"encoder_{index}" for index in range(int(channel_count))),
        "bottleneck",
        *(
            f"decoder_{index}"
            for index in range(int(channel_count) - 1, 0, -1)
        ),
        "reconstruction_and_topology",
    )


__all__ = [
    "VdfAutoencoder",
    "build_autoencoder",
    "calculate_contiguous_stage_assignment",
]
