"""Stage 4: construct the raw or Hermite multitask CNN.

This stage follows training-derived scaling and precedes loss calculation.
It builds the representation-specific encoder, a shared learned projection,
physical-class logits, and the six scaled auxiliary topology outputs used by
training and prediction. A separate dense branch embeds the 16-value plasma
context on the output stage before both heads.

The stage receives the representation shape, class mapping,
training-derived input normalization, and model configuration. It returns
one newly initialized :class:`VdfCNN`; stage 5 defines the loss applied to
its two output heads.
"""

from math import prod

import torch
from torch import nn

from src.data.metadata_columns import TOPOLOGY_TARGET_COLUMNS
from src.physics.plasma_context import PLASMA_CONTEXT_FEATURE_NAMES


TOPOLOGY_TARGET_NAMES = TOPOLOGY_TARGET_COLUMNS
DEFAULT_RAW_ENCODER_CONFIG = {
    "channels": (4, 8, 16),
    "adaptive_pool_shape": (2, 2, 2),
}
DEFAULT_HERMITE_ENCODER_CONFIG = {
    "channels": (16, 32, 64),
    "adaptive_pool_shape": (2, 2, 2),
}


class VdfCNN(nn.Module):
    """Predict class and topology from a VDF plus plasma context.

    Stage 4 constructs this multitask network after all preprocessing
    statistics have been fitted from training rows. Separate 3-D encoders
    consume either the complete raw ``(vx, vy, vz)`` VDF or the complete
    Hermite coefficient volume. A small dense branch separately consumes the
    same-cell 16-value plasma context: ``(Bx, By, Bz)`` in teslas,
    ``(Ex, Ey, Ez)`` in volts per metre, configured-population fluid velocity
    ``(Vx, Vy, Vz)`` in metres per second, number density in particles per
    cubic metre, and pressure ``(Pxx, Pyy, Pzz, Pxy, Pxz, Pyz)`` in pascals.
    Training-only scaling is applied per component, and no redundant vector
    magnitude is included.

    Parameters
    ----------
    representation : {"raw", "hermite"}
        Raw tensors use shape ``(batch, 1, vx, vy, vz)``; Hermite tensors
        use ``(batch, 1, *saved_coefficient_shape)`` in the extraction
        frame's three Hermite-order axes.
    input_shape : sequence of int
        Spatial shape without batch and channel axes.
    class_ids : sequence of int
        Physical project class IDs in model-output order.
    class_names : sequence of str
        Physical names aligned with ``class_ids``.
    input_mean : array-like
        Training-only mean per pixel or coefficient.
    input_scale : array-like
        Training-only scale per pixel or coefficient.
    topology_target_names : sequence of str, optional
        Six topology targets in saved schema order.
    raw_encoder_config : mapping, optional
        Raw convolution channels and adaptive-pooling shape.
    hermite_encoder_config : mapping, optional
        Hermite convolution channels and adaptive-pooling shape.
    shared_hidden_size : int, optional
        Width of the VDF convolutional embedding.
    plasma_context_hidden_size : int, optional
        Width of the dense branch applied to the 16 scaled context values.
    dropout : float, optional
        Dropout probability before both output heads.

    Attributes
    ----------
    input_mean, input_scale : torch.Tensor
        Non-trainable feature normalization buffers with leading batch and
        channel axes.
    class_head : torch.nn.Linear
        Output layer producing ``n_classes`` unnormalized class logits.
    topology_head : torch.nn.Linear
        Output layer producing six scaled topology estimates.

    Notes
    -----
    The model normalizes VDF tensors internally. Plasma context is a model
    input, while topology values remain auxiliary targets and are never
    supplied to the network. PyTorch DataLoader workers only read samples;
    ``model_parallel_gpus`` places consecutive Conv3d and output stages in
    one Python process, and dataset ``extraction_n_jobs`` is unrelated
    timestep parallelism. The context branch and concatenation live on the
    output device, so only the small ``(batch, 16)`` tensor crosses directly
    to that device. Each Conv3d block remains indivisible and must fit on its
    owning GPU.
    """

    def __init__(
        self,
        representation,
        input_shape,
        class_ids,
        class_names,
        input_mean,
        input_scale,
        topology_target_names=TOPOLOGY_TARGET_NAMES,
        raw_encoder_config=None,
        hermite_encoder_config=None,
        shared_hidden_size=64,
        plasma_context_hidden_size=32,
        dropout=0.2,
    ):
        """Initialize representation-specific encoders and output heads."""

        super().__init__()
        self.representation = str(representation).strip().lower()
        self.input_shape = tuple(int(value) for value in input_shape)
        self.class_ids = tuple(int(value) for value in class_ids)
        self.class_names = tuple(str(value) for value in class_names)
        self.project_class_id_to_output_index = {
            class_id: output_index
            for output_index, class_id in enumerate(self.class_ids)
        }
        self.output_index_to_project_class_id = tuple(self.class_ids)
        self.topology_target_names = tuple(
            str(name) for name in topology_target_names
        )
        self.shared_hidden_size = int(shared_hidden_size)
        self.plasma_context_hidden_size = int(
            plasma_context_hidden_size
        )
        self.combined_embedding_size = (
            self.shared_hidden_size + self.plasma_context_hidden_size
        )
        self.dropout_probability = float(dropout)
        self.raw_encoder_config = _encoder_config(
            raw_encoder_config,
            DEFAULT_RAW_ENCODER_CONFIG,
        )
        self.hermite_encoder_config = _encoder_config(
            hermite_encoder_config,
            DEFAULT_HERMITE_ENCODER_CONFIG,
        )
        encoder_config = (
            self.raw_encoder_config
            if self.representation == "raw"
            else self.hermite_encoder_config
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

        channels = encoder_config["channels"]
        adaptive_pool_shape = encoder_config["adaptive_pool_shape"]
        self.encoder = _create_encoder(
            input_shape=self.input_shape,
            channels=channels,
            adaptive_pool_shape=adaptive_pool_shape,
        )
        self._encoder_block_slices = _find_encoder_block_slices(self.encoder)
        encoded_size = int(channels[-1] * prod(adaptive_pool_shape))
        self.shared_projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(encoded_size, self.shared_hidden_size),
            nn.ReLU(),
        )
        self.plasma_context_projection = nn.Sequential(
            nn.Linear(
                len(PLASMA_CONTEXT_FEATURE_NAMES),
                self.plasma_context_hidden_size,
            ),
            nn.ReLU(),
        )
        self.output_dropout = nn.Dropout(self.dropout_probability)
        self.class_head = nn.Linear(
            self.combined_embedding_size,
            len(self.class_ids),
        )
        self.topology_head = nn.Linear(
            self.combined_embedding_size,
            len(self.topology_target_names),
        )

    def forward(self, inputs, plasma_context):
        """Predict outputs from a complete VDF and 16 plasma features.

        Stage 5 consumes both output heads for training; evaluation and
        prediction apply softmax or inverse topology scaling afterward.
        The shared embedding is returned for inspection but is not another
        supervised target.

        Parameters
        ----------
        inputs : torch.Tensor
            Raw ``(batch, 1, vx, vy, vz)`` tensor or Hermite
            ``(batch, 1, *saved_coefficient_shape)`` tensor.
        plasma_context : torch.Tensor
            Training-scaled context with shape ``(batch, 16)`` in
            ``PLASMA_CONTEXT_FEATURE_NAMES`` order. All Cartesian B, E, and
            configured-population fluid-velocity components are retained.

        Returns
        -------
        dict
            Class logits ``(batch, n_classes)``, scaled topology values
            ``(batch, 6)``, and combined ``embedding`` with shape
            ``(batch, shared_hidden_size + plasma_context_hidden_size)``.
        """

        encoded = self.normalize_inputs(inputs)
        for block_slice, block_device in zip(
            self._encoder_block_slices,
            self.encoder_block_devices,
            strict=True,
        ):
            encoded = encoded.to(block_device)
            for layer_index in range(block_slice.start, block_slice.stop):
                encoded = self.encoder[layer_index](encoded)
        vdf_embedding = self.shared_projection(
            encoded.to(self.output_device)
        )
        context_embedding = self.plasma_context_projection(
            plasma_context.to(
                device=self.output_device,
                dtype=vdf_embedding.dtype,
            )
        )
        embedding = torch.cat(
            (vdf_embedding, context_embedding),
            dim=1,
        )
        head_input = self.output_dropout(embedding)
        return {
            "class_logits": self.class_head(head_input),
            "topology_predictions": self.topology_head(head_input),
            "embedding": embedding,
        }

    def normalize_inputs(self, inputs):
        """Apply the stored training-only feature normalization.

        The forward path calls this once before the convolutional encoder.
        Casting to the buffer dtype keeps raw and Hermite preprocessing
        consistent with the float32 statistics saved in the checkpoint.

        Parameters
        ----------
        inputs : torch.Tensor
            Unstandardized raw or Hermite tensor.

        Returns
        -------
        torch.Tensor
            Float32 tensor normalized once per spatial feature.
        """

        inputs = inputs.to(
            device=self.input_device,
            dtype=self.input_mean.dtype,
        )
        return (inputs - self.input_mean) / self.input_scale

    @property
    def input_device(self):
        """Return the device that owns normalization and model input.

        Returns
        -------
        torch.device
            First occupied stage device.
        """

        return self.input_mean.device

    @property
    def encoder_block_devices(self):
        """Return devices hosting the existing flat encoder blocks.

        Returns
        -------
        tuple of torch.device
            Device of the leading convolution in each contiguous block.
        """

        return tuple(
            self.encoder[block_slice.start].weight.device
            for block_slice in self._encoder_block_slices
        )

    @property
    def output_device(self):
        """Return the device hosting projection and both output heads.

        Returns
        -------
        torch.device
            Final occupied stage device.
        """

        return self.class_head.weight.device

    def place_model_parallel(self, device, model_parallel_gpus=1):
        """Place contiguous encoder stages on consecutive devices.

        Runtime placement moves the existing flat encoder modules rather
        than registering a second model structure. This preserves scientific
        layer mathematics and checkpoint keys while distributing parameters
        and forward activations across the occupied devices.

        Parameters
        ----------
        device : str or torch.device
            Runtime device used by one-device placement. Multi-GPU placement
            uses visible ``cuda:0`` through the final occupied stage device.
        model_parallel_gpus : int, optional
            Requested number of consecutive devices. The effective count is
            capped at the number of encoder blocks plus the shared output
            stage.

        Returns
        -------
        VdfCNN
            This model after in-place runtime placement.

        Notes
        -----
        Existing modules remain registered only under ``encoder``,
        ``shared_projection``, ``plasma_context_projection``, and the two
        output-head names. Runtime placement therefore does not alter
        checkpoint state-dictionary keys. Encoder blocks and the shared
        fused output path form sequential stages.
        Their balanced, contiguous device indices are calculated as
        ``stage_index * effective_device_count // stage_count``.
        Normalization uses the first occupied device and every returned tensor
        uses the final occupied device.
        """

        stage_count = len(self._encoder_block_slices) + 1
        stage_device_indices = calculate_contiguous_stage_assignment(
            stage_count,
            model_parallel_gpus,
        )
        effective_device_count = min(
            int(model_parallel_gpus),
            stage_count,
        )
        devices = _consecutive_devices(device, effective_device_count)
        self.input_mean = self.input_mean.to(
            devices[stage_device_indices[0]]
        )
        self.input_scale = self.input_scale.to(
            devices[stage_device_indices[0]]
        )
        for block_index, block_slice in enumerate(
            self._encoder_block_slices
        ):
            block_device = devices[stage_device_indices[block_index]]
            for layer_index in range(block_slice.start, block_slice.stop):
                self.encoder[layer_index].to(block_device)
        output_device = devices[stage_device_indices[-1]]
        self.shared_projection.to(output_device)
        self.plasma_context_projection.to(output_device)
        self.output_dropout.to(output_device)
        self.class_head.to(output_device)
        self.topology_head.to(output_device)
        return self

    def class_id_to_output_index(self, class_id):
        """Return the model index for one physical class ID.

        This mapping lets external callers address the correct logit column
        without assuming project class IDs are contiguous.

        Parameters
        ----------
        class_id : int
            Project physical class ID.

        Returns
        -------
        int
            Contiguous class-logit index.
        """

        return self.project_class_id_to_output_index[int(class_id)]

    def output_index_to_class_id(self, output_index):
        """Return the physical class ID for one model index.

        Evaluation and prediction use this inverse relation after selecting
        a class-logit column.

        Parameters
        ----------
        output_index : int
            Contiguous class-logit index.

        Returns
        -------
        int
            Project physical class ID.
        """

        return self.output_index_to_project_class_id[int(output_index)]

    def constructor_config(self):
        """Return arguments needed to reconstruct this CNN.

        Stage 8 saves this primitive architecture record. Checkpoint loading
        passes it directly back to the constructor before restoring learned
        tensors.

        Returns
        -------
        dict
            Architecture, class mapping, normalization, and topology order.
        """

        return {
            "representation": self.representation,
            "input_shape": list(self.input_shape),
            "class_ids": list(self.class_ids),
            "class_names": list(self.class_names),
            "input_mean": self.input_mean[0, 0].detach().cpu().tolist(),
            "input_scale": self.input_scale[0, 0].detach().cpu().tolist(),
            "topology_target_names": list(self.topology_target_names),
            "raw_encoder_config": _json_encoder_config(
                self.raw_encoder_config
            ),
            "hermite_encoder_config": _json_encoder_config(
                self.hermite_encoder_config
            ),
            "shared_hidden_size": self.shared_hidden_size,
            "plasma_context_hidden_size": (
                self.plasma_context_hidden_size
            ),
            "dropout": self.dropout_probability,
        }


def build_cnn(data, input_scaler, config):
    """Construct the configured CNN for loaded training data.

    This stage-level factory connects the stage-1 representation/class
    description and stage-3 normalization to the representation-specific
    architecture. The returned model is untrained and remains on its
    default device until the orchestrator moves it.

    Parameters
    ----------
    data : CnnTrainingData
        Loaded representation shape and physical class mapping.
    input_scaler : InputFeatureScaler
        Training-only spatial normalization.
    config : mapping
        CNN model configuration.

    Returns
    -------
    VdfCNN
        Newly initialized raw or Hermite multitask model.
    """

    model_config = config["model"]
    return VdfCNN(
        representation=data.representation,
        input_shape=data.input_shape,
        class_ids=data.class_mapping.class_ids,
        class_names=data.class_mapping.class_names,
        input_mean=input_scaler.mean,
        input_scale=input_scaler.scale,
        topology_target_names=TOPOLOGY_TARGET_COLUMNS,
        raw_encoder_config={
            "channels": tuple(model_config["raw_channels"]),
            "adaptive_pool_shape": tuple(
                model_config["raw_adaptive_pool_shape"]
            ),
        },
        hermite_encoder_config={
            "channels": tuple(model_config["hermite_channels"]),
            "adaptive_pool_shape": tuple(
                model_config["adaptive_pool_shape_3d"]
            ),
        },
        shared_hidden_size=model_config["shared_hidden_size"],
        plasma_context_hidden_size=config["plasma_context"]["hidden_size"],
        dropout=model_config["dropout"],
    )


def calculate_contiguous_stage_assignment(
    stage_count,
    requested_device_count,
):
    """Calculate balanced contiguous device indices for sequential stages.

    The pure calculation groups adjacent stages on adjacent device indices
    and caps the occupied count at the number of meaningful stages. It is
    shared by model construction and structural tests without touching CUDA.

    Parameters
    ----------
    stage_count : int
        Number of meaningful sequential model stages.
    requested_device_count : int
        Requested number of consecutive runtime devices.

    Returns
    -------
    tuple of int
        Device index for each stage. The effective device count is capped at
        ``stage_count`` and assignment follows
        ``stage_index * effective_device_count // stage_count``.
    """

    stage_count = int(stage_count)
    effective_device_count = min(
        int(requested_device_count),
        stage_count,
    )
    return tuple(
        stage_index * effective_device_count // stage_count
        for stage_index in range(stage_count)
    )


def _create_encoder(
    *,
    input_shape,
    channels,
    adaptive_pool_shape,
):
    """Build a complete-volume convolution and pooling stack.

    Parameters
    ----------
    input_shape : sequence of int
        Raw ``(vx, vy, vz)`` or Hermite coefficient shape.
    channels : sequence of int
        Output channel width for each convolutional block.
    adaptive_pool_shape : sequence of int
        Final spatial shape consumed by the shared linear projection.

    Returns
    -------
    torch.nn.Sequential
        Convolution/ReLU blocks with shape-safe average pooling followed by
        adaptive pooling.

    Notes
    -----
    A factor-of-two pool is inserted only when every resulting spatial axis
    remains at least as large as its adaptive-pooling target. This preserves
    the established encoder geometry for small synthetic inputs.
    """

    layers = []
    current_channels = 1
    current_shape = tuple(input_shape)

    for output_channels in channels:
        layers.extend(
            (
                nn.Conv3d(
                    current_channels,
                    output_channels,
                    kernel_size=3,
                    padding=1,
                ),
                nn.ReLU(),
            )
        )
        pooled_shape = tuple(size // 2 for size in current_shape)
        if all(
            pooled >= target
            for pooled, target in zip(
                pooled_shape,
                adaptive_pool_shape,
                strict=True,
            )
        ):
            layers.append(nn.AvgPool3d(kernel_size=2))
            current_shape = pooled_shape
        current_channels = output_channels
    layers.append(nn.AdaptiveAvgPool3d(adaptive_pool_shape))
    return nn.Sequential(*layers)


def _find_encoder_block_slices(encoder):
    """Divide the flat encoder into consecutive placement stages.

    Each existing Conv3d starts one block that also owns its following
    activation and pooling layers up to the next convolution. Keeping the
    registered encoder flat preserves checkpoint parameter names while the
    returned slices let runtime placement and forward execution transfer an
    activation only at meaningful block boundaries.

    Parameters
    ----------
    encoder : torch.nn.Sequential
        Existing convolution, activation, pooling, and adaptive-pooling
        sequence.

    Returns
    -------
    tuple of slice
        Consecutive encoder index intervals in forward order.
    """

    starts = tuple(
        index
        for index, layer in enumerate(encoder)
        if isinstance(layer, nn.Conv3d)
    )
    stops = (*starts[1:], len(encoder))
    return tuple(slice(start, stop) for start, stop in zip(starts, stops))


def _consecutive_devices(device, device_count):
    """Create the visible-device sequence used by model stages.

    Slurm constrains physical devices through ``CUDA_VISIBLE_DEVICES``, so
    multi-GPU placement always uses process-local ``cuda:0`` through the
    requested final visible index. A single requested device preserves the
    original device object, including an unindexed ``cuda`` or CPU device
    used by established workflows.

    Parameters
    ----------
    device : str or torch.device
        Runtime device used unchanged by the single-device workflow.
    device_count : int
        Number of consecutive visible devices occupied by meaningful model
        stages.

    Returns
    -------
    tuple of torch.device
        Ordered devices used by contiguous model stages.
    """

    first_device = torch.device(device)
    if int(device_count) == 1:
        return (first_device,)
    return tuple(
        torch.device("cuda", offset)
        for offset in range(int(device_count))
    )


def _encoder_config(config, defaults):
    """Return convolution channels and adaptive-pooling dimensions.

    Parameters
    ----------
    config : mapping or None
        Optional representation-specific model settings.
    defaults : mapping
        Established channels and adaptive-pooling shape.

    Returns
    -------
    dict
        Tuple-valued encoder settings used by model construction.
    """

    source = defaults if config is None else config
    return {
        "channels": tuple(source.get("channels", defaults["channels"])),
        "adaptive_pool_shape": tuple(
            source.get(
                "adaptive_pool_shape",
                defaults["adaptive_pool_shape"],
            )
        ),
    }


def _json_encoder_config(config):
    """Convert tuple-valued encoder settings to checkpoint-safe lists.

    Parameters
    ----------
    config : mapping
        Encoder channels and adaptive-pooling shape.

    Returns
    -------
    dict
        Equivalent list-valued settings for checkpoint serialization.
    """

    return {
        "channels": list(config["channels"]),
        "adaptive_pool_shape": list(config["adaptive_pool_shape"]),
    }


__all__ = [
    "TOPOLOGY_TARGET_NAMES",
    "VdfCNN",
    "build_cnn",
    "calculate_contiguous_stage_assignment",
]
