"""Stage 4: construct the raw or Hermite convolutional autoencoder.

This stage follows training-derived normalization and precedes optimization.
It builds the fixed encoder, non-variational latent projection, and
interpolation/convolution decoder that reconstructs the original spatial
shape in normalized representation space.

The stage receives the representation shape, stage-3 normalization, and
model configuration. It returns a newly initialized
:class:`VdfAutoencoder`; stage 5 optimizes its mean-squared reconstruction
loss.
"""

from math import prod

import torch
from torch import nn
from torch.nn import functional


DEFAULT_RAW_CHANNELS = (8, 16, 32)
DEFAULT_HERMITE_CHANNELS = (8, 16, 32)


class VdfAutoencoder(nn.Module):
    """Reconstruct one normalized raw or Hermite representation.

    Stage 4 constructs this non-variational convolutional autoencoder after
    fitting feature normalization from training rows. The 2-D raw path
    reconstructs a full-resolution ``vx-vz`` log plane, while the 3-D
    Hermite path reconstructs the complete coefficient volume. Neither
    labels nor topology values enter the encoder.

    Parameters
    ----------
    representation : {"raw", "hermite"}
        Raw tensors have shape ``(batch, 1, vx, vz)`` and Hermite tensors
        have ``(batch, 1, *saved_coefficient_shape)`` in the extraction
        frame's three Hermite-order axes.
    input_shape : sequence of int
        Spatial input shape without batch or channel axes.
    input_mean : array-like
        Training-only mean per pixel or coefficient.
    input_scale : array-like
        Training-only scale per pixel or coefficient.
    raw_channels : sequence of int, optional
        Raw convolution channel widths.
    hermite_channels : sequence of int, optional
        Hermite convolution channel widths.
    latent_dim : int, optional
        Learned latent-vector width.
    pooling : str, optional
        Pooling convention, currently ``"average"``.

    Notes
    -----
    Reconstruction and mean-squared loss are calculated in
    training-normalized representation space. The decoder output can be
    converted back to raw log values or Hermite coefficients with
    :meth:`denormalize_reconstruction`.

    The latent vector is a learned reconstruction embedding, not a
    probability distribution or generative model.
    """

    def __init__(
        self,
        representation,
        input_shape,
        input_mean,
        input_scale,
        raw_channels=DEFAULT_RAW_CHANNELS,
        hermite_channels=DEFAULT_HERMITE_CHANNELS,
        latent_dim=64,
        pooling="average",
    ):
        """Initialize the encoder, latent projection, and matching decoder."""

        super().__init__()
        self.representation = str(representation).strip().lower()
        self.input_shape = tuple(int(value) for value in input_shape)
        self.raw_channels = tuple(int(value) for value in raw_channels)
        self.hermite_channels = tuple(
            int(value) for value in hermite_channels
        )
        self.latent_dim = int(latent_dim)
        self.pooling = str(pooling).strip().lower()
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

        channels = (
            self.raw_channels
            if self.representation == "raw"
            else self.hermite_channels
        )
        (
            self.encoder,
            self._pre_pool_shapes,
            self._pooled_stages,
            encoded_shape,
        ) = _create_encoder(
            self.representation,
            self.input_shape,
            channels,
        )
        self.encoded_shape = encoded_shape
        encoded_count = int(channels[-1] * prod(encoded_shape))
        self.to_latent = nn.Linear(encoded_count, self.latent_dim)
        self.from_latent = nn.Linear(self.latent_dim, encoded_count)
        self.decoder_layers = _create_decoder_layers(
            self.representation,
            channels,
        )

    def forward(self, inputs):
        """Reconstruct a batch in normalized feature space.

        Stage 5 passes unnormalized representation tensors to this method.
        The model applies its stored training-only normalization once,
        encodes the result, and returns the normalized reconstruction and
        deterministic latent vector.

        Parameters
        ----------
        inputs : torch.Tensor
            Unstandardized one-channel raw or Hermite batch.

        Returns
        -------
        dict
            ``reconstruction`` has the input shape in normalized space and
            ``latent`` has shape ``(batch, latent_dim)``.
        """

        return self._forward_normalized(self.normalize_inputs(inputs))

    def normalize_inputs(self, inputs):
        """Apply stored training-only normalization exactly once.

        Training and evaluation use this operation before MSE calculation.
        Casting to the buffer dtype keeps preprocessing consistent with the
        float32 statistics stored in the checkpoint.

        Parameters
        ----------
        inputs : torch.Tensor
            Unstandardized one-channel representation batch.

        Returns
        -------
        torch.Tensor
            Float32 batch in training-normalized feature space.
        """

        if inputs.dtype != self.input_mean.dtype:
            inputs = inputs.to(dtype=self.input_mean.dtype)
        return (inputs - self.input_mean) / self.input_scale

    def denormalize_reconstruction(self, reconstruction):
        """Return a reconstruction to representation units.

        This inverse operation is available for external inspection after
        the model has reconstructed its normalized input. It does not alter
        the model or supplied tensor.

        Parameters
        ----------
        reconstruction : torch.Tensor
            Batch in training-normalized feature space.

        Returns
        -------
        torch.Tensor
            Raw log-plane or Hermite coefficient values.
        """

        return reconstruction * self.input_scale + self.input_mean

    def constructor_config(self):
        """Return architecture settings needed for reconstruction.

        Stage 7 stores these primitive settings. Checkpoint loading passes
        them back to the constructor before restoring trained tensors.

        Returns
        -------
        dict
            Representation, input shape, channels, latent width, and
            pooling convention.
        """

        return {
            "representation": self.representation,
            "input_shape": list(self.input_shape),
            "raw_channels": list(self.raw_channels),
            "hermite_channels": list(self.hermite_channels),
            "latent_dim": self.latent_dim,
            "pooling": self.pooling,
        }

    def _forward_normalized(self, normalized):
        """Encode and decode an already normalized representation batch.

        Parameters
        ----------
        normalized : torch.Tensor
            Raw ``(batch, 1, vx, vz)`` values or Hermite
            ``(batch, 1, *saved_coefficient_shape)`` values in
            training-normalized coefficient space.

        Returns
        -------
        dict
            ``reconstruction`` with the input shape and ``latent`` with
            shape ``(batch, latent_dim)``.

        Notes
        -----
        Decoder interpolation reverses only encoder stages that pooled and
        uses ``align_corners=False`` to preserve the established numerical
        reconstruction.
        """

        encoded = self.encoder(normalized)
        latent = self.to_latent(torch.flatten(encoded, start_dim=1))
        decoded = self.from_latent(latent).reshape(
            len(normalized),
            -1,
            *self.encoded_shape,
        )
        mode = "bilinear" if self.representation == "raw" else "trilinear"
        for stage_index in range(len(self.decoder_layers) - 1, -1, -1):
            if self._pooled_stages[stage_index]:
                decoded = functional.interpolate(
                    decoded,
                    size=self._pre_pool_shapes[stage_index],
                    mode=mode,
                    align_corners=False,
                )
            decoded = self.decoder_layers[stage_index](decoded)
            if stage_index:
                decoded = functional.relu(decoded)
        return {"reconstruction": decoded, "latent": latent}


def build_autoencoder(data, input_scaler, config):
    """Construct the configured autoencoder for loaded data.

    This stage-level factory joins the stage-1 representation description,
    stage-3 normalization, and model settings. The returned model is
    untrained and remains on its default device until the orchestrator
    moves it.

    Parameters
    ----------
    data : AutoencoderTrainingData
        Loaded representation identity and spatial shape.
    input_scaler : InputFeatureScaler
        Training-derived normalization.
    config : mapping
        Autoencoder model configuration.

    Returns
    -------
    VdfAutoencoder
        Newly initialized raw or Hermite reconstruction model.
    """

    return VdfAutoencoder(
        representation=data.representation,
        input_shape=data.input_shape,
        input_mean=input_scaler.mean,
        input_scale=input_scaler.scale,
        raw_channels=config["raw_channels"],
        hermite_channels=config["hermite_channels"],
        latent_dim=config["latent_dim"],
        pooling=config["pooling"],
    )


def _create_encoder(representation, input_shape, channels):
    """Build the convolutional encoder and record reversible pool geometry.

    Parameters
    ----------
    representation : {"raw", "hermite"}
        Select 2-D or 3-D convolution and average pooling.
    input_shape : sequence of int
        Raw ``(vx, vz)`` or Hermite coefficient shape.
    channels : sequence of int
        Output channel width for each convolutional block.

    Returns
    -------
    encoder : torch.nn.Sequential
        Convolution, ReLU, and applicable average-pooling operations.
    pre_pool_shapes : tuple of tuple
        Spatial shape before each possible pool operation.
    pooled_stages : tuple of bool
        Whether each encoder stage reduced its spatial shape.
    encoded_shape : tuple of int
        Final spatial shape entering the latent projection.

    Notes
    -----
    Pooling occurs only while every spatial axis has length at least two.
    Recording each decision lets the decoder restore the exact input shape.
    """

    if representation == "raw":
        convolution = nn.Conv2d
        pooling = nn.AvgPool2d
    else:
        convolution = nn.Conv3d
        pooling = nn.AvgPool3d
    layers = []
    pre_pool_shapes = []
    pooled_stages = []
    current_channels = 1
    current_shape = tuple(input_shape)
    for output_channels in channels:
        layers.extend(
            (
                convolution(
                    current_channels,
                    output_channels,
                    kernel_size=3,
                    padding=1,
                ),
                nn.ReLU(),
            )
        )
        pre_pool_shapes.append(current_shape)
        should_pool = all(size >= 2 for size in current_shape)
        pooled_stages.append(should_pool)
        if should_pool:
            layers.append(pooling(kernel_size=2))
            current_shape = tuple(size // 2 for size in current_shape)
        current_channels = output_channels
    return (
        nn.Sequential(*layers),
        tuple(pre_pool_shapes),
        tuple(pooled_stages),
        current_shape,
    )


def _create_decoder_layers(representation, channels):
    """Build decoder convolutions in reverse encoder-channel order.

    Parameters
    ----------
    representation : {"raw", "hermite"}
        Select 2-D or 3-D convolution layers.
    channels : sequence of int
        Encoder channel widths in forward order.

    Returns
    -------
    torch.nn.ModuleList
        Convolutions used from the deepest channel width back to the
        singleton reconstructed channel.
    """

    convolution = nn.Conv2d if representation == "raw" else nn.Conv3d
    layers = []
    for stage_index, input_channels in enumerate(channels):
        output_channels = channels[stage_index - 1] if stage_index else 1
        layers.append(
            convolution(
                input_channels,
                output_channels,
                kernel_size=3,
                padding=1,
            )
        )
    return nn.ModuleList(layers)


__all__ = ["VdfAutoencoder", "build_autoencoder"]
