"""Stage 3: fit autoencoder normalization from training samples.

This stage follows the timestep split and precedes model construction. It
binds the autoencoder's normalization batch size to the shared stable
float64 accumulation and returns one mean and scale per reconstructed pixel
or coefficient.

The stage receives sample-wise representation data, training row indices,
and loader settings. It returns the shared :class:`InputFeatureScaler`
consumed by model construction and saved in the autoencoder checkpoint.
"""

from src.cnn.step_03_scale_cnn_inputs import (
    DEFAULT_SCALER_EPSILON,
    fit_input_scaler,
)


def scale_autoencoder_inputs(data, train_indices, config):
    """Fit per-feature normalization from training rows only.

    This adapter delegates to the same bounded float64 accumulation used by
    CNN training. The resulting float32 raw-pixel or Hermite-coefficient
    statistics define the normalized space reconstructed by stage 4.

    Parameters
    ----------
    data : AutoencoderTrainingData
        Memmap-backed raw or Hermite representation.
    train_indices : sequence of int
        Saved rows assigned to training.
    config : mapping
        Data-loader configuration containing normalization batch size.

    Returns
    -------
    InputFeatureScaler
        Float32 feature means and scales accumulated in float64.
    """

    return fit_input_scaler(
        data,
        train_indices,
        batch_size=config["normalization_batch_size"],
        epsilon=DEFAULT_SCALER_EPSILON,
    )


__all__ = ["scale_autoencoder_inputs"]
