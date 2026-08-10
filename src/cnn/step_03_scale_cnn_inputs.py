"""Stage 3: fit and apply training-derived CNN input scaling.

This stage follows the timestep split and precedes model construction. It
uses only training rows to calculate one mean and scale per representation
feature, then applies those values to every partition and later prediction.
The focused ``src.learning.topology_supervision`` module owns the independent
six-target topology scaling used alongside this input normalization.

The stage receives sample-wise training data plus the stage-2 training
indices. It returns immutable model-input scaling serialized into the CNN
checkpoint for prediction.
"""

from dataclasses import dataclass

import numpy as np
import torch

DEFAULT_SCALER_EPSILON = 1.0e-8


@dataclass(frozen=True)
class InputFeatureScaler:
    """Hold per-pixel or per-coefficient training normalization.

    Stage 3 fits this immutable scaler from training samples only. The CNN
    and autoencoder store the resulting float32 arrays as model buffers,
    while their checkpoints retain the same values for reconstruction and
    prediction. The object owns NumPy arrays but no dataset mapping.

    Parameters
    ----------
    mean : numpy.ndarray
        Training mean with the representation spatial shape.
    scale : numpy.ndarray
        Training population standard deviation with the same shape.
    epsilon : float, optional
        Threshold at or below which scale is replaced by one.

    Attributes
    ----------
    mean, scale : numpy.ndarray
        Float32 arrays with complete raw ``(vx, vy, vz)`` or the complete
        saved Hermite coefficient shape.
    epsilon : float
        Near-constant-feature threshold used during fitting.
    """

    mean: np.ndarray
    scale: np.ndarray
    epsilon: float = DEFAULT_SCALER_EPSILON

    def __post_init__(self):
        """Copy normalization arrays into immutable float32-owned storage."""

        object.__setattr__(
            self,
            "mean",
            np.asarray(self.mean, dtype=np.float32).copy(),
        )
        object.__setattr__(
            self,
            "scale",
            np.asarray(self.scale, dtype=np.float32).copy(),
        )
        object.__setattr__(self, "epsilon", float(self.epsilon))

    @classmethod
    def fit(cls, inputs, train_indices, epsilon=DEFAULT_SCALER_EPSILON):
        """Fit feature normalization from training tensors only.

        This in-memory fitting path is used for already materialized model
        tensors. It computes population mean and standard deviation in
        float64, replaces near-constant scales with one, and stores float32
        values for the next model-construction stage.

        Parameters
        ----------
        inputs : numpy.ndarray or torch.Tensor
            Tensors with shape ``(n_samples, 1, *input_shape)``.
        train_indices : sequence of int
            Saved rows assigned to training.
        epsilon : float, optional
            Near-zero scale threshold.

        Returns
        -------
        InputFeatureScaler
            Float32 mean and scale accumulated in float64.
        """

        values = (
            inputs.detach().cpu().numpy()
            if torch.is_tensor(inputs)
            else np.asarray(inputs)
        )
        indices = np.asarray(train_indices, dtype=np.int64)
        training = values[indices, 0].astype(np.float64, copy=False)
        mean = np.mean(training, axis=0, dtype=np.float64)
        scale = np.std(training, axis=0, dtype=np.float64)
        scale = np.where(scale <= float(epsilon), 1.0, scale)
        return cls(
            mean=mean.astype(np.float32),
            scale=scale.astype(np.float32),
            epsilon=epsilon,
        )

    def transform(self, inputs):
        """Apply stored normalization while preserving array kind.

        Model preprocessing calls this operation when normalization must
        remain outside a network. NumPy arrays stay NumPy arrays and
        PyTorch tensors retain their device and differentiability.

        Parameters
        ----------
        inputs : numpy.ndarray or torch.Tensor
            Representation tensors with channel dimension one.

        Returns
        -------
        numpy.ndarray or torch.Tensor
            Normalized values with the input shape and array kind.
        """

        if torch.is_tensor(inputs):
            mean = torch.as_tensor(
                self.mean,
                dtype=inputs.dtype,
                device=inputs.device,
            ).unsqueeze(0).unsqueeze(0)
            scale = torch.as_tensor(
                self.scale,
                dtype=inputs.dtype,
                device=inputs.device,
            ).unsqueeze(0).unsqueeze(0)
            return (inputs - mean) / scale
        values = np.asarray(inputs, dtype=np.float32)
        return (
            values - self.mean[np.newaxis, np.newaxis, ...]
        ) / self.scale[np.newaxis, np.newaxis, ...]

    def to_dict(self):
        """Return feature normalization as checkpoint data.

        The primitive lists let checkpoint saving retain preprocessing
        without pickling this Python class.

        Returns
        -------
        dict
            Per-feature mean, scale, epsilon, and spatial shape.
        """

        return {
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
            "epsilon": self.epsilon,
            "shape": list(self.mean.shape),
        }

    @classmethod
    def from_dict(cls, record):
        """Reconstruct feature normalization from checkpoint data.

        CNN and autoencoder checkpoint loading call this before rebuilding
        the model so inference uses the exact training-derived values.

        Parameters
        ----------
        record : dict
            Stored mean, scale, and epsilon.

        Returns
        -------
        InputFeatureScaler
            Training-derived feature scaling.
        """

        return cls(
            mean=np.asarray(record["mean"], dtype=np.float32),
            scale=np.asarray(record["scale"], dtype=np.float32),
            epsilon=record["epsilon"],
        )


def fit_input_scaler(
    data,
    train_indices,
    *,
    batch_size=1,
    epsilon=DEFAULT_SCALER_EPSILON,
):
    """Fit bounded input normalization in stable training-row order.

    This is the sample-wise stage-3 fitting path used by both training
    workflows. It reads bounded batches from the representation memory map
    and combines their float64 moments without constructing the complete
    processed dataset in RAM.

    Parameters
    ----------
    data : CnnTrainingData or AutoencoderTrainingData
        Memmap-backed representation description.
    train_indices : sequence of int
        Saved rows assigned to training.
    batch_size : int, optional
        Maximum sample count processed in one accumulation batch.
    epsilon : float, optional
        Near-zero scale threshold.

    Returns
    -------
    InputFeatureScaler
        Float32 feature means and scales accumulated in float64.

    Notes
    -----
    Welford batch combination is performed in caller order to preserve the
    established floating-point results. CNN raw statistics follow complete
    ``(vx, vy, vz)`` axes for both CNN and autoencoder; Hermite statistics
    follow the complete saved unrotated
    ``(n_x, n_y, n_z)`` or rotated
    ``(n_parallel, n_perp1, n_perp2)`` axes.
    """

    indices = np.asarray(train_indices, dtype=np.int64)
    count = 0
    mean = np.zeros(data.input_shape, dtype=np.float64)
    sum_squared_deviations = np.zeros_like(mean)
    with data.create_reader() as reader:
        for start in range(0, len(indices), int(batch_size)):
            batch_indices = indices[start : start + int(batch_size)]
            batch = np.empty(
                (len(batch_indices), *data.input_shape),
                dtype=np.float64,
            )
            for batch_offset, source_index in enumerate(batch_indices):
                batch[batch_offset] = reader.read(int(source_index))

            batch_count = len(batch)
            batch_mean = np.mean(batch, axis=0, dtype=np.float64)
            batch_deviations = batch - batch_mean
            batch_sum_squared_deviations = np.sum(
                batch_deviations * batch_deviations,
                axis=0,
                dtype=np.float64,
            )
            combined_count = count + batch_count
            delta = batch_mean - mean
            sum_squared_deviations += (
                batch_sum_squared_deviations
                + delta
                * delta
                * (count * batch_count / combined_count)
            )
            mean += delta * (batch_count / combined_count)
            count = combined_count

    variance = np.maximum(sum_squared_deviations / count, 0.0)
    scale = np.sqrt(variance)
    scale = np.where(scale <= float(epsilon), 1.0, scale)
    return InputFeatureScaler(
        mean=mean.astype(np.float32),
        scale=scale.astype(np.float32),
        epsilon=epsilon,
    )


__all__ = [
    "DEFAULT_SCALER_EPSILON",
    "InputFeatureScaler",
    "fit_input_scaler",
]
