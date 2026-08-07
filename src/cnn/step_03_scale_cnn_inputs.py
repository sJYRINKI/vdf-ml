"""Stage 3: fit and apply training-derived CNN scaling.

This stage follows the timestep split and precedes model construction. It
uses only training rows to calculate one mean and scale per representation
feature and per valid topology target, then applies those values to every
partition and to later prediction.

The stage receives sample-wise training data plus the stage-2 training
indices. It returns separate immutable scalers for model inputs and the six
topology targets; both are serialized into the CNN checkpoint for
prediction.
"""

from dataclasses import dataclass

import numpy as np
import torch

from src.data.metadata_columns import (
    TOPOLOGY_SCHEMA_VERSION,
    TOPOLOGY_TARGET_COLUMNS,
)


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
        Float32 arrays with CNN raw ``(vx, vy, vz)``, autoencoder raw
        ``(vx, vz)``, or the complete saved Hermite coefficient shape.
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


@dataclass(frozen=True)
class TopologyTargetScaler:
    """Hold masked scaling for the six topology targets.

    CNN stage 3 fits one mean and population standard deviation per target
    from valid training values only. Training consumes scaled targets;
    evaluation and prediction apply the inverse transform to recover Earth
    radii. The target order is part of the saved checkpoint contract.

    Parameters
    ----------
    mean : numpy.ndarray
        Per-target training means in topology schema order.
    scale : numpy.ndarray
        Per-target training population standard deviations.
    epsilon : float, optional
        Threshold at or below which scale is replaced by one.
    target_names : tuple of str, optional
        Ordered topology target names.
    schema_version : str, optional
        Numeric topology schema version.

    Attributes
    ----------
    mean, scale : numpy.ndarray
        Float64 vectors with shape ``(6,)`` in
        ``TOPOLOGY_TARGET_COLUMNS`` order.
    target_names : tuple of str
        Distances to X and O points followed by the four x-z displacement
        components.

    Notes
    -----
    The physical values and inverse-transformed predictions are expressed
    in Earth radii. Missing targets are represented by the separate mask,
    not by a learned numerical value.
    """

    mean: np.ndarray
    scale: np.ndarray
    epsilon: float = DEFAULT_SCALER_EPSILON
    target_names: tuple = TOPOLOGY_TARGET_COLUMNS
    schema_version: str = TOPOLOGY_SCHEMA_VERSION

    def __post_init__(self):
        """Copy topology parameters and freeze their target-name order."""

        object.__setattr__(
            self,
            "mean",
            np.asarray(self.mean, dtype=np.float64).copy(),
        )
        object.__setattr__(
            self,
            "scale",
            np.asarray(self.scale, dtype=np.float64).copy(),
        )
        object.__setattr__(self, "epsilon", float(self.epsilon))
        object.__setattr__(
            self,
            "target_names",
            tuple(str(value) for value in self.target_names),
        )

    @classmethod
    def fit(
        cls,
        targets,
        mask,
        train_indices,
        epsilon=DEFAULT_SCALER_EPSILON,
    ):
        """Fit each topology column from valid training values.

        The mask excludes unavailable X- or O-point targets independently
        in each column. Population statistics use float64 and become the
        single scaling source for every partition and later prediction.

        Parameters
        ----------
        targets : numpy.ndarray or torch.Tensor
            Physical topology values with shape ``(n_samples, 6)``.
        mask : numpy.ndarray or torch.Tensor
            Missing-target mask with the same shape.
        train_indices : sequence of int
            Saved rows assigned to training.
        epsilon : float, optional
            Near-zero scale threshold.

        Returns
        -------
        TopologyTargetScaler
            Float64 target means and scales in schema order.
        """

        values = (
            targets.detach().cpu().numpy()
            if torch.is_tensor(targets)
            else np.asarray(targets)
        )
        boolean_mask = (
            mask.detach().cpu().numpy()
            if torch.is_tensor(mask)
            else np.asarray(mask, dtype=bool)
        )
        indices = np.asarray(train_indices, dtype=np.int64)
        means = np.empty(len(TOPOLOGY_TARGET_COLUMNS), dtype=np.float64)
        scales = np.empty_like(means)
        for target_index in range(len(TOPOLOGY_TARGET_COLUMNS)):
            valid = boolean_mask[indices, target_index]
            selected = values[indices, target_index][valid]
            means[target_index] = np.mean(selected, dtype=np.float64)
            standard_deviation = np.std(selected, dtype=np.float64)
            scales[target_index] = (
                1.0
                if standard_deviation <= float(epsilon)
                else standard_deviation
            )
        return cls(mean=means, scale=scales, epsilon=epsilon)

    def transform(self, targets):
        """Scale complete topology values.

        This direct path is used when every supplied value is meaningful.
        Masked training batches use :meth:`transform_masked` instead.

        Parameters
        ----------
        targets : numpy.ndarray or torch.Tensor
            Values with final dimension six.

        Returns
        -------
        numpy.ndarray or torch.Tensor
            Scaled values with the input kind and shape.
        """

        return self._linear_transform(targets, inverse=False)

    def transform_masked(self, targets, mask):
        """Scale valid topology values and zero-fill missing entries.

        Stage-1 partition datasets call this once for their selected rows.
        Zero filling is paired with the returned mask, so unavailable
        values contribute nothing to the auxiliary topology loss.

        Parameters
        ----------
        targets : numpy.ndarray or torch.Tensor
            Topology values with shape ``(n_samples, 6)``.
        mask : numpy.ndarray or torch.Tensor
            Missing-target mask with the same shape.

        Returns
        -------
        scaled : numpy.ndarray or torch.Tensor
            Scaled values with invalid entries filled by zero.
        mask : numpy.ndarray or torch.Tensor
            Boolean mask copied to the same array system.
        """

        transformed = self._linear_transform(targets, inverse=False)
        if torch.is_tensor(targets):
            boolean_mask = torch.as_tensor(
                mask,
                dtype=torch.bool,
                device=targets.device,
            )
            return (
                torch.where(
                    boolean_mask,
                    transformed,
                    torch.zeros_like(transformed),
                ),
                boolean_mask.clone(),
            )
        boolean_mask = np.asarray(mask, dtype=bool)
        return np.where(boolean_mask, transformed, 0.0), boolean_mask.copy()

    def inverse_transform(self, targets, mask=None):
        """Restore scaled topology values to physical Earth radii.

        Evaluation and prediction use this operation after the CNN forward
        pass. When a mask is supplied, unavailable targets become ``NaN``;
        unmasked model predictions are restored in full.

        Parameters
        ----------
        targets : numpy.ndarray or torch.Tensor
            Scaled values with final dimension six.
        mask : numpy.ndarray or torch.Tensor, optional
            Missing-target mask. Invalid returned values become ``NaN``.

        Returns
        -------
        numpy.ndarray or torch.Tensor
            Physical topology values in Earth radii.
        """

        restored = self._linear_transform(targets, inverse=True)
        if mask is None:
            return restored
        if torch.is_tensor(restored):
            boolean_mask = torch.as_tensor(
                mask,
                dtype=torch.bool,
                device=restored.device,
            )
            return torch.where(
                boolean_mask,
                restored,
                torch.full_like(restored, float("nan")),
            )
        return np.where(np.asarray(mask, dtype=bool), restored, np.nan)

    def to_dict(self):
        """Return topology scaling as checkpoint data.

        Stage 8 stores this primitive form so prediction retains the six
        target names, numeric schema version, and training statistics.

        Returns
        -------
        dict
            Target order, means, scales, epsilon, and schema version.
        """

        return {
            "schema_version": self.schema_version,
            "target_names": list(self.target_names),
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
            "epsilon": self.epsilon,
        }

    @classmethod
    def from_dict(cls, record):
        """Reconstruct topology scaling from checkpoint data.

        Checkpoint loading calls this before inverse-transforming topology
        predictions to Earth radii.

        Parameters
        ----------
        record : dict
            Stored target means, scales, order, and schema version.

        Returns
        -------
        TopologyTargetScaler
            Six-target training scaling.
        """

        return cls(
            mean=np.asarray(record["mean"], dtype=np.float64),
            scale=np.asarray(record["scale"], dtype=np.float64),
            epsilon=record["epsilon"],
            target_names=tuple(record["target_names"]),
            schema_version=record["schema_version"],
        )

    def _linear_transform(self, targets, *, inverse):
        """Apply topology scaling while preserving NumPy or PyTorch storage.

        Parameters
        ----------
        targets : numpy.ndarray or torch.Tensor
            Values whose final dimension follows the six-target order.
        inverse : bool
            Apply ``value * scale + mean`` when true, otherwise
            ``(value - mean) / scale``.

        Returns
        -------
        numpy.ndarray or torch.Tensor
            Transformed values with the input shape, device, and array kind.
        """

        if torch.is_tensor(targets):
            mean = torch.as_tensor(
                self.mean,
                dtype=targets.dtype,
                device=targets.device,
            )
            scale = torch.as_tensor(
                self.scale,
                dtype=targets.dtype,
                device=targets.device,
            )
        else:
            values = np.asarray(targets)
            dtype = np.result_type(values.dtype, np.float64)
            mean = self.mean.astype(dtype, copy=False)
            scale = self.scale.astype(dtype, copy=False)
            targets = values
        return targets * scale + mean if inverse else (targets - mean) / scale


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
    ``(vx, vy, vz)`` axes; autoencoder raw statistics retain ``(vx, vz)``;
    Hermite statistics follow the complete saved unrotated
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
    "TopologyTargetScaler",
    "fit_input_scaler",
]
