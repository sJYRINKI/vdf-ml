"""Prepare, scale, and score shared auxiliary topology supervision.

The current topology representation contains two X/O-point distances and
four x-z displacement components in Earth radii. Arrays use shape
``(n_samples, 6)`` and the fixed order from
``src.data.metadata_columns.TOPOLOGY_TARGET_COLUMNS``:
``distance_to_x_point_re``, ``distance_to_o_point_re``,
``vdf_to_x_point_dx_re``, ``vdf_to_x_point_dz_re``,
``vdf_to_o_point_dx_re``, and ``vdf_to_o_point_dz_re``. Metadata extraction
retains every sample, records finite entries in a Boolean validity mask, and
zero-fills unavailable values only so masked tensor operations remain direct.

The immutable scaler fits one population mean and standard deviation per
target from valid training entries only. It applies those training-derived
values to all partitions and restores predictions to Earth radii for
evaluation. The masked Smooth L1 calculation averages globally over valid
scalar entries and returns a graph-connected zero when none are available.
Topology values are auxiliary supervised targets; they are never model
inputs and never determine sample selection or reconstruction weighting.
"""

from dataclasses import dataclass

import numpy as np
import torch
from torch.nn import functional

from src.data.metadata_columns import TOPOLOGY_TARGET_COLUMNS


_DEFAULT_SCALER_EPSILON = 1.0e-8


def create_topology_targets(metadata):
    """Create the six topology targets and their finite-value mask.

    Dataset consumers call this after loading ``metadata.csv`` in saved
    sample order. Each topology entry is handled independently, so a sample
    with one or more unavailable X/O quantities remains usable by every
    model objective that does not depend on those missing entries.

    Parameters
    ----------
    metadata : pandas.DataFrame
        Saved rows containing topology values in Earth radii.

    Returns
    -------
    targets : numpy.ndarray
        Earth-radius values with shape ``(n_samples, 6)`` in the fixed
        topology-target order, with nonfinite entries filled by zero.
    mask : numpy.ndarray
        Boolean finite-value mask with shape ``(n_samples, 6)``.

    Notes
    -----
    The returned values are auxiliary targets only. They are not appended to
    raw VDF voxels, Hermite coefficients, or model activations.
    """

    values = np.asarray(
        metadata.loc[:, TOPOLOGY_TARGET_COLUMNS],
        dtype=np.float64,
    )
    mask = np.isfinite(values)
    return np.where(mask, values, 0.0), mask


@dataclass(frozen=True)
class TopologyTargetScaler:
    """Hold masked scaling for the six topology targets.

    Training fits one mean and population standard deviation per target from
    valid training values only. Optimization consumes scaled targets;
    evaluation and prediction apply the inverse transform to recover Earth
    radii. Validation and test values never influence the fitted parameters,
    and the fixed target order is part of the checkpoint contract.

    Parameters
    ----------
    mean : numpy.ndarray
        Per-target training means with shape ``(6,)`` in topology schema
        order and units of Earth radii.
    scale : numpy.ndarray
        Per-target training population standard deviations with shape
        ``(6,)`` and units of Earth radii.
    epsilon : float, optional
        Threshold at or below which a scale is replaced by one.
    target_names : tuple of str, optional
        Ordered topology target names.
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
    Missing targets are represented by a separate Boolean mask rather than
    a physical zero. Topology supervision shapes a model's shared latent or
    embedding representation but does not condition the model input.
    """

    mean: np.ndarray
    scale: np.ndarray
    epsilon: float = _DEFAULT_SCALER_EPSILON
    target_names: tuple = TOPOLOGY_TARGET_COLUMNS

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
        epsilon=_DEFAULT_SCALER_EPSILON,
    ):
        """Fit each topology column from valid training values.

        The mask excludes unavailable X- or O-point targets independently in
        each column. Population statistics use float64 and become the single
        scaling source for training, validation, test, and later prediction.

        Parameters
        ----------
        targets : numpy.ndarray or torch.Tensor
            Physical topology values with shape ``(n_samples, 6)`` in Earth
            radii.
        mask : numpy.ndarray or torch.Tensor
            Boolean finite-value mask with shape ``(n_samples, 6)``.
        train_indices : sequence of int
            Saved rows assigned to training in unchanged sample order.
        epsilon : float, optional
            Near-zero scale threshold in Earth radii.

        Returns
        -------
        TopologyTargetScaler
            Float64 training means and scales in fixed target order.
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
            Values whose final dimension follows the six-target order.

        Returns
        -------
        numpy.ndarray or torch.Tensor
            Scaled values with the input kind, device, dtype, and shape.
        """

        return self._linear_transform(targets, inverse=False)

    def transform_masked(self, targets, mask):
        """Scale valid topology values and zero-fill missing entries.

        Partition datasets call this for their selected rows. Zero filling is
        paired with the returned mask, so unavailable values contribute
        nothing to masked topology loss while samples remain available for
        other model objectives.

        Parameters
        ----------
        targets : numpy.ndarray or torch.Tensor
            Physical topology values with shape ``(n_samples, 6)``.
        mask : numpy.ndarray or torch.Tensor
            Boolean finite-value mask with the same shape.

        Returns
        -------
        scaled : numpy.ndarray or torch.Tensor
            Scaled values with invalid entries filled by zero.
        mask : numpy.ndarray or torch.Tensor
            Boolean mask copied into the same array system.
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

        Evaluation and prediction use this operation after model inference.
        When a mask is supplied, unavailable physical targets become
        ``NaN``; unmasked model predictions are restored in full.

        Parameters
        ----------
        targets : numpy.ndarray or torch.Tensor
            Scaled values whose final dimension has length six.
        mask : numpy.ndarray or torch.Tensor, optional
            Boolean finite-value mask. Invalid returned values become
            ``NaN``.

        Returns
        -------
        numpy.ndarray or torch.Tensor
            Physical topology values in Earth radii with the input shape.
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
        """Return topology scaling as device-independent checkpoint data.

        Returns
        -------
        dict
            Target order, means, scales, and epsilon. Mean and scale retain
            the established checkpoint field names and values.
        """

        return {
            "target_names": list(self.target_names),
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
            "epsilon": self.epsilon,
        }

    @classmethod
    def from_dict(cls, record):
        """Reconstruct topology scaling from checkpoint data.

        Parameters
        ----------
        record : dict
            Stored target means, scales, order, and epsilon.

        Returns
        -------
        TopologyTargetScaler
            Six-target training scaling for inference or evaluation.
        """

        return cls(
            mean=np.asarray(record["mean"], dtype=np.float64),
            scale=np.asarray(record["scale"], dtype=np.float64),
            epsilon=record["epsilon"],
            target_names=tuple(record["target_names"]),
        )

    def _linear_transform(self, targets, *, inverse):
        """Apply topology scaling while preserving array storage.

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


def calculate_masked_topology_loss(predictions, targets, mask):
    """Calculate globally masked Smooth L1 topology error.

    The calculation compares scaled predictions and training-derived scaled
    targets only where metadata supplied a finite physical value. It sums
    Smooth L1 error over all valid scalar entries and divides by their global
    count, matching the established CNN objective exactly. A batch without
    valid entries returns a differentiable zero connected to predictions.

    Parameters
    ----------
    predictions : torch.Tensor
        Scaled topology predictions with shape ``(batch_size, 6)``.
    targets : torch.Tensor
        Scaled topology targets with shape ``(batch_size, 6)`` in fixed
        project order.
    mask : torch.Tensor
        Boolean validity mask with shape ``(batch_size, 6)``.

    Returns
    -------
    topology_loss : torch.Tensor
        Scalar Smooth L1 error averaged over valid topology entries.
    valid_count : int
        Number of valid scalar topology targets in the batch.

    Notes
    -----
    Topology targets are auxiliary supervision and never model inputs. The
    graph-connected zero keeps the topology output branch part of autograd
    even when the current batch has no available topology values.
    """

    valid_count = int(torch.count_nonzero(mask).item())
    if valid_count:
        topology_loss = functional.smooth_l1_loss(
            predictions[mask],
            targets[mask],
            reduction="sum",
        ) / valid_count
    else:
        topology_loss = predictions.sum() * 0.0
    return topology_loss, valid_count


__all__ = [
    "TopologyTargetScaler",
    "calculate_masked_topology_loss",
    "create_topology_targets",
]
