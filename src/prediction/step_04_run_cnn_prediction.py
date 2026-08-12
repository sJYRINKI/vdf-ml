"""Stage 4: run CNN inference and restore physical topology units.

This stage follows raw or Hermite tensor preparation and precedes output
serialization. It receives aligned batches of unnormalized representation
tensors and physical plasma-context rows, then returns
physical-class probabilities plus six topology predictions in Earth radii.
"""

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class PredictionBatch:
    """Hold decoded class and physical topology predictions.

    Stage 4 creates one immutable record per inference batch. Arrays follow
    the input-row order and no object owns a device tensor after decoding;
    class probabilities and inverse-scaled topology values are copied back
    to NumPy for Stage 5 serialization.

    Attributes
    ----------
    class_ids : numpy.ndarray
        Predicted project class IDs with shape ``(batch,)``.
    class_names : tuple of str
        Physical class names in input-row order.
    probabilities : numpy.ndarray
        Class probabilities with shape ``(batch, n_classes)`` in checkpoint
        output order.
    maximum_probabilities : numpy.ndarray
        Maximum class probability for each input row.
    topology_values : numpy.ndarray
        Six inverse-scaled topology predictions with shape ``(batch, 6)``.
        Distances and ``dx,dz`` components are expressed in Earth radii.
    """

    class_ids: np.ndarray
    class_names: tuple
    probabilities: np.ndarray
    maximum_probabilities: np.ndarray
    topology_values: np.ndarray


def run_cnn_prediction(loaded, tensors, plasma_context):
    """Infer outputs from prepared VDF tensors and plasma context.

    This stage inserts a missing channel axis, moves float32 inputs to the
    model device, scales the aligned physical context with training-only
    checkpoint statistics, and evaluates the current CNN without gradients.
    It maps argmax output indices through the checkpoint's explicit
    physical-class order and restores all six topology outputs from
    training-scaled space to Earth radii before returning.

    Parameters
    ----------
    loaded : LoadedCnnCheckpoint
        CNN, training representation/context/topology scalers, and checkpoint
        mapping.
    tensors : numpy.ndarray
        Unnormalized raw VDF volumes or Hermite coefficient volumes with shape
        ``(batch, *input_shape)`` or ``(batch, 1, *input_shape)``.
    plasma_context : numpy.ndarray
        Physical same-cell values with shape ``(batch, 16)`` in fixed
        B-component, E-component, fluid-velocity-component, density, and
        pressure order stored by the checkpoint.

    Returns
    -------
    PredictionBatch
        Class probabilities and inverse-scaled topology values in input-row
        order.

    Notes
    -----
    All Cartesian B, E, and V components are supplied to the model after
    training-only scaling; no redundant vector magnitude is appended. Spatial
    coordinates and topology values remain outside model inputs. The scaled
    context tensor moves directly to the output device while Conv3d stages
    retain their existing one-process model-parallel placement.
    """

    values = np.asarray(tensors, dtype=np.float32)
    if values.ndim == len(loaded.model.input_shape) + 1:
        values = values[:, None, ...]
    device = loaded.model.input_device
    inputs = torch.as_tensor(
        values,
        dtype=torch.float32,
        device=device,
    )
    scaled_context = loaded.plasma_context_scaler.transform(plasma_context)
    context_inputs = torch.as_tensor(
        scaled_context,
        device=loaded.model.output_device,
    )
    loaded.model.eval()
    with torch.inference_mode():
        outputs = loaded.model(inputs, context_inputs)
        probabilities = torch.softmax(
            outputs["class_logits"],
            dim=1,
        ).cpu().numpy()
        scaled_topology = outputs[
            "topology_predictions"
        ].cpu().numpy()
    topology_values = np.asarray(
        loaded.topology_scaler.inverse_transform(scaled_topology),
        dtype=np.float64,
    )
    output_indices = np.argmax(probabilities, axis=1)
    mapping = loaded.checkpoint["class_mapping"]
    class_ids = np.asarray(
        mapping["class_ids"],
        dtype=np.int64,
    )[output_indices]
    configured_names = tuple(str(value) for value in mapping["class_names"])
    class_names = tuple(
        configured_names[int(index)] for index in output_indices
    )
    return PredictionBatch(
        class_ids=class_ids,
        class_names=class_names,
        probabilities=probabilities,
        maximum_probabilities=np.max(probabilities, axis=1),
        topology_values=topology_values,
    )


__all__ = ["PredictionBatch", "run_cnn_prediction"]
