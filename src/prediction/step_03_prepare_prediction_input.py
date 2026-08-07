"""Stage 3: prepare one source VDF as a raw or Hermite CNN tensor.

This stage follows file-scoped VLSV loading and precedes CNN inference. It
receives a selected spatial cell and the prepared source, then returns the
complete raw VDF volume or complete Hermite coefficient volume used by the
checkpoint.
"""

from dataclasses import dataclass
import gc

import numpy as np

from src.physics.hermite_rotation import rotate_vdf
from src.physics.hermite_transform import vdf_to_hermite
from src.physics.physical_context import (
    get_hermite_rotation_context,
)
from src.representations.step_02_prepare_raw_input import (
    prepare_raw_input,
)


@dataclass(frozen=True)
class PredictionSample:
    """Hold one unnormalized CNN representation tensor.

    Stage 3 returns this lightweight record for exactly one selected VDF
    cell. It owns the prepared in-memory array but no VLSV reader or file
    handle; Stage 4 adds the batch and channel axes and applies the
    training-derived normalization stored by the CNN.

    Attributes
    ----------
    tensor : numpy.ndarray
        Raw volume in ``[vx, vy, vz]`` order, unrotated Hermite coefficients
        in ``[n_x, n_y, n_z]`` order, or rotated coefficients in
        ``[n_parallel, n_perp1, n_perp2]`` order.
    """

    tensor: np.ndarray


def prepare_prediction_input(prepared_source, cid, loaded):
    """Create one raw or Hermite input tensor for CNN inference.

    This dispatch stage applies the representation convention recorded in
    the checkpoint. Raw input is interpolated on physical VDF values across
    all three velocity axes before the complete volume is transformed,
    whereas Hermite input is projected from same-cell physical context
    without resampling a coefficient volume.

    Parameters
    ----------
    prepared_source : PreparedPredictionSource
        Open source reader, physical producers, and dense VDF extractor.
    cid : int
        Selected VDF-carrying spatial cell ID.
    loaded : LoadedCnnCheckpoint
        Checkpoint defining representation preprocessing.

    Returns
    -------
    PredictionSample
        Unnormalized tensor consumed by Stage 4.
    """

    if loaded.checkpoint["representation"] == "raw":
        return _prepare_raw_prediction_input(
            prepared_source,
            cid,
            loaded,
        )
    return _prepare_hermite_prediction_input(
        prepared_source,
        cid,
        loaded,
    )


def _prepare_raw_prediction_input(prepared_source, cid, loaded):
    """Interpolate and transform one complete raw VDF for ``Conv3d``.

    Physical VDF values are interpolated to the checkpoint's training grid
    before a positive floor and ``log10`` are applied to every
    ``(vx, vy, vz)`` cell. The source VDF remains unchanged, and prediction
    reproduces the complete-volume preprocessing used by CNN training.

    Parameters
    ----------
    prepared_source : PreparedPredictionSource
        Source extractor and precomputed source-to-training interpolation
        plan.
    cid : int
        Selected VDF-bearing spatial cell ID.
    loaded : LoadedCnnCheckpoint
        Checkpoint containing raw preprocessing and training-grid metadata.

    Returns
    -------
    PredictionSample
        Complete ``(vx, vy, vz)`` logarithmic input in model axis order.
    """

    raw_input = prepared_source.extractor.extract_interpolated_volume(
        int(cid),
        prepared_source.interpolation_plan,
    )
    preprocessing = loaded.checkpoint["preprocessing"][
        "raw_preprocessing"
    ]
    return PredictionSample(
        tensor=prepare_raw_input(
            raw_input,
            log_eps=preprocessing["log_floor"],
        )
    )


def _prepare_hermite_prediction_input(prepared_source, cid, loaded):
    """Project one physical VDF with the checkpoint's Hermite settings.

    The branch copies the physical ``[vx, vy, vz]`` VDF and projects its
    linear values directly with endpoint coordinates and the physical-velocity
    basis. Unrotated checkpoints require no magnetic field, bulk velocity, or
    MinValue read. Rotated checkpoints first use the same-cell magnetic field
    and total bulk flow to reproduce the training grid, then calculate VDF
    moments and coefficients in ``[parallel, perp1, perp2]`` order.

    Parameters
    ----------
    prepared_source : PreparedPredictionSource
        Source reader, same-cell producers, and dense VDF extractor.
    cid : int
        Selected VDF-bearing spatial cell ID.
    loaded : LoadedCnnCheckpoint
        Checkpoint containing the actual Hermite order and rotation setting.

    Returns
    -------
    PredictionSample
        Contiguous float32 physical-VDF Hermite cube in checkpoint order.
    """

    raw_vdf = np.array(
        prepared_source.extractor.extract(int(cid)),
        dtype=np.float32,
        order="C",
        copy=True,
    )
    gc.collect()
    transform_vdf = raw_vdf
    shape = tuple(int(value) for value in raw_vdf.shape)
    velocity_limits_mps = np.asarray(
        prepared_source.extractor.velocity_grid["extent_mps"],
        dtype=np.float64,
    )
    if loaded.checkpoint["hermite_rotate"]:
        magnetic_field, bulk_velocity = get_hermite_rotation_context(
            prepared_source.reader,
            int(cid),
            resolved_source=prepared_source.resolved_source,
            velocity_source=prepared_source.bulk_velocity_source,
        )
        transform_vdf, shape, velocity_limits_mps, _ = rotate_vdf(
            raw_vdf,
            shape,
            velocity_limits_mps,
            magnetic_field,
            bulk_velocity,
        )
    coefficients = vdf_to_hermite(
        transform_vdf,
        shape,
        velocity_limits_mps,
        order=loaded.checkpoint["hermite_order"],
    )
    return PredictionSample(
        tensor=np.array(
            coefficients,
            dtype=np.float32,
            copy=True,
            order="C",
        )
    )


__all__ = ["PredictionSample", "prepare_prediction_input"]
