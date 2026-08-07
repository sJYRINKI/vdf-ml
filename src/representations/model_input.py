"""Describe and construct sample-wise raw or Hermite model tensors.

This module dispatches a loaded representation to the alternative raw and
Hermite preparation stages for CNN consumers. PCA owns a separate restartable
sample-batch iterator so it never materializes every feature row. The
autoencoder-specific raw plane preparation remains inside the autoencoder
package.
"""

from dataclasses import dataclass

import numpy as np

from src.representations.step_02_prepare_raw_input import (
    describe_raw_input,
    prepare_raw_input,
)
from src.representations.step_03_prepare_hermite_input import (
    describe_hermite_input,
)


@dataclass(frozen=True)
class RepresentationTensorSpec:
    """Describe one sample-wise representation tensor.

    CNN datasets retain this lightweight description instead of
    materializing every processed sample. A dataset worker uses it to read
    one memory-mapped row, apply the representation transform, and construct
    a channel-free tensor immediately before model normalization.

    Parameters
    ----------
    representation : str
        Public representation name, ``"raw"`` or ``"hermite"``.
    source_filename : str
        Stored source array filename.
    source_shape : tuple of int
        Complete stored source-array shape.
    source_dtype : str
        NumPy dtype string for the stored source array.
    sample_count : int
        Number of aligned representation samples.
    tensor_shape : tuple of int
        One sample's spatial tensor shape without a channel axis.
    feature_metadata : dict
        Raw preprocessing or saved physical-VDF Hermite shape, order,
        rotation, and coefficient-axis metadata.
    """

    representation: str
    source_filename: str
    source_shape: tuple
    source_dtype: str
    sample_count: int
    tensor_shape: tuple
    feature_metadata: dict

    def __post_init__(self):
        """Store stable scalar and tuple types in the frozen description.

        NumPy archive values often arrive as NumPy scalar types. Converting
        them once keeps worker serialization predictable without opening or
        copying the represented sample array.
        """

        object.__setattr__(
            self,
            "source_shape",
            tuple(int(value) for value in self.source_shape),
        )
        object.__setattr__(
            self,
            "source_dtype",
            np.dtype(self.source_dtype).str,
        )
        object.__setattr__(self, "sample_count", int(self.sample_count))
        object.__setattr__(
            self,
            "tensor_shape",
            tuple(int(value) for value in self.tensor_shape),
        )
        object.__setattr__(
            self,
            "feature_metadata",
            dict(self.feature_metadata),
        )


def describe_representation_tensor(loaded, *, raw_config=None):
    """Describe sample-wise tensors without materializing representation rows.

    This model-facing dispatch follows saved-representation loading and
    precedes CNN dataset construction. It derives only the per-sample tensor
    shape and preprocessing metadata so workers can keep the source array
    memory mapped and read complete rows on demand.

    Parameters
    ----------
    loaded : LoadedRepresentation
        Directly loaded public representation.
    raw_config : mapping, optional
        Raw-only configuration containing ``log_eps``.

    Returns
    -------
    RepresentationTensorSpec
        Stored-array identity, one-sample tensor shape, and numerical
        preprocessing convention without materializing sample tensors.
    """

    if loaded.representation == "raw":
        feature_metadata = describe_raw_input(
            loaded.values,
            log_eps=raw_config["log_eps"],
        )
        tensor_shape = tuple(feature_metadata["volume_shape"])
    else:
        feature_metadata = describe_hermite_input(
            loaded.values,
            rotated=bool(loaded.metadata["hermite_rotate"].iloc[0]),
        )
        tensor_shape = tuple(feature_metadata["volume_shape"])
    feature_metadata["source_shape"] = [
        int(value) for value in loaded.values.shape
    ]
    return RepresentationTensorSpec(
        representation=loaded.representation,
        source_filename=loaded.source_filename,
        source_shape=tuple(loaded.values.shape),
        source_dtype=np.dtype(loaded.values.dtype).str,
        sample_count=len(loaded.values),
        tensor_shape=tensor_shape,
        feature_metadata=feature_metadata,
    )


def create_representation_tensor_sample(values, spec, sample_index):
    """Construct one unstandardized representation tensor.

    CNN datasets call this function for one requested row before adding the
    channel axis and applying training-derived scaling. Returning an owned
    float32 array prevents later tensor operations from mutating the
    read-only memory map.

    Parameters
    ----------
    values : numpy.ndarray
        Stored raw ``[sample, vx, vy, vz]`` array or Hermite
        ``[sample, n_x, n_y, n_z]`` array when unrotated, or
        ``[sample, n_parallel, n_perp1, n_perp2]`` when rotated.
    spec : RepresentationTensorSpec
        Sample-wise representation description.
    sample_index : int
        Zero-based source-array row index.

    Returns
    -------
    numpy.ndarray
        Writable C-contiguous float32 tensor without sample or channel
        axes.
    """

    sample_index = int(sample_index)
    if spec.representation == "raw":
        metadata = spec.feature_metadata
        tensor = prepare_raw_input(
            values[sample_index],
            log_eps=float(metadata["log_floor"]),
        )
    else:
        tensor = values[sample_index]
    return np.array(
        tensor,
        dtype=np.float32,
        order="C",
        copy=True,
    )
