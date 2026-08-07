"""Map physical dataset classes to stable CNN output indices.

The explicit numeric class order is shared by training, checkpoint loading,
and prediction. Class names never determine model-output order.
"""

from dataclasses import dataclass
from types import MappingProxyType

import numpy as np


@dataclass(frozen=True)
class ClassMapping:
    """Map project class IDs to contiguous CNN output indices.

    CNN logits require contiguous zero-based columns, while the scientific
    class IDs stored in ``metadata.csv`` need not be contiguous. Training
    builds this immutable mapping once, saves it in the checkpoint, and
    prediction uses the same order to translate ``argmax`` results back to
    physical classes.

    Parameters
    ----------
    class_ids : tuple of int
        Project class IDs in explicit model-output order.
    class_names : tuple of str
        Physical class names aligned with ``class_ids``.

    Attributes
    ----------
    class_ids : tuple of int
        Project class IDs in model-output order.
    class_names : tuple of str
        Physical names in the same order.

    Notes
    -----
    Class names describe output columns but never determine their order.
    """

    class_ids: tuple
    class_names: tuple

    def __post_init__(self):
        """Store immutable integer IDs and string names in output order."""

        object.__setattr__(
            self,
            "class_ids",
            tuple(int(class_id) for class_id in self.class_ids),
        )
        object.__setattr__(
            self,
            "class_names",
            tuple(str(class_name) for class_name in self.class_names),
        )

    @classmethod
    def from_metadata(cls, metadata, class_id_order=None):
        """Build the model-output mapping from saved metadata.

        Stage 1 calls this while loading CNN targets. Explicit
        ``class_id_order`` preserves a caller-selected output contract;
        otherwise numeric project IDs determine the stable logit order.

        Parameters
        ----------
        metadata : pandas.DataFrame
            Dataset rows containing ``class_id`` and ``class_name``.
        class_id_order : sequence of int, optional
            Explicit output order. Project IDs are sorted numerically when
            no order is supplied.

        Returns
        -------
        ClassMapping
            Project-ID and output-index mapping.
        """

        metadata_mapping = _mapping_from_metadata(metadata)
        ordered_ids = (
            tuple(sorted(metadata_mapping))
            if class_id_order is None
            else tuple(int(value) for value in class_id_order)
        )
        return cls(
            class_ids=ordered_ids,
            class_names=tuple(metadata_mapping[value] for value in ordered_ids),
        )

    @property
    def project_id_to_output_index(self):
        """Return the immutable project-ID to output-index mapping.

        Training target encoding and external callers share this view so
        no code infers output order from class names.

        Returns
        -------
        types.MappingProxyType
            Physical project IDs mapped to contiguous model indices.
        """

        return MappingProxyType(
            {
                class_id: output_index
                for output_index, class_id in enumerate(self.class_ids)
            }
        )

    @property
    def output_index_to_project_id(self):
        """Return project class IDs in CNN output-index order.

        Prediction uses this order to decode the largest class-logit column
        back to the dataset's scientific class identifier.

        Returns
        -------
        tuple of int
            Physical project IDs aligned with class logits.
        """

        return self.class_ids

    def encode(self, project_class_ids):
        """Convert project class IDs to contiguous output indices.

        Stage 1 applies this conversion before PyTorch loss calculation.
        The input shape and row order are preserved.

        Parameters
        ----------
        project_class_ids : array-like
            Physical project class IDs in any array shape.

        Returns
        -------
        numpy.ndarray
            Integer output indices with the input shape.
        """

        values = np.asarray(project_class_ids, dtype=np.int64)
        mapping = self.project_id_to_output_index
        return np.asarray(
            [mapping[int(value)] for value in values.ravel()],
            dtype=np.int64,
        ).reshape(values.shape)

    def decode(self, model_output_indices):
        """Convert output indices to physical project class IDs.

        Evaluation and prediction call this inverse mapping after selecting
        CNN output columns. The input shape and ordering are preserved.

        Parameters
        ----------
        model_output_indices : array-like
            Contiguous class-logit indices in any array shape.

        Returns
        -------
        numpy.ndarray
            Physical project class IDs with the input shape.
        """

        values = np.asarray(model_output_indices, dtype=np.int64)
        return np.asarray(self.class_ids, dtype=np.int64)[values]

    def to_dict(self):
        """Return the class mapping as checkpoint data.

        Stage 8 stores this primitive representation so prediction can
        recover the exact class-logit order without pickling this class.

        Returns
        -------
        dict
            Physical IDs, names, and explicit output-index mapping.
        """

        return {
            "class_ids": list(self.class_ids),
            "class_names": list(self.class_names),
            "project_class_id_to_model_output_index": {
                str(class_id): output_index
                for output_index, class_id in enumerate(self.class_ids)
            },
        }

    @classmethod
    def from_dict(cls, record):
        """Reconstruct a class mapping from checkpoint data.

        Checkpoint loading uses the saved ID and name order directly before
        evaluation or prediction decodes model outputs.

        Parameters
        ----------
        record : dict
            Stored physical class IDs and names.

        Returns
        -------
        ClassMapping
            Class mapping in saved model-output order.
        """

        return cls(
            class_ids=tuple(record["class_ids"]),
            class_names=tuple(record["class_names"]),
        )


def as_integer_array(values):
    """Convert values to a NumPy integer array.

    CNN stage 1 uses this small conversion at the metadata boundary so
    PyTorch class and identity tensors share a stable ``int64`` dtype.

    Parameters
    ----------
    values : array-like
        Integer-like values in any shape.

    Returns
    -------
    numpy.ndarray
        Values converted to ``int64``.
    """

    return np.asarray(values, dtype=np.int64)


def _mapping_from_metadata(metadata):
    """Associate each physical class ID with its first saved class name.

    Parameters
    ----------
    metadata : pandas.DataFrame
        Saved rows containing aligned ``class_id`` and ``class_name``.

    Returns
    -------
    dict
        Integer class IDs mapped to the first corresponding saved name.
    """

    class_ids = np.asarray(metadata["class_id"], dtype=np.int64)
    class_names = metadata["class_name"].astype(str).to_numpy()
    unique_ids, first_indices = np.unique(class_ids, return_index=True)
    return {
        int(class_id): str(class_names[first_index])
        for class_id, first_index in zip(unique_ids, first_indices)
    }


__all__ = ["ClassMapping", "as_integer_array"]
