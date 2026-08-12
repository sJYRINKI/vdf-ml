"""Load and scale sample-aligned plasma context for ML models.

Every current dataset stores ``plasma_context.npy`` as a float32 array with
shape ``(n_samples, 16)``. Rows align exactly with ``X.npy``, optional
``X_hermite.npy``, and ``metadata.csv``. The fixed feature order is imported
from the physics owner and contains Cartesian magnetic-field components in
teslas, electric-field components in volts per metre, configured-population
fluid bulk-velocity components in metres per second, number density in
particles per cubic metre, and pressure components ``(Pxx, Pyy, Pzz, Pxy,
Pxz, Pyz)`` in pascals. No redundant vector magnitudes are stored.

CNN and autoencoder datasets use process-local read-only memory maps so each
PyTorch DataLoader worker reads selected rows on demand without copying the
complete context array. A single scaler implementation fits feature means and
scales from chronological training rows only, then supplies aligned float32
context tensors to either model. DataLoader worker processes are independent
of dataset timestep extraction workers and one-process model-stage GPU
parallelism.
"""

from dataclasses import dataclass
import os
from pathlib import Path

import numpy as np

from src.physics.plasma_context import PLASMA_CONTEXT_FEATURE_NAMES


PLASMA_CONTEXT_FILENAME = "plasma_context.npy"


class PlasmaContextMemmapReader:
    """Read aligned plasma-context rows in one process.

    The reader opens the current dataset array only when a sample is first
    requested and reuses that read-only mapping for later samples in the same
    process. Pickling clears the open mapping so every PyTorch DataLoader
    worker owns its file descriptor instead of inheriting the training
    process's mapping.

    Parameters
    ----------
    source_path : str or pathlib.Path
        Path to ``plasma_context.npy`` with saved shape ``(n_samples, 16)``
        and float32 dtype.

    Notes
    -----
    Row ``i`` contains the context for representation row ``i`` and metadata
    row ``i``. Each returned row follows ``PLASMA_CONTEXT_FEATURE_NAMES``;
    it contains B, E, and fluid-velocity components in Cartesian order,
    density, and the canonical pressure order
    ``(Pxx, Pyy, Pzz, Pxy, Pxz, Pyz)``.
    """

    def __init__(self, source_path):
        """Create a closed reader for one current context array."""

        self.source_path = Path(source_path)
        self._values = None
        self._process_id = None

    def read(self, sample_index):
        """Read one saved context row without modifying the dataset.

        Parameters
        ----------
        sample_index : int
            Zero-based dataset row shared by the VDF representation,
            context, and metadata.

        Returns
        -------
        numpy.ndarray
            Read-only float32 view with shape ``(16,)`` in the fixed plasma
            context feature order.
        """

        return np.asarray(self._open()[int(sample_index)])

    def close(self):
        """Close the process-local memory map owned by this reader."""

        if self._values is not None:
            mmap_object = getattr(self._values, "_mmap", None)
            if mmap_object is not None:
                mmap_object.close()
        self._values = None
        self._process_id = None

    def __enter__(self):
        """Return this reader for bounded memory-map ownership."""

        return self

    def __exit__(self, exception_type, exception, traceback):
        """Close the context mapping when a bounded read block ends."""

        self.close()

    def __getstate__(self):
        """Return DataLoader-worker state without an open file mapping."""

        state = self.__dict__.copy()
        state["_values"] = None
        state["_process_id"] = None
        return state

    def _open(self):
        """Open or reuse the read-only context array for this process."""

        process_id = os.getpid()
        if self._values is not None and self._process_id == process_id:
            return self._values
        self.close()
        self._values = np.load(
            self.source_path,
            mmap_mode="r",
            allow_pickle=False,
        )
        self._process_id = process_id
        return self._values


@dataclass(frozen=True)
class PlasmaContextScaler:
    """Store training-only normalization for 16 plasma features.

    Mean and population standard deviation are fitted independently for the
    fixed vector-component, density, and pressure features using only
    chronological training rows. Validation, test, and prediction values use
    those same statistics. Scaling never changes ``plasma_context.npy`` and
    never appends redundant magnetic-field, electric-field, or bulk-speed
    magnitudes.

    Parameters
    ----------
    mean : numpy.ndarray
        Float64 training means with shape ``(16,)``.
    scale : numpy.ndarray
        Float64 training population standard deviations with shape ``(16,)``.
    """

    mean: np.ndarray
    scale: np.ndarray

    def __post_init__(self):
        """Copy scaler statistics into owned float64 arrays."""

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

    @classmethod
    def fit(cls, source_path, train_indices):
        """Fit context scaling from selected training samples only.

        CNN and autoencoder orchestration call this after chronological
        timestep splitting. The calculation opens the current context
        array read-only and uses no validation or test rows, preventing
        physical-value leakage into model preprocessing.

        Parameters
        ----------
        source_path : str or pathlib.Path
            Float32 ``plasma_context.npy`` path included in the dataset.
        train_indices : sequence of int
            Dataset rows assigned to the chronological training partition.

        Returns
        -------
        PlasmaContextScaler
            Float64 per-feature mean and population standard deviation in
            ``PLASMA_CONTEXT_FEATURE_NAMES`` order.

        Notes
        -----
        Only the selected ``(n_train, 16)`` values are materialized. This is
        small compared with one complete raw VDF volume and keeps fitting
        independent from DataLoader worker and GPU model parallelism.
        """

        indices = np.asarray(train_indices, dtype=np.int64)
        with PlasmaContextMemmapReader(source_path) as reader:
            values = np.asarray(
                reader._open()[indices],
                dtype=np.float64,
            )
        return cls(
            mean=np.mean(values, axis=0, dtype=np.float64),
            scale=np.std(values, axis=0, dtype=np.float64),
        )

    def transform(self, values):
        """Scale one row or a batch with training-derived statistics.

        Parameters
        ----------
        values : numpy.ndarray
            Physical context with shape ``(16,)`` or ``(batch, 16)`` in the
            fixed feature order and SI units.

        Returns
        -------
        numpy.ndarray
            Float32 scaled context with the same shape. Conversion occurs
            once after float64 subtraction and division.
        """

        scaled = (
            np.asarray(values, dtype=np.float64) - self.mean
        ) / self.scale
        return scaled.astype(np.float32)

    def to_dict(self):
        """Return direct checkpoint values for this training scaler.

        Returns
        -------
        dict
            Float64 feature means and scales as primitive lists.
        """

        return {
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
        }

    @classmethod
    def from_dict(cls, record):
        """Reconstruct context scaling from one current checkpoint.

        Parameters
        ----------
        record : mapping
            Stored ``mean`` and ``scale`` sequences in fixed feature order.

        Returns
        -------
        PlasmaContextScaler
            Training-derived 16-feature context scaling used for inference.
        """

        return cls(
            mean=np.asarray(record["mean"], dtype=np.float64),
            scale=np.asarray(record["scale"], dtype=np.float64),
        )


__all__ = [
    "PLASMA_CONTEXT_FEATURE_NAMES",
    "PLASMA_CONTEXT_FILENAME",
    "PlasmaContextMemmapReader",
    "PlasmaContextScaler",
]
