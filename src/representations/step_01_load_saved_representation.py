"""Stage 1: load one saved raw or Hermite representation.

This stage opens the dataset files before representation-specific numerical
preparation. The raw and Hermite preparation stages follow it and consume the
returned memory-mapped array together with aligned metadata.

The separation keeps storage ownership distinct from raw logarithms and
Hermite tensor construction. Input rows retain the saved dataset order.
"""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.load_velocity_grid import load_velocity_grid


PUBLIC_REPRESENTATIONS = ("raw", "hermite")
RAW_AXIS_ORDER = ("vx", "vy", "vz")


@dataclass
class LoadedRepresentation:
    """Hold one directly loaded read-only representation.

    The stage-1 loader creates this short-lived resource owner before raw or
    Hermite preparation begins. The metadata frame and representation array
    share saved row order; only the representation array owns an open NumPy
    memory map.

    Parameters
    ----------
    dataset_dir : pathlib.Path
        Directory containing the stored dataset.
    representation : str
        Public representation name, either ``"raw"`` or ``"hermite"``.
    values : numpy.ndarray
        Read-only memory-mapped representation array.
    metadata : pandas.DataFrame
        Saved metadata in exact sample order.
    velocity_grid : dict or None
        Raw velocity-grid descriptor. Hermite loading does not open it.
    source_filename : str
        Current saved representation filename.

    Notes
    -----
    Use the instance as a context manager or call :meth:`close` to release
    the underlying NumPy memory map deterministically.
    """

    dataset_dir: Path
    representation: str
    values: np.ndarray
    metadata: object
    velocity_grid: dict | None
    source_filename: str
    _owned_arrays: tuple = field(default=(), repr=False)
    _closed: bool = field(default=False, init=False, repr=False)

    def close(self):
        """Close every memory map owned by this loaded representation.

        This is the terminal storage action for representation consumers.
        It releases each distinct NumPy mapping once while leaving the loaded
        pandas metadata and plain velocity-grid values usable.
        """

        closed_mmaps = set()
        for array in self._owned_arrays:
            mmap_object = getattr(array, "_mmap", None)
            if mmap_object is not None and id(mmap_object) not in closed_mmaps:
                mmap_object.close()
                closed_mmaps.add(id(mmap_object))
        self._closed = True

    def __enter__(self):
        """Return the loaded representation for a managed workflow scope."""

        return self

    def __exit__(self, exception_type, exception, traceback):
        """Close owned memory maps when the managed workflow scope ends."""

        self.close()
        return False


def load_saved_representation(dataset_dir, representation):
    """Load one stored raw or Hermite representation directly.

    This is the storage stage shared by PCA, CNN, and autoencoder workflows.
    It opens metadata and the requested array in saved sample order; raw
    loading also reads the velocity grid retained for workflows such as the
    autoencoder and checkpoint creation. Numerical preparation is
    deliberately deferred to the following branch.

    Parameters
    ----------
    dataset_dir : str or pathlib.Path
        Directory containing ``metadata.csv`` and the requested saved
        representation.
    representation : {"raw", "hermite"}
        Public representation to open.

    Returns
    -------
    LoadedRepresentation
        Read-only representation array, aligned metadata, and raw
        velocity-grid context when applicable.
    """

    dataset_dir = Path(dataset_dir)
    metadata = pd.read_csv(dataset_dir / "metadata.csv")
    velocity_grid = (
        load_velocity_grid(dataset_dir)
        if representation == "raw"
        else None
    )
    source_filename = (
        "X.npy" if representation == "raw" else "X_hermite.npy"
    )
    values = np.load(
        dataset_dir / source_filename,
        mmap_mode="r",
        allow_pickle=False,
    )
    return LoadedRepresentation(
        dataset_dir=dataset_dir,
        representation=representation,
        values=values,
        metadata=metadata,
        velocity_grid=velocity_grid,
        source_filename=source_filename,
        _owned_arrays=(values,),
    )
