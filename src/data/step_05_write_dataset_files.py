"""Stage 5: write and rename the completed dataset directory.

This final extraction stage follows VDF sample streaming. It allocates raw
and optional Hermite NumPy memory maps in one hidden sibling directory,
writes their aligned rows, flushes and closes the arrays, writes metadata
and the velocity grid, and renames the staging directory to the requested
dataset path.

Inputs are array sources or one sequential sample callback. The returned
path contains only ``X.npy``, optional ``X_hermite.npy``, ``metadata.csv``,
and ``velocity_grid.npz``.
"""

from pathlib import Path
import tempfile

import numpy as np

from src.data.load_velocity_grid import (
    VELOCITY_GRID_FILENAME,
    VELOCITY_GRID_KEYS,
)


def write_dataset(
    dataset_dir,
    *,
    velocity_grid,
    raw=None,
    metadata=None,
    hermite=None,
    raw_shape=None,
    raw_dtype=None,
    hermite_shape=None,
    hermite_dtype=None,
    sample_writer=None,
):
    """Stage and write one current-format dataset.

    This final dataset stage owns all filesystem side effects. It creates a
    hidden sibling directory, allocates raw and optional Hermite NumPy memory
    maps, accepts either existing arrays or a sequential writer callback,
    flushes and closes the completed mappings, writes aligned metadata and
    velocity-grid files, and renames the completed directory to
    ``dataset_dir``.

    Parameters
    ----------
    dataset_dir : str or pathlib.Path
        Final dataset directory.
    velocity_grid : dict
        Velocity-grid descriptor with ``[vx, vy, vz]`` shape and bounds in
        metres per second.
    raw : array-like, optional
        Existing VDF source with shape ``(samples, vx, vy, vz)``.
    metadata : pandas.DataFrame, optional
        Metadata rows aligned with the array sources.
    hermite : array-like, optional
        Existing physical-VDF coefficients with shape
        ``(samples, order, order, order)``. Axes are ``(n_x, n_y, n_z)``
        without rotation or ``(n_parallel, n_perp1, n_perp2)`` after optional
        rotation.
    raw_shape : tuple of int, optional
        Raw output shape for callback-based writing.
    raw_dtype : data-type, optional
        Raw output dtype for callback-based writing.
    hermite_shape : tuple of int, optional
        Hermite output shape for callback-based writing.
    hermite_dtype : data-type, optional
        Hermite output dtype for callback-based writing.
    sample_writer : callable, optional
        Callback receiving writable raw and optional Hermite arrays and
        returning aligned metadata.

    Returns
    -------
    pathlib.Path
        Renamed final dataset directory.
    """

    target = Path(dataset_dir).absolute()
    callback_mode = sample_writer is not None
    if callback_mode:
        raw_shape = tuple(raw_shape)
        hermite_shape = (
            tuple(hermite_shape)
            if hermite_shape is not None
            else None
        )
    else:
        raw_shape = tuple(raw.shape)
        raw_dtype = raw.dtype
        if hermite is not None:
            hermite_shape = tuple(hermite.shape)
            hermite_dtype = hermite.dtype
    target.parent.mkdir(parents=True, exist_ok=True)
    staging_path = Path(
        tempfile.mkdtemp(
            prefix=f".{target.name}.staging-",
            dir=target.parent,
        )
    )
    raw_output = _create_memmap(
        staging_path / "X.npy",
        raw_shape,
        raw_dtype,
    )
    hermite_output = (
        _create_memmap(
            staging_path / "X_hermite.npy",
            hermite_shape,
            hermite_dtype,
        )
        if hermite_shape is not None
        else None
    )
    if callback_mode:
        metadata = sample_writer(raw_output, hermite_output)
    else:
        raw_output[:] = raw
        if hermite_output is not None:
            hermite_output[:] = hermite
    arrays = [raw_output]
    if hermite_output is not None:
        arrays.append(hermite_output)
    for array in arrays:
        array.flush()
    for array in reversed(arrays):
        array._mmap.close()
    metadata.to_csv(staging_path / "metadata.csv", index=False)
    _save_velocity_grid(staging_path, velocity_grid)
    staging_path.rename(target)
    return target


def _create_memmap(path, shape, dtype):
    """Allocate a writable ``.npy`` memory map for one staged array.

    Stage 5 returns the live mapping to either direct array assignment or the
    sequential sample callback. The owning writer flushes and closes it before
    the staging directory is renamed.
    """

    return np.lib.format.open_memmap(
        path,
        mode="w+",
        dtype=dtype,
        shape=shape,
    )


def _save_velocity_grid(dataset_dir, velocity_grid):
    """Save the shared physical velocity-grid descriptor as one NPZ archive.

    Only keys consumed by current representation and plotting workflows are
    written. Array shape uses ``[vx, vy, vz]`` order and extents remain in
    metres per second.
    """

    np.savez(
        Path(dataset_dir) / VELOCITY_GRID_FILENAME,
        **{key: velocity_grid[key] for key in VELOCITY_GRID_KEYS},
    )
