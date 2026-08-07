"""Load the saved velocity grid used by raw representations and plots.

The archive contains the ``[vx, vy, vz]`` mesh shape, physical extents in
metres per second, population, axis names, and middle-``vy`` slice index.
Dataset writing owns the corresponding save operation.
"""

from pathlib import Path

import numpy as np


VELOCITY_GRID_FILENAME = "velocity_grid.npz"
VELOCITY_GRID_KEYS = (
    "shape",
    "extent_mps",
    "population",
    "axis_order",
    "slice_axis",
    "slice_index",
)


def load_velocity_grid(dataset_dir):
    """Load one dataset-level velocity-grid descriptor.

    Raw representation preparation and physical VDF plotting call this loader
    after opening their dataset. Values are copied out of the temporary NPZ
    archive so the returned descriptor owns no open file handle.

    Parameters
    ----------
    dataset_dir : str or pathlib.Path
        Dataset directory containing ``velocity_grid.npz``.

    Returns
    -------
    dict
        Velocity-grid shape and axes with bounds in metres per second.
    """

    with np.load(
        Path(dataset_dir) / VELOCITY_GRID_FILENAME,
        allow_pickle=False,
    ) as values:
        return {
            "shape": np.asarray(values["shape"], dtype=np.int64),
            "extent_mps": np.asarray(
                values["extent_mps"],
                dtype=np.float64,
            ),
            "population": str(np.asarray(values["population"]).item()),
            "axis_order": np.asarray(values["axis_order"]),
            "slice_axis": str(np.asarray(values["slice_axis"]).item()),
            "slice_index": int(np.asarray(values["slice_index"]).item()),
        }
