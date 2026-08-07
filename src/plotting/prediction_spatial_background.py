"""Resolve spatial-mesh and magnetic-streamline inputs for VLSV maps.

Prediction and dataset figures share these source-reader queries with the
single bulk-x-velocity renderer in ``dataset_selection_colormap``. This
module does not choose an alternate density background.
"""

import numpy as np


def spatial_mesh_shape(reader):
    """
    Read an optional three-dimensional spatial mesh shape.

    The shape is used only to choose between Analysator's 2-D colormap and
    3-D y-slice renderer. Values are returned exactly as integer dimensions
    exposed by the source reader.

    Parameters
    ----------
    reader : object
        Open VLSV reader that may expose ``get_spatial_mesh_size``.

    Returns
    -------
    tuple of int or None
        ``(x, y, z)`` mesh shape, or ``None`` when unavailable.
    """

    get_shape = getattr(reader, "get_spatial_mesh_size", None)
    if not callable(get_shape):
        return None
    values = np.asarray(get_shape(), dtype=np.int64)
    return tuple(int(value) for value in values)


def _resolve_streamline_variable(reader):
    """Return the first available magnetic streamline variable.

    Parameters
    ----------
    reader : object
        Open VLSV reader exposing its variable names.

    Returns
    -------
    str
        First available field in the established volume/legacy preference
        order.
    """

    available = set(reader.get_all_variables())
    return next(
        variable
        for variable in ("vg_b_vol", "B_vol", "B")
        if variable in available
    )
