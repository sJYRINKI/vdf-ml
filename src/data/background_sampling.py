"""Assign the historical magnetotail background to ``current_layer``.

Stage 3 calls this module after manual-coordinate and X/O-point samples have
been planned. It selects VDF-cell centres inside the configured magnetotail
box, excludes every cell already assigned or rejected by the more specific
classes, and appends the remaining cells in ascending source-CID order.

The selector is geometric only. It does not read or calculate current
density.
"""

import numpy as np

from src.data.step_02_find_vdf_cells import create_region_mask_re


def create_current_layer_sample_specs(
    config,
    file_location,
    existing_sample_specs,
    timestep,
    rejected_cellids,
    vdf_cellids,
    vdf_coords_re,
):
    """Create ordered ``current_layer`` records from unassigned tail cells.

    This operation follows manual and unchanged X/O selection in Stage 3.
    Existing and rejected CIDs are removed before the historical magnetotail
    background is assigned, so an X/O or manual classification is never
    replaced by the broader ``current_layer`` class.

    Parameters
    ----------
    config : dict
        Extraction configuration containing the authoritative class IDs and
        the ``points.regions_re.tail`` bounds in Earth radii.
    file_location : str or pathlib.Path
        Timestep VLSV path retained in each extraction specification.
    existing_sample_specs : list of dict
        Manual and X/O sample records already accepted by Stage 3.
    timestep : int
        Simulation timestep identifier.
    rejected_cellids : iterable of int
        CIDs excluded by unchanged point and cross-class conflict handling.
    vdf_cellids : numpy.ndarray
        VDF-carrying CIDs in stable ascending order.
    vdf_coords_re : numpy.ndarray
        Cell-centre coordinates with shape ``(n_cells, 3)`` and axis order
        ``[x_re, y_re, z_re]``.

    Returns
    -------
    list of dict
        ``current_layer`` extraction records in ascending candidate-CID
        order, ready to append after the specific classes.

    Notes
    -----
    The historical rule is an inclusive spatial fallback, not a
    current-density detector. The shipped magnetotail has no y bound.
    """

    class_name = "current_layer"
    label = int(config["labels"][class_name])
    excluded_cellids = {
        int(sample_spec["cid"])
        for sample_spec in existing_sample_specs
    }
    excluded_cellids.update(int(cid) for cid in rejected_cellids)
    coordinates_by_cid = {
        int(cid): tuple(float(value) for value in coord_re)
        for cid, coord_re in zip(vdf_cellids, vdf_coords_re)
    }
    return [
        {
            "file_location": file_location,
            "cid": int(cid),
            "label": label,
            "class_name": class_name,
            "coord_re": coordinates_by_cid[int(cid)],
            "vdf_coord_re": coordinates_by_cid[int(cid)],
            "timestep": int(timestep),
        }
        for cid in select_current_layer_cells(
            config,
            vdf_cellids,
            vdf_coords_re,
        )
        if int(cid) not in excluded_cellids
    ]


def select_current_layer_cells(config, vdf_cellids, vdf_coords_re):
    """Select VDF cells in the configured historical magnetotail region.

    Stage 3 calls this selector after the unchanged manual and X/O
    calculations and before adding point-reference metadata. It applies the
    inclusive ``points.regions_re.tail`` mask to the already sorted VDF-cell
    arrays and therefore preserves ascending CID order.

    Parameters
    ----------
    config : dict
        Extraction configuration containing the magnetotail bounds.
    vdf_cellids : numpy.ndarray
        VDF-carrying spatial cell IDs in ascending order.
    vdf_coords_re : numpy.ndarray
        Aligned cell-centre coordinates with shape ``(n_cells, 3)`` in Earth
        radii and ``[x, y, z]`` order.

    Returns
    -------
    list of int
        Candidate magnetotail CIDs in the same order as ``vdf_cellids``.

    Notes
    -----
    The current-layer class is the historical unassigned-tail fallback.
    Current density is neither an input nor a selection criterion; X/O and
    manual CIDs are excluded by the caller before records are returned.
    """

    region_re = config["points"]["regions_re"]["tail"]
    selected = create_region_mask_re(
        np.asarray(vdf_coords_re, dtype=float),
        region_re,
    )
    return [
        int(cid)
        for cid in np.asarray(vdf_cellids, dtype=int)[selected]
    ]


__all__ = [
    "create_current_layer_sample_specs",
    "select_current_layer_cells",
]
