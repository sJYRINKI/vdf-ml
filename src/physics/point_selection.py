"""Select accepted VDF cells around already-detected X/O source points.

Stage 3 calls this policy layer after topology detection and VDF-cell
discovery. Manual Earth-radius boxes and physical X/O geometry are combined
with the configured ``manual``, ``physical``, ``consensus``, or ``union``
method in deterministic cell order. Empty results are omitted, then complete
X and O source records that claim any common stored VDF cell are ejected.

This module does not detect magnetic critical points. Raw detector records
remain unchanged and continue to own their Hessian, ion-scale, flux, and
island-contour calculations.
"""

from src.physics.point_selection_geometry import (
    get_point_selection_config,
    get_vdf_cellids_in_flux_contour,
    get_vdf_cellids_in_hessian_di_box,
    get_vdf_cellids_in_manual,
)


def is_point_record(labeled_coord):
    """Return whether a labeled entry is a detected X/O point record.

    Sample planning accepts both configured coordinate tuples and topology
    dictionaries. This discriminator selects the point-geometry branch
    without changing either input form.

    Parameters
    ----------
    labeled_coord : object
        Static coordinate tuple or candidate point mapping.

    Returns
    -------
    bool
        Whether the entry is a point record.
    """

    return (
        isinstance(labeled_coord, dict)
        and labeled_coord.get("is_point_record")
    )


def unpack_labeled_coord(labeled_coord):
    """Return class name, ID, and Earth-radius coordinate from one entry.

    Stage 3 uses this common projection before mapping either static or
    detected labels to VDF cells, preserving the original coordinate object
    and configured class identity.

    Parameters
    ----------
    labeled_coord : tuple or dict
        Static ``(class_name, class_id, coord_re)`` entry or point record.

    Returns
    -------
    tuple
        ``(class_name, class_id, coord_re)`` without changing coordinates.
    """

    if is_point_record(labeled_coord):
        return (
            labeled_coord["class_name"],
            int(labeled_coord["label"]),
            labeled_coord["coord_re"],
        )
    class_name, label, coord_re = labeled_coord
    return class_name, int(label), coord_re


def get_point_cellids_by_position(
    config,
    point_record,
    vdf_cellids,
    vdf_coords_re,
):
    """Return only selected cell IDs for one detected critical point.

    Callers that do not need rejection or agreement metadata use this view of
    the complete selection result. Position keys retain the geometry-specific
    deterministic order used when samples are appended.

    Parameters
    ----------
    config : dict
        Dataset point-selection configuration.
    point_record : dict
        Detected X/O point record.
    vdf_cellids : numpy.ndarray
        VDF-carrying spatial cell identifiers.
    vdf_coords_re : numpy.ndarray
        Cell centres in Earth radii with shape ``(n_cells, 3)``.

    Returns
    -------
    dict
        Selection position names mapped to VDF cell IDs.
    """

    return get_point_selection_result(
        config=config,
        point_record=point_record,
        vdf_cellids=vdf_cellids,
        vdf_coords_re=vdf_coords_re,
    )["cellids_by_position"]


def get_point_selection_result(
    config,
    point_record,
    vdf_cellids,
    vdf_coords_re,
):
    """Apply one configured cell-selection method to a detected point.

    The source point is the continuous X/O coordinate produced by topology
    detection. Manual and physical searches select stored VDF cells around
    that source. ``manual`` and ``physical`` use one set, ``consensus`` uses
    their intersection, and ``union`` uses both sets. Mapping iteration keeps
    the historical physical/VDF order rather than arbitrary set order.

    Parameters
    ----------
    config : dict
        Dataset point-selection configuration.
    point_record : dict
        Detected X/O point record with its physical geometry.
    vdf_cellids : numpy.ndarray
        VDF-carrying spatial cell identifiers.
    vdf_coords_re : numpy.ndarray
        Cell centres in Earth radii with shape ``(n_cells, 3)``.

    Returns
    -------
    dict
        Source record, separate manual and physical selections, ordered final
        cells, rejected disagreement cells, and per-position metadata.

    """

    point_kind = point_record["point_kind"]
    selection_method = get_point_selection_method(
        config=config,
        point_kind=point_kind,
    )
    manual_cellids_by_position = get_vdf_cellids_in_manual(
        config=config,
        point_record=point_record,
        vdf_cellids=vdf_cellids,
        vdf_coords_re=vdf_coords_re,
    )
    if selection_method == "manual":
        return create_point_selection_result(
            point_record=point_record,
            cellids_by_position=manual_cellids_by_position,
            manual_cellids_by_position=manual_cellids_by_position,
        )

    physical_cellids_by_position = get_physical_point_cellids_by_position(
        config=config,
        point_record=point_record,
        vdf_cellids=vdf_cellids,
        vdf_coords_re=vdf_coords_re,
    )
    if selection_method == "physical":
        return create_point_selection_result(
            point_record=point_record,
            cellids_by_position=physical_cellids_by_position,
            manual_cellids_by_position=manual_cellids_by_position,
            physical_cellids_by_position=physical_cellids_by_position,
        )
    if selection_method == "consensus":
        return create_consensus_point_selection_result(
            point_record=point_record,
            physical_cellids_by_position=physical_cellids_by_position,
            manual_cellids_by_position=manual_cellids_by_position,
        )
    if selection_method == "union":
        return create_union_point_selection_result(
            point_record=point_record,
            physical_cellids_by_position=physical_cellids_by_position,
            manual_cellids_by_position=manual_cellids_by_position,
        )


def get_physical_point_cellids_by_position(
    config,
    point_record,
    vdf_cellids,
    vdf_coords_re,
):
    """Select cells with X Hessian or O island-contour geometry.

    The critical-point kind determines which scientific selection owns the
    sample: X points use local Hessian eigenvectors scaled by ion inertial
    length, whereas O points use the saved closed island contour.

    Parameters
    ----------
    config : dict
        Dataset point-selection configuration.
    point_record : dict
        Detected X/O point record.
    vdf_cellids : numpy.ndarray
        VDF-carrying spatial cell identifiers.
    vdf_coords_re : numpy.ndarray
        Cell centres in Earth radii with shape ``(n_cells, 3)``.

    Returns
    -------
    dict
        Physical selection position names mapped to VDF cell IDs.

    """

    point_kind = point_record["point_kind"]
    if point_kind == "x":
        return get_vdf_cellids_in_hessian_di_box(
            config=config,
            point_record=point_record,
            vdf_cellids=vdf_cellids,
            vdf_coords_re=vdf_coords_re,
        )
    return get_vdf_cellids_in_flux_contour(
        config=config,
        point_record=point_record,
        vdf_cellids=vdf_cellids,
        vdf_coords_re=vdf_coords_re,
    )


def create_point_selection_result(
    cellids_by_position,
    point_record=None,
    manual_cellids_by_position=None,
    physical_cellids_by_position=None,
    rejected_cellids=None,
    metadata_by_position=None,
):
    """Package one point-selection result with stable selected-cell order.

    Manual, physical, consensus, and union branches use this common return
    shape so stage-3 planning can consume selected positions, rejected cell
    IDs, and optional descriptive metadata identically. The conversion does
    not perform a scientific validity check.

    Parameters
    ----------
    cellids_by_position : dict
        Selection positions mapped to VDF cell IDs.
    point_record : dict, optional
        Unmodified detected source-point record that owns this selection.
    manual_cellids_by_position : dict, optional
        Manual-box positions mapped to VDF cell IDs.
    physical_cellids_by_position : dict, optional
        Physical X-box or O-contour positions mapped to VDF cell IDs.
    rejected_cellids : iterable of int, optional
        Cells excluded from all physical classes.
    metadata_by_position : dict, optional
        Per-position agreement and plot metadata.

    Returns
    -------
    dict
        Normalized selected cells, rejected cells, and metadata mappings.
    """

    selected_by_position = {
        position: int(cid)
        for position, cid in cellids_by_position.items()
    }
    manual_by_position = {
        position: int(cid)
        for position, cid in (manual_cellids_by_position or {}).items()
    }
    physical_by_position = {
        position: int(cid)
        for position, cid in (physical_cellids_by_position or {}).items()
    }
    return {
        "point_record": point_record,
        "cellids_by_position": selected_by_position,
        "selected_cellids": tuple(dict.fromkeys(selected_by_position.values())),
        "manual_cellids_by_position": manual_by_position,
        "manual_cellids": tuple(dict.fromkeys(manual_by_position.values())),
        "physical_cellids_by_position": physical_by_position,
        "physical_cellids": tuple(dict.fromkeys(physical_by_position.values())),
        "rejected_cellids": {
            int(cid)
            for cid in (rejected_cellids or set())
        },
        "metadata_by_position": metadata_by_position or {},
    }


def create_consensus_point_selection_result(
    point_record,
    physical_cellids_by_position,
    manual_cellids_by_position,
):
    """Keep cells selected by both physical and manual methods.

    Consensus treats physical/manual disagreement as ambiguity: only their
    cell-ID intersection becomes point samples, while the symmetric difference
    is rejected from every later physical class.

    Parameters
    ----------
    point_record : dict
        Detected source-point record that owns both selections.
    physical_cellids_by_position : dict
        Physical positions mapped to VDF cell IDs.
    manual_cellids_by_position : dict
        Manual positions mapped to VDF cell IDs.

    Returns
    -------
    dict
        Consensus result; disagreement cells are rejected from every class.
    """

    physical_positions_by_cellid = invert_cellids_by_position(
        physical_cellids_by_position
    )
    manual_positions_by_cellid = invert_cellids_by_position(
        manual_cellids_by_position
    )
    physical_cellids = set(physical_positions_by_cellid)
    manual_cellids = set(manual_positions_by_cellid)
    consensus_cellids = physical_cellids & manual_cellids
    rejected_cellids = (physical_cellids | manual_cellids) - consensus_cellids

    cellids_by_position = {}
    metadata_by_position = {}
    for position, cid in physical_cellids_by_position.items():
        cid = int(cid)
        if cid not in consensus_cellids:
            continue
        cellids_by_position[position] = cid
        metadata_by_position[position] = create_combined_selection_metadata(
            selection_agreement="both",
            physical_selected=True,
            manual_selected=True,
            plot_selection_method="physical",
        )
    return create_point_selection_result(
        point_record=point_record,
        cellids_by_position=cellids_by_position,
        manual_cellids_by_position=manual_cellids_by_position,
        physical_cellids_by_position=physical_cellids_by_position,
        rejected_cellids=rejected_cellids,
        metadata_by_position=metadata_by_position,
    )


def create_union_point_selection_result(
    point_record,
    physical_cellids_by_position,
    manual_cellids_by_position,
):
    """Keep the physical/manual union in historical deterministic order.

    Union retains every selected cell but emits a physical position and plot
    convention when both methods choose the same cell. Manual-only cells are
    appended afterward, preserving the historical physical-first order.

    Parameters
    ----------
    point_record : dict
        Detected source-point record that owns both selections.
    physical_cellids_by_position : dict
        Physical positions mapped to VDF cell IDs.
    manual_cellids_by_position : dict
        Manual positions mapped to VDF cell IDs.

    Returns
    -------
    dict
        Union result with deterministic physical-first ordering.
    """

    physical_positions_by_cellid = invert_cellids_by_position(
        physical_cellids_by_position
    )
    manual_positions_by_cellid = invert_cellids_by_position(
        manual_cellids_by_position
    )
    physical_cellids = set(physical_positions_by_cellid)
    manual_cellids = set(manual_positions_by_cellid)
    cellids_by_position = {}
    metadata_by_position = {}
    for position, cid in physical_cellids_by_position.items():
        cid = int(cid)
        manual_selected = cid in manual_cellids
        cellids_by_position[position] = cid
        metadata_by_position[position] = create_combined_selection_metadata(
            selection_agreement="both" if manual_selected else "physical_only",
            physical_selected=True,
            manual_selected=manual_selected,
            plot_selection_method="physical",
        )
    for position, cid in manual_cellids_by_position.items():
        cid = int(cid)
        if cid in physical_cellids:
            continue
        cellids_by_position[position] = cid
        metadata_by_position[position] = create_combined_selection_metadata(
            selection_agreement="manual_only",
            physical_selected=False,
            manual_selected=True,
            plot_selection_method="manual",
        )
    return create_point_selection_result(
        point_record=point_record,
        cellids_by_position=cellids_by_position,
        manual_cellids_by_position=manual_cellids_by_position,
        physical_cellids_by_position=physical_cellids_by_position,
        metadata_by_position=metadata_by_position,
    )


def create_accepted_point_selection_results(
    config,
    point_records,
    vdf_cellids,
    vdf_coords_re,
):
    """Select cells, omit empty points, and eject X/O conflicts.

    Selection is applied independently to each detected source point. The
    historical consensus disagreement cells are accumulated before empty
    results are omitted. Cross-kind conflicts are then evaluated from the
    final selected cells, and every cell belonging to an ejected complete
    source record is rejected from later background assignment.

    Parameters
    ----------
    config : dict
        Dataset point-selection configuration.
    point_records : iterable of dict
        Labeled detected X/O records in detector order.
    vdf_cellids : numpy.ndarray
        VDF-carrying spatial cell identifiers in stable dataset order.
    vdf_coords_re : numpy.ndarray
        Matching cell centres in Earth radii with shape ``(n_cells, 3)``.

    Returns
    -------
    dict
        Accepted selection results in source order and cell IDs excluded from
        later classes. An ejected CID retained by a surviving same-kind point
        remains owned by that survivor rather than becoming background. Each
        accepted in-memory point record carries the unchanged manual,
        physical, and final selected cell tuples needed by colormap plotting.
    """

    coord_by_cellid = {
        int(cid): tuple(float(value) for value in coord_re)
        for cid, coord_re in zip(vdf_cellids, vdf_coords_re)
    }
    selection_results = []
    rejected_cellids = set()
    for point_record in point_records:
        selection_result = get_point_selection_result(
            config=config,
            point_record=point_record,
            vdf_cellids=vdf_cellids,
            vdf_coords_re=vdf_coords_re,
        )
        rejected_cellids.update(selection_result["rejected_cellids"])
        selected_cellids = selection_result["selected_cellids"]
        selected_record = dict(point_record)
        selected_record["selected_cellids"] = selected_cellids
        selected_record["manual_cellids"] = selection_result[
            "manual_cellids"
        ]
        selected_record["physical_cellids"] = selection_result[
            "physical_cellids"
        ]
        selected_record["selected_vdf_coords_re"] = tuple(
            coord_by_cellid[cid]
            for cid in selected_cellids
        )
        selection_result["point_record"] = selected_record
        if selected_cellids:
            selection_results.append(selection_result)

    accepted_results, conflict_cellids = (
        eject_cross_kind_point_selection_conflicts(selection_results)
    )
    accepted_cellids = {
        cid
        for result in accepted_results
        for cid in result["selected_cellids"]
    }
    rejected_cellids.update(conflict_cellids - accepted_cellids)
    return {
        "accepted_results": accepted_results,
        "rejected_cellids": rejected_cellids,
    }


def eject_cross_kind_point_selection_conflicts(selection_results):
    """Reject complete X/O records that claim any common VDF cell.

    Sparse VDF storage can make an X source and an O source claim the same
    stored cell, which cannot carry an unambiguous training label. Every
    involved complete record is therefore rejected, including its otherwise
    unique selected cells. No X/O priority or overlap class is used.
    Same-kind overlaps retain the historical first-sample ownership, and
    surviving source-record order is unchanged.

    Parameters
    ----------
    selection_results : sequence of dict
        Nonempty point-selection results in stable source order.

    Returns
    -------
    accepted_results : tuple of dict
        Results not involved in a cross-kind selected-cell conflict.
    rejected_cellids : set of int
        Every selected cell belonging to an ejected source record.
    """

    selected_sets = [
        set(result["selected_cellids"])
        for result in selection_results
    ]
    conflicting_indices = set()
    x_indices = [
        index
        for index, result in enumerate(selection_results)
        if result["point_record"]["point_kind"] == "x"
    ]
    o_indices = [
        index
        for index, result in enumerate(selection_results)
        if result["point_record"]["point_kind"] == "o"
    ]
    for x_index in x_indices:
        for o_index in o_indices:
            if selected_sets[x_index] & selected_sets[o_index]:
                conflicting_indices.update((x_index, o_index))

    rejected_cellids = {
        cid
        for index in conflicting_indices
        for cid in selected_sets[index]
    }
    accepted_results = tuple(
        result
        for index, result in enumerate(selection_results)
        if index not in conflicting_indices
    )
    return accepted_results, rejected_cellids


def invert_cellids_by_position(cellids_by_position):
    """Map each VDF cell to its first deterministic selection position.

    Combination policies use this inversion to compare methods by physical
    cell rather than geometry-specific position names. ``setdefault`` retains
    the first position when one method names the same cell more than once.

    Parameters
    ----------
    cellids_by_position : dict
        Selection positions mapped to VDF cell IDs.

    Returns
    -------
    dict
        Cell IDs mapped to their first position name.
    """

    positions_by_cellid = {}
    for position, cid in cellids_by_position.items():
        positions_by_cellid.setdefault(int(cid), position)
    return positions_by_cellid


def create_combined_selection_metadata(
    selection_agreement,
    physical_selected,
    manual_selected,
    plot_selection_method,
):
    """Describe agreement between physical and manual selection.

    Stage-3 planning attaches these scalar fields to selected positions so
    later metadata can describe which geometry supplied a sample without
    altering its class label or VDF values.

    Parameters
    ----------
    selection_agreement : str
        Agreement category such as ``"both"`` or ``"manual_only"``.
    physical_selected : bool
        Whether physical geometry selected this cell.
    manual_selected : bool
        Whether the configured manual box selected this cell.
    plot_selection_method : {"physical", "manual"}
        Geometry emphasized in diagnostic plotting.

    Returns
    -------
    dict
        Agreement fields stored with the planned sample.
    """

    return {
        "selection_agreement": selection_agreement,
        "physical_selected": bool(physical_selected),
        "manual_selected": bool(manual_selected),
        "plot_selection_method": plot_selection_method,
    }


def get_point_selection_method(config, point_kind):
    """Return the configured method for one critical-point kind.

    Topology and sample-selection stages share this lookup so expensive
    X-point or O-point geometry is calculated only for methods that use it.

    Parameters
    ----------
    config : dict
        Dataset or nested point-selection configuration.
    point_kind : {"x", "o"}
        Magnetic critical-point kind.

    Returns
    -------
    str
        One of ``manual``, ``physical``, ``consensus``, or ``union``.

    """

    selection_config = get_point_selection_config(
        config=config,
        point_kind=point_kind,
    )
    return selection_config.get("method", "physical")
