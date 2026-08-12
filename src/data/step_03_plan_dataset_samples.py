"""Stage 3: plan ordered physical VDF samples for all timesteps.

This stage follows VDF-cell discovery and precedes physical VDF extraction.
It maps manual plasma-region coordinates and detected X/O records to
VDF-carrying cells, omits empty point selections, ejects complete X/O records
that claim a common stored cell, assigns the historical magnetotail fallback
to remaining cells, and records point-reference geometry.

Inputs are the extraction configuration and requested timesteps. Outputs are
stable per-timestep sample records, their shared ``[vx, vy, vz]`` velocity
grid, and an optional in-memory handoff of accepted point geometry for Stage
6. Spatial coordinates are expressed in Earth radii.
"""

from dataclasses import dataclass
import time

import analysator as pt
from joblib import Parallel, delayed

from src.data.background_sampling import (
    create_current_layer_sample_specs,
)
from src.data.metadata import (
    OMITTED_DATASET_METADATA_COLUMNS,
    add_point_reference_metadata,
)
from src.data.step_02_find_vdf_cells import (
    get_nearest_vdf_cellid,
    get_vdf_cells_with_coords_re,
)
from src.data.dense_vdf import resolve_velocity_population
from src.data.step_01_open_vlsv_files import create_timestep_path
from src.data.velocity_grid import create_velocity_grid_descriptor
from src.data.velocity_grid import normalize_velocity_grid_descriptor
from src.physics.point_labels import (
    create_point_label_data,
    iter_labeled_coords,
)
from src.physics.point_selection import (
    create_accepted_point_selection_results,
    get_point_selection_method,
    is_point_record,
    unpack_labeled_coord,
)
from src.physics.point_selection_metadata import (
    create_point_sample_metadata,
)


@dataclass(frozen=True)
class TimestepPlotData:
    """Retain accepted source-point geometry for one colormap frame.

    Stage 3 creates this lightweight handoff only when dataset colormap
    frames are enabled. It keeps accepted X/O point records after planning
    so Stage 6 can draw the exact Hessian boxes and island contours without
    rerunning topology detection or writing a plotting sidecar.

    Attributes
    ----------
    timestep : int
        Simulation timestep represented by the point records.
    x_point_records : tuple of dict
        Nonempty, conflict-free X-point records with their selected VDF cells.
    o_point_records : tuple of dict
        Nonempty, conflict-free O-point records with their selected VDF cells.
    """

    timestep: int
    x_point_records: tuple
    o_point_records: tuple


def create_timestep_sample_specs_for_timestep(config, timestep):
    """Plan samples for one timestep using a shared VLSV reader.

    A planning worker calls this entry point for one independent simulation
    timestep. It opens the bulk file once, discovers VDF cells, detects and
    labels X/O points, and passes all aligned inputs to the detailed planner;
    extraction follows only after every requested timestep has been planned.

    Parameters
    ----------
    config : dict
        Extraction configuration with VLSV/flux templates, manual plasma
        classes, point-selection methods, the magnetotail region, and the
        explicit VDF population or ``auto`` selection. The selected
        population owns both the VDF and its fluid moments.
    timestep : int
        Simulation timestep identifier.

    Returns
    -------
    tuple
        ``(timestep, sample_specs, plot_data)``. Specifications preserve the
        configured manual/point order followed by stable current-layer CID
        order. The first nonempty record owns the current ``[vx, vy,vz]``
        velocity-grid descriptor. ``plot_data`` is ``None`` unless dataset
        colormap frames are enabled.

    Notes
    -----
    This entry point opens the timestep VLSV file and prints stage timings.
    """

    file_location = create_timestep_path(
        path_template=config["file_template_bulk"],
        timestep=timestep,
    )
    planning_start = time.perf_counter()
    reader_start = time.perf_counter()
    reader = pt.vlsvfile.VlsvReader(str(file_location))
    reader_elapsed = time.perf_counter() - reader_start

    population = resolve_velocity_population(
        reader,
        config.get("population"),
    )
    velocity_grid = create_velocity_grid_descriptor(
        reader=reader,
        pop=population,
    )
    vdf_cells_start = time.perf_counter()
    vdf_cellids, vdf_coords_re = get_vdf_cells_with_coords_re(
        reader,
        pop=population,
    )
    vdf_cells_elapsed = time.perf_counter() - vdf_cells_start

    point_start = time.perf_counter()
    labeled_coords = list(iter_labeled_coords(config))
    point_label_data = create_point_label_data(
        config=config,
        timestep=timestep,
        reader=reader,
    )
    labeled_coords.extend(point_label_data["point_labeled_coords"])
    point_selection_data = create_accepted_point_selection_results(
        config=config,
        point_records=point_label_data["point_labeled_coords"],
        vdf_cellids=vdf_cellids,
        vdf_coords_re=vdf_coords_re,
    )
    point_elapsed = time.perf_counter() - point_start

    specs_start = time.perf_counter()
    sample_specs = create_timestep_sample_specs(
        config=config,
        timestep=timestep,
        labeled_coords=labeled_coords,
        rejected_cellids=point_selection_data["rejected_cellids"],
        reader=reader,
        vdf_cellids=vdf_cellids,
        vdf_coords_re=vdf_coords_re,
        point_selection_results=point_selection_data["accepted_results"],
        raw_x_point_records=point_label_data["raw_x_point_records"],
        raw_o_point_records=point_label_data["raw_o_point_records"],
    )
    if sample_specs:
        sample_specs[0]["velocity_grid"] = velocity_grid

    specs_elapsed = time.perf_counter() - specs_start
    planning_elapsed = time.perf_counter() - planning_start
    print(
        f"Timestep {int(timestep)} planning: "
        f"reader={reader_elapsed:.2f}s, "
        f"vdf_cells={vdf_cells_elapsed:.2f}s, "
        f"points={point_elapsed:.2f}s, "
        f"samples={specs_elapsed:.2f}s, "
        f"total={planning_elapsed:.2f}s, "
        f"n={len(sample_specs)}"
    )
    plot_data = None
    if config.get("postprocessing", {}).get("plot_colormaps", False):
        accepted_point_records = tuple(
            result["point_record"]
            for result in point_selection_data["accepted_results"]
        )
        plot_data = TimestepPlotData(
            timestep=int(timestep),
            x_point_records=tuple(
                record
                for record in accepted_point_records
                if record["point_kind"] == "x"
            ),
            o_point_records=tuple(
                record
                for record in accepted_point_records
                if record["point_kind"] == "o"
            ),
        )
    return int(timestep), sample_specs, plot_data


def create_timestep_sample_specs(
    config,
    timestep,
    labeled_coords,
    rejected_cellids=None,
    reader=None,
    vdf_cellids=None,
    vdf_coords_re=None,
    point_selection_results=None,
    raw_x_point_records=None,
    raw_o_point_records=None,
):
    """Create ordered manual, point, and current-layer sample specifications.

    This central Stage 3 operation maps manual coordinates to their nearest
    VDF cells, applies the historical X/O selection geometry, removes
    cross-class conflicts, appends still-unassigned magnetotail cells as
    ``current_layer``, and adds X/O reference vectors for Stage 4 metadata.

    Parameters
    ----------
    config : dict
        Dataset configuration with source templates and physical labels.
    timestep : int
        Simulation timestep identifier.
    labeled_coords : iterable of tuple or dict
        Static ``(class_name, class_id, coord_re)`` entries and detected point
        records.
    rejected_cellids : set of int, optional
        Cells rejected during point-label conflict handling.
    reader : analysator.vlsvfile.VlsvReader, optional
        Existing source reader; a reader is opened when omitted.
    vdf_cellids : numpy.ndarray, optional
        VDF-carrying spatial cell identifiers.
    vdf_coords_re : numpy.ndarray, optional
        VDF-cell centres in Earth radii with shape ``(n_cells, 3)``.
    point_selection_results : iterable of dict, optional
        Nonempty, cross-kind-conflict-free point results. Direct callers may
        omit them to run the same accepted-point pipeline locally.
    raw_x_point_records : iterable of dict, optional
        Unfiltered X-point records used for nearest-reference geometry.
    raw_o_point_records : iterable of dict, optional
        Unfiltered O-point records used for nearest-reference geometry.

    Returns
    -------
    list of dict
        Extraction specifications in manual-class order, X/O order, then
        ascending magnetotail CID order. Records contain selected CIDs,
        physical classes, coordinates in Earth radii, and X/O topology
        metadata.

    Notes
    -----
    A VLSV reader is opened only when ``reader`` is omitted.
    """

    file_location = create_timestep_path(
        path_template=config["file_template_bulk"],
        timestep=timestep,
    )
    if reader is None:
        reader = pt.vlsvfile.VlsvReader(str(file_location))
    labeled_coords = list(labeled_coords)
    if raw_x_point_records is None:
        raw_x_point_records = [
            value
            for value in labeled_coords
            if is_point_record(value)
            and str(value.get("point_kind", "")).lower() == "x"
        ]
    else:
        raw_x_point_records = list(raw_x_point_records)
    if raw_o_point_records is None:
        raw_o_point_records = [
            value
            for value in labeled_coords
            if is_point_record(value)
            and str(value.get("point_kind", "")).lower() == "o"
        ]
    else:
        raw_o_point_records = list(raw_o_point_records)

    rejected_cellids = {int(cid) for cid in (rejected_cellids or set())}
    if vdf_cellids is None or vdf_coords_re is None:
        vdf_cellids, vdf_coords_re = get_vdf_cells_with_coords_re(reader)
    if point_selection_results is None:
        point_selection_data = create_accepted_point_selection_results(
            config=config,
            point_records=[
                value
                for value in labeled_coords
                if is_point_record(value)
            ],
            vdf_cellids=vdf_cellids,
            vdf_coords_re=vdf_coords_re,
        )
        point_selection_results = point_selection_data["accepted_results"]
        rejected_cellids.update(point_selection_data["rejected_cellids"])
    else:
        point_selection_results = tuple(point_selection_results)
    vdf_coord_by_cellid = {
        int(cid): tuple(float(value) for value in coord_re)
        for cid, coord_re in zip(vdf_cellids, vdf_coords_re)
    }

    sample_specs = []
    seen_class_cellids = set()
    ordered_entries = [
        labeled_coord
        for labeled_coord in labeled_coords
        if not is_point_record(labeled_coord)
    ]
    ordered_entries.extend(point_selection_results)
    for entry in ordered_entries:
        selection_result = (
            entry
            if isinstance(entry, dict) and "point_record" in entry
            else None
        )
        labeled_coord = (
            selection_result["point_record"]
            if selection_result is not None
            else entry
        )
        class_name, label, coord_re = unpack_labeled_coord(labeled_coord)
        sample_metadata = {}
        metadata_by_position = {}
        if selection_result is not None:
            cellids_by_position = selection_result["cellids_by_position"]
            metadata_by_position = selection_result["metadata_by_position"]
            sample_metadata = create_point_sample_metadata(
                config=config,
                point_record=labeled_coord,
                selection_method=get_point_selection_method(
                    config=config,
                    point_kind=labeled_coord["point_kind"],
                ),
            )
        else:
            cid = get_nearest_vdf_cellid(
                coord_re=coord_re,
                vdf_cellids=vdf_cellids,
                vdf_coords_re=vdf_coords_re,
            )
            cellids_by_position = {"closest": cid}

        for neighbor_position, cid in cellids_by_position.items():
            class_cellid = (class_name, int(cid))
            if class_cellid in seen_class_cellids:
                continue
            seen_class_cellids.add(class_cellid)
            sample_spec = {
                "file_location": file_location,
                "cid": int(cid),
                "label": int(label),
                "class_name": class_name,
                "coord_re": coord_re,
                "vdf_coord_re": vdf_coord_by_cellid[int(cid)],
                "timestep": int(timestep),
            }
            sample_spec.update(
                {
                    key: value
                    for key, value in sample_metadata.items()
                    if key not in OMITTED_DATASET_METADATA_COLUMNS
                }
            )
            sample_spec.update(
                {
                    key: value
                    for key, value in metadata_by_position.get(
                        neighbor_position,
                        {},
                    ).items()
                    if key not in OMITTED_DATASET_METADATA_COLUMNS
                }
            )
            sample_specs.append(sample_spec)

    conflicting_cellids = find_conflicting_cellids(sample_specs)
    rejected_cellids.update(conflicting_cellids)
    sample_specs = remove_conflicting_cellids(
        sample_specs,
        rejected_cellids,
    )
    sample_specs.extend(
        create_current_layer_sample_specs(
            config=config,
            file_location=file_location,
            existing_sample_specs=sample_specs,
            timestep=timestep,
            rejected_cellids=rejected_cellids,
            vdf_cellids=vdf_cellids,
            vdf_coords_re=vdf_coords_re,
        )
    )
    sample_specs = remove_conflicting_cellids(
        sample_specs,
        find_conflicting_cellids(sample_specs),
    )
    add_point_reference_metadata(
        sample_specs=sample_specs,
        raw_x_point_records=raw_x_point_records,
        raw_o_point_records=raw_o_point_records,
    )
    return sample_specs


def find_conflicting_cellids(sample_specs):
    """Find cells assigned to more than one physical class.

    Planning calls this before and after current-layer assignment. Returning
    cell IDs rather than individual records lets the caller exclude every
    label for a physically contradictory spatial cell.

    Parameters
    ----------
    sample_specs : list of dict
        Planned sample records containing ``cid`` and ``class_name``.

    Returns
    -------
    set of int
        Spatial cell IDs with contradictory physical-class assignments.
    """

    class_names_by_cellid = {}
    for sample_spec in sample_specs:
        cid = int(sample_spec["cid"])
        class_names_by_cellid.setdefault(cid, set()).add(
            sample_spec["class_name"]
        )
    return {
        cid
        for cid, class_names in class_names_by_cellid.items()
        if len(class_names) > 1
    }


def remove_conflicting_cellids(sample_specs, conflicting_cellids):
    """Remove every sample for a contradictory spatial cell.

    This filter preserves the relative order of all retained sample records,
    which is required because planning order becomes dataset row order in the
    following extraction stage.

    Parameters
    ----------
    sample_specs : list of dict
        Planned sample records.
    conflicting_cellids : iterable of int
        Cell IDs excluded from every physical class.

    Returns
    -------
    list of dict
        Records whose cell IDs are not in ``conflicting_cellids``.
    """

    if not conflicting_cellids:
        return sample_specs
    conflicting_cellids = {int(cid) for cid in conflicting_cellids}
    return [
        sample_spec
        for sample_spec in sample_specs
        if int(sample_spec["cid"]) not in conflicting_cellids
    ]


def plan_dataset_sample_specs(config, timesteps, planning_n_jobs):
    """Plan every requested timestep without changing timestep order.

    Stage 0 calls this before allocating dataset arrays, because the combined
    sample count determines their first dimension. Serial and joblib branches
    invoke the same per-timestep planner and convert results back to requested
    timestep order.

    Parameters
    ----------
    config : dict
        Effective extraction configuration.
    timesteps : sequence of int
        Requested simulation timesteps in dataset order.
    planning_n_jobs : int
        Joblib worker count used for independent per-timestep planning.

    Returns
    -------
    sample_specs_by_timestep : dict
        Mapping from timestep to ordered sample specifications.
    plot_data_by_timestep : dict
        Mapping from timestep to accepted point geometry for enabled
        colormap frames. Disabled plotting produces an empty mapping.
    elapsed : float
        Planning wall-clock duration in seconds.
    """

    planning_start = time.perf_counter()
    if planning_n_jobs == 1:
        sample_spec_results = [
            create_timestep_sample_specs_for_timestep(
                config=config,
                timestep=timestep,
            )
            for timestep in timesteps
        ]
    else:
        sample_spec_results = Parallel(n_jobs=planning_n_jobs)(
            delayed(create_timestep_sample_specs_for_timestep)(
                config=config,
                timestep=timestep,
            )
            for timestep in timesteps
        )
    sample_specs_by_timestep = {}
    plot_data_by_timestep = {}
    for timestep, sample_specs, plot_data in sample_spec_results:
        sample_specs_by_timestep[int(timestep)] = sample_specs
        if plot_data is not None:
            plot_data_by_timestep[int(timestep)] = plot_data
    return (
        sample_specs_by_timestep,
        plot_data_by_timestep,
        time.perf_counter() - planning_start,
    )


def count_sample_specs_by_timestep(sample_specs_by_timestep):
    """Count planned sample records independently for each timestep.

    Stage 0 retains these counts to report the sample distribution and
    allocate the leading array dimension without extracting any VDF values.

    Parameters
    ----------
    sample_specs_by_timestep : dict
        Mapping from timestep to ordered sample specifications.

    Returns
    -------
    dict
        Integer timestep keys mapped to sample counts.
    """

    return {
        int(timestep): len(sample_specs)
        for timestep, sample_specs in sample_specs_by_timestep.items()
    }


def count_sample_counts(sample_counts_by_timestep):
    """Sum per-timestep sample counts.

    The total becomes the leading dimension allocated for raw, optional
    Hermite, and metadata outputs.

    Parameters
    ----------
    sample_counts_by_timestep : dict
        Mapping from timestep to sample count.

    Returns
    -------
    int
        Total number of planned dataset samples.
    """

    return sum(sample_counts_by_timestep.values())


def get_planned_velocity_grid(sample_specs_by_timestep):
    """Return the grid recorded by the first nonempty timestep plan.

    The dataset stores one shared descriptor because all output rows use the
    same ``[vx, vy, vz]`` mesh convention. Normalizing the selected record
    converts reader-derived values to stable archive types before stage 5.

    Parameters
    ----------
    sample_specs_by_timestep : dict
        Mapping from timestep to planned sample specifications.

    Returns
    -------
    dict
        Velocity grid with ``[vx, vy, vz]`` shape and bounds in metres per
        second.
    """

    return normalize_velocity_grid_descriptor(
        next(
            sample_specs[0]["velocity_grid"]
            for sample_specs in sample_specs_by_timestep.values()
            if sample_specs
        )
    )
