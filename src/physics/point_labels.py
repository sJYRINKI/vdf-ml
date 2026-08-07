"""Attach current class identities to static and detected coordinates.

Dataset sample planning calls this layer after opening a timestep VLSV file.
It leaves raw X/O detection records unchanged, adds configured class names and
IDs to copies of those records, and retains the raw geometry for downstream
topology metadata. Final VDF-cell selection, empty omission, and X/O conflict
ejection occur only after Stage 3 discovers the available stored VDF cells.
"""

from src.data.step_01_open_vlsv_files import create_timestep_path
from src.physics.point_topology import find_point_records


def create_labeled_coords_for_timestep(config, timestep):
    """Create configured and detected labeled coordinates for one timestep.

    This standalone entry point retains raw detector records for later
    topology-reference metadata. Final accepted point membership is resolved
    by Stage 3 after VDF-cell coordinates are available.

    Parameters
    ----------
    config : dict
        Dataset extraction configuration.
    timestep : int
        Simulation timestep identifier.

    Returns
    -------
    tuple
        ``(timestep, label_data)`` containing configured coordinates, labeled
        detected points, and unchanged raw X/O records.
    """

    timestep_labeled_coords = list(iter_labeled_coords(config))
    point_label_data = create_point_label_data(
        config=config,
        timestep=timestep,
    )
    timestep_labeled_coords.extend(
        point_label_data["point_labeled_coords"]
    )
    return int(timestep), {
        "labeled_coords": timestep_labeled_coords,
        "rejected_cellids": point_label_data["rejected_cellids"],
        "raw_x_point_records": point_label_data["raw_x_point_records"],
        "raw_o_point_records": point_label_data["raw_o_point_records"],
    }


def create_label_data_by_timestep(config, timesteps):
    """Create labeled coordinate data for each requested timestep.

    Parameters
    ----------
    config : dict
        Dataset extraction configuration.
    timesteps : iterable of int
        Simulation timestep identifiers in requested order.

    Returns
    -------
    dict
        Integer timesteps mapped to their coordinate and detector records.
    """

    return dict(
        create_labeled_coords_for_timestep(
            config=config,
            timestep=timestep,
        )
        for timestep in timesteps
    )


def iter_labeled_coords(config):
    """Yield configured manual coordinates in class and source order.

    Parameters
    ----------
    config : dict
        Dataset configuration containing class IDs and Earth-radius source
        coordinates.

    Yields
    ------
    class_name : str
        Current physical class name.
    label : int
        Configured integer class ID.
    coord_re : list of float
        Source coordinate ``[x, y, z]`` in Earth radii.
    """

    labels = config["labels"]
    for class_name, coords in config.get("class_coords_re", {}).items():
        for coord_re in coords:
            yield class_name, int(labels[class_name]), coord_re


def iter_point_labeled_coords(config, timestep):
    """Yield labeled raw X/O detector records for one timestep.

    Final manual/physical selection and conflict ejection require the stored
    VDF-cell array and therefore occur later in Stage 3.

    Parameters
    ----------
    config : dict
        Dataset extraction configuration.
    timestep : int
        Simulation timestep identifier.

    Yields
    ------
    point_record : dict
        Labeled copy of one unchanged detector record.
    """

    point_labeled_coords, _ = create_point_labeled_coords(
        config=config,
        timestep=timestep,
    )
    yield from point_labeled_coords


def create_point_labeled_coords(config, timestep, reader=None):
    """Return labeled detected points and no pre-selection rejections.

    Parameters
    ----------
    config : dict
        Dataset extraction configuration.
    timestep : int
        Simulation timestep identifier.
    reader : analysator.vlsvfile.VlsvReader, optional
        Existing reader for the timestep bulk source.

    Returns
    -------
    point_labeled_coords : list of dict
        X-then-O labeled copies of raw detector records.
    rejected_cellids : set of int
        Empty set because conflicts are evaluated after final cell selection.
    """

    point_label_data = create_point_label_data(
        config=config,
        timestep=timestep,
        reader=reader,
    )
    return point_label_data["point_labeled_coords"], set()


def create_point_label_data(
    config,
    timestep,
    reader=None,
    include_physical_geometry=None,
):
    """Label detected X/O records without changing detector output.

    Raw records are direct outputs from :func:`find_point_records`. Labeled
    copies are returned in X-then-O detector order so Stage 3 can apply final
    search selection and conflict ejection after VDF-cell discovery.

    Parameters
    ----------
    config : dict
        Dataset extraction configuration.
    timestep : int
        Simulation timestep identifier.
    reader : analysator.vlsvfile.VlsvReader, optional
        Existing reader for the timestep bulk source.
    include_physical_geometry : bool, optional
        Whether to retain physical X boxes and O contours independently of
        the configured selection method. Extraction defaults to its Stage 6
        plotting setting; standalone colormap rendering requests them.

    Returns
    -------
    dict
        Labeled detector copies, an empty pre-selection rejection set, and
        unchanged raw X/O detector records.
    """

    points_config = config["points"]
    labels = config["labels"]
    flux_file_location = create_timestep_path(
        path_template=config["file_template_flux"],
        timestep=timestep,
    )
    if reader is None:
        import analysator as pt

        bulk_file_location = create_timestep_path(
            path_template=config["file_template_bulk"],
            timestep=timestep,
        )
        reader = pt.vlsvfile.VlsvReader(str(bulk_file_location))

    if include_physical_geometry is None:
        include_physical_geometry = bool(
            config.get("postprocessing", {}).get("plot_colormaps", False)
        )
    raw_x_point_records, raw_o_point_records = find_point_records(
        reader=reader,
        flux_file_location=flux_file_location,
        points_config=points_config,
        include_physical_geometry=include_physical_geometry,
    )
    point_labeled_coords = []
    for point_record in raw_x_point_records:
        class_name = points_config["x_class_name"]
        labeled_record = dict(point_record)
        labeled_record["class_name"] = class_name
        labeled_record["label"] = int(labels[class_name])
        point_labeled_coords.append(labeled_record)
    for point_record in raw_o_point_records:
        class_name = points_config["o_class_name"]
        labeled_record = dict(point_record)
        labeled_record["class_name"] = class_name
        labeled_record["label"] = int(labels[class_name])
        point_labeled_coords.append(labeled_record)

    return {
        "point_labeled_coords": point_labeled_coords,
        "rejected_cellids": set(),
        "raw_x_point_records": raw_x_point_records,
        "raw_o_point_records": raw_o_point_records,
    }
