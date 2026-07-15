import analysator as pt

from src.point_topology import find_point_records
from src.timesteps import create_timestep_path
from src.vdf_helpers import get_cellid_with_vdf


def create_labeled_coords_for_timestep(config, timestep):
    """
    Create labeled coordinate data for one timestep.

    Parameters
    ----------
    config : dict
        Dataset config.
    timestep : int
        Timestep to process.

    Returns
    -------
    tuple
        Pair of ``(timestep, label_data)``. ``label_data`` contains
        ``labeled_coords``, point-conflict ``rejected_cellids``, and the raw
        X/O point records used for nearest-point geometry.
    """

    timestep_labeled_coords = list(iter_labeled_coords(config))
    point_label_data = create_point_label_data(
        config=config,
        timestep=timestep,
    )
    point_labeled_coords = point_label_data["point_labeled_coords"]
    rejected_cellids = point_label_data["rejected_cellids"]
    timestep_labeled_coords.extend(point_labeled_coords)

    return int(timestep), {
        "labeled_coords": timestep_labeled_coords,
        "rejected_cellids": rejected_cellids,
        "raw_x_point_records": point_label_data["raw_x_point_records"],
        "raw_o_point_records": point_label_data["raw_o_point_records"],
    }


def create_label_data_by_timestep(config, timesteps):
    """
    Create labeled coordinate data for each timestep.

    Parameters
    ----------
    config : dict
        Dataset config.
    timesteps : iterable of int
        Timesteps to process.

    Returns
    -------
    dict
        Mapping from timestep to labeled coordinate data.
    """

    return dict(
        create_labeled_coords_for_timestep(
            config=config,
            timestep=timestep,
        )
        for timestep in timesteps
    )


def iter_labeled_coords(config):
    """
    Iterate over labeled coordinates for each class from config.

    Parameters
    ----------
    config : dict
        Dataset config dictionary.

    Yields
    ------
    class_name : str
        Name of the class.
    label : int
        Integer label for the class.
    coord_re : list of float
        Coordinate in Earth radii, given as ``[x, y, z]``.
    """

    labels = config["labels"]
    class_coords = config.get("class_coords_re", {})

    for class_name, coords in class_coords.items():
        label = labels[class_name]

        for coord_re in coords:
            yield class_name, int(label), coord_re


def iter_point_labeled_coords(config, timestep):
    """
    Iterate over labeled X-point and O-point records for one timestep.

    Parameters
    ----------
    config : dict
        Dataset config dictionary.
    timestep : int
        Timestep to process.

    Yields
    ------
    point_record : dict
        Point record containing class, label, coordinate, and selection data.
    """

    point_labeled_coords, _ = create_point_labeled_coords(
        config=config,
        timestep=timestep,
    )

    yield from point_labeled_coords


def create_point_labeled_coords(config, timestep, reader=None):
    """
    Create point records and rejected point cell IDs.

    Parameters
    ----------
    config : dict
        Dataset config dictionary.
    timestep : int
        Timestep to process.
    reader : analysator.vlsvfile.VlsvReader, optional
        Existing reader for the timestep bulk file.

    Returns
    -------
    point_labeled_coords : list of dict
        Point records for accepted X/O points.
    rejected_cellids : set of int
        VDF cell IDs rejected because X and O points mapped to the same cell.
    """

    point_label_data = create_point_label_data(
        config=config,
        timestep=timestep,
        reader=reader,
    )
    return (
        point_label_data["point_labeled_coords"],
        point_label_data["rejected_cellids"],
    )


def create_point_label_data(config, timestep, reader=None):
    """
    Create accepted point labels while retaining all detected point records.

    The raw records are the direct output of ``find_point_records``. They are
    retained before X/O records that map to the same VDF cell are removed, so
    downstream code can use every detected point for nearest-point geometry.

    Parameters
    ----------
    config : dict
        Dataset config dictionary.
    timestep : int
        Timestep to process.
    reader : analysator.vlsvfile.VlsvReader, optional
        Existing reader for the timestep bulk file.

    Returns
    -------
    dict
        Accepted labeled point records, rejected VDF cell IDs, and raw X/O
        point records from topology detection.
    """

    points_config = config.get("points")
    labels = config["labels"]

    x_class_name = points_config["x_class_name"]
    o_class_name = points_config["o_class_name"]

    flux_file_location = create_timestep_path(
        path_template=config["file_template_flux"],
        timestep=timestep,
    )

    if reader is None:
        bulk_file_location = create_timestep_path(
            path_template=config["file_template_bulk"],
            timestep=timestep,
        )
        reader = pt.vlsvfile.VlsvReader(str(bulk_file_location))

    raw_x_point_records, raw_o_point_records = find_point_records(
        reader=reader,
        flux_file_location=flux_file_location,
        points_config=points_config,
    )

    x_point_records, o_point_records, rejected_cellids = remove_shared_cellid_points(
        reader=reader,
        x_point_records=raw_x_point_records,
        o_point_records=raw_o_point_records,
    )

    point_labeled_coords = []

    for point_record in x_point_records:
        labeled_record = dict(point_record)
        labeled_record["class_name"] = x_class_name
        labeled_record["label"] = int(labels[x_class_name])
        point_labeled_coords.append(labeled_record)

    for point_record in o_point_records:
        labeled_record = dict(point_record)
        labeled_record["class_name"] = o_class_name
        labeled_record["label"] = int(labels[o_class_name])
        point_labeled_coords.append(labeled_record)

    return {
        "point_labeled_coords": point_labeled_coords,
        "rejected_cellids": rejected_cellids,
        "raw_x_point_records": raw_x_point_records,
        "raw_o_point_records": raw_o_point_records,
    }


def remove_shared_cellid_points(reader, x_point_records, o_point_records):
    """
    Remove duplicate or contradictory point labels by VDF cell ID.

    Parameters
    ----------
    reader : analysator.vlsvfile.VlsvReader
        Open VLSV file reader for the matching timestep.
    x_point_records : list of dict
        X-point records.
    o_point_records : list of dict
        O-point records.

    Returns
    -------
    filtered_x_point_records : list of dict
        One X-point record per non-conflicting VDF cell ID.
    filtered_o_point_records : list of dict
        One O-point record per non-conflicting VDF cell ID.
    shared_cellids : set of int
        VDF cell IDs rejected because both point classes mapped to them.
    """

    x_points_by_cellid = group_point_records_by_cellid(reader, x_point_records)
    o_points_by_cellid = group_point_records_by_cellid(reader, o_point_records)

    shared_cellids = set(x_points_by_cellid) & set(o_points_by_cellid)

    filtered_x_point_records = [
        point_records[0]
        for cellid, point_records in x_points_by_cellid.items()
        if cellid not in shared_cellids
    ]

    filtered_o_point_records = [
        point_records[0]
        for cellid, point_records in o_points_by_cellid.items()
        if cellid not in shared_cellids
    ]

    return filtered_x_point_records, filtered_o_point_records, shared_cellids


def group_point_records_by_cellid(reader, point_records):
    """
    Group point records by the VDF cell ID used for saved samples.

    Parameters
    ----------
    reader : analysator.vlsvfile.VlsvReader
        Open VLSV file reader used to convert coordinates to VDF cell IDs.
    point_records : list of dict
        Point records containing ``coord_re``.

    Returns
    -------
    dict[int, list of dict]
        Mapping from VDF cell ID to all point records that resolve to that cell.
    """

    point_records_by_cellid = {}

    for point_record in point_records:
        cellid = get_cellid_with_vdf(reader, point_record["coord_re"])
        point_records_by_cellid.setdefault(cellid, []).append(point_record)

    return point_records_by_cellid


def group_coords_by_cellid(reader, coords_re):
    """
    Group coordinates by the VDF cell ID used for saved samples.

    Parameters
    ----------
    reader : analysator.vlsvfile.VlsvReader
        Open VLSV file reader used to convert coordinates to VDF cell IDs.
    coords_re : list of list of float
        Coordinates in Earth radii, each given as ``[x, y, z]``.

    Returns
    -------
    dict[int, list of list of float]
        Mapping from VDF cell ID to all coordinates that resolve to that cell.
    """

    coords_by_cellid = {}

    for coord_re in coords_re:
        cellid = get_cellid_with_vdf(reader, coord_re)
        coords_by_cellid.setdefault(cellid, []).append(coord_re)

    return coords_by_cellid
