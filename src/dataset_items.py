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
    class_coords = config.get("class_coords_re")

    for class_name, label in labels.items():
        if class_coords is None:
            coord_key = f"{class_name}_coord_re"
            coords = [config[coord_key]]
        else:
            coords = class_coords[class_name]

        for coord_re in coords:
            yield class_name, int(label), coord_re
