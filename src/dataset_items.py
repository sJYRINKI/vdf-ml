def iter_labeled_coords(config):
    """
    Iterate over labeled coordinates for each class from config.

    Parameters
    ----------
    config : dict
        Dataset config dictionary.

    Yields
    ------
    class_name :str
        Name of the class.
    label : int
        Integer label for the class
    coord_re : list of float
        Coordinate in Earth radii, given as ``[x, y, z]``.
    """

    labels = config["labels"]

    for class_name, label in labels.items():
        coord_key = f"{class_name}_coord_re"

        yield class_name, int(label), config[coord_key]