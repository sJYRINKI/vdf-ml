import numpy as np

R_EARTH = 6.371e6

def coord_re_to_m(coord_re):
    """
    Convert coordinates from Earth radii to meters.

    Parameters
    ----------
    coord_re : array-like of float
        Coordinate in Earth radii, given as ``[x, y, z]``.

    Returns
    -------
    numpy.ndarray
        Coordinate in maters.
    """
    return np.array(coord_re, dtype=float) *R_EARTH

def get_cellid_with_vdf(reader, coord_re, pop="avgs"):
    """
    Find the nearest cell ID with a VDF.

    Parameters
    ----------
    reader : analysator.vlsvfile.VlsvReader
        Open VLSV file reader.
    coord_re : array-like of float
        Coordinate in Earth radii, given as ``[x, y, z]``.
    pop : str, optional
        Particle population name.

    Returns
    -------
    int
        Cell ID with a VDF
    """

    coord_m = coord_re_to_m(coord_re)
    cid = reader.get_cellid_with_vdf(coord_m, pop=pop)

    return int(cid)