import numpy as np
import analysator as pt


class VdfExtractor:
    """
    Extract dense VDF arrays using one cached velocity-space ordering.

    Parameters
    ----------
    reader : analysator.vlsvfile.VlsvReader
        Open VLSV file reader.
    pop : str, optional
        Particle population name used by Analysator.
    dtype : data-type, optional
        Output data type for dense VDF arrays.
    """

    def __init__(self, reader, pop="avgs", dtype=np.float32):
        self.reader = reader
        self.pop = pop
        self.dtype = dtype

        size = reader.get_velocity_mesh_size(pop)
        self.vdf_shape = (
            4 * int(size[0]),
            4 * int(size[1]),
            4 * int(size[2]),
        )
        self.n_velocity_cells = int(np.prod(self.vdf_shape))
        self.sorted_velocity_indices = create_sorted_velocity_indices(
            reader=reader,
            n_velocity_cells=self.n_velocity_cells,
            pop=pop,
        )

    def extract(self, cid, box=-1):
        """
        Extract a 3D VDF from one Vlasiator cell.

        Parameters
        ----------
        cid : int
            Spatial cell ID from which the VDF is extracted.
        box : int, optional
            If positive, crop the VDF around its maximum value using this value
            as the half-width of the crop in index space. If ``-1``, return the
            full VDF.

        Returns
        -------
        numpy.ndarray
            Dense 3D VDF array with dtype ``float32`` by default. The returned
            axis order is ``[vx, vy, vz]``.
        """

        if int(cid) <= 0:
            raise ValueError("cid must be positive")

        velocity_cells = self.reader.read_velocity_cells(int(cid), self.pop)
        dist = np.zeros(self.n_velocity_cells, dtype=self.dtype)
        dist[list(velocity_cells.keys())] = list(velocity_cells.values())

        vdf = dist[self.sorted_velocity_indices].reshape(self.vdf_shape)
        max_indices = np.unravel_index(np.argmax(vdf), vdf.shape)
        box = int(box)

        if box > 0:
            i, j, k = max_indices
            data = vdf[
                (i - box):(i + box),
                (j - box):(j + box),
                (k - box):(k + box),
            ]
        else:
            data = vdf

        data = np.swapaxes(data, 2, 0)

        return np.asarray(data, dtype=self.dtype)


def create_sorted_velocity_indices(reader, n_velocity_cells, pop="avgs"):
    """
    Create the velocity-cell ordering used by VDF extraction.

    Parameters
    ----------
    reader : analysator.vlsvfile.VlsvReader
        Open VLSV file reader.
    n_velocity_cells : int
        Number of cells in the dense velocity mesh.
    pop : str, optional
        Particle population name used by Analysator.

    Returns
    -------
    numpy.ndarray
        Indices that sort velocity cells in the same order as the original
        repeated stable sorting implementation.
    """

    vids = np.arange(int(n_velocity_cells))
    velocity_coords = reader.get_velocity_cell_coordinates(vids, pop)
    sorted_indices = vids

    for axis_index in range(3):
        axis_order = np.argsort(
            velocity_coords[:, axis_index],
            kind="stable",
        )
        velocity_coords = velocity_coords[axis_order]
        sorted_indices = sorted_indices[axis_order]

    return sorted_indices


def extract_vdf_from_reader(reader, cid, box=-1, pop="avgs"):
    """
    Extract a 3D VDF from one Vlasiator cell.

    This function reads the sparse velocity-space data stored in a `.vlsv`
    file, places the values into a full velocity-space grid, sorts the grid by
    velocity coordinates, and returns the VDF as a dense NumPy array.

    Parameters
    ----------
    reader : analysator.vlsvfile.VlsvReader
        Open VLSV file reader.
    cid : int
        Spatial cell ID from which the VDF is extracted.
    box : int, optional
        If positive, crop the VDF around its maximum value using this value as
        the half-width of the crop in index space. If `-1`, return VDF.
    pop : str, optional
        Particle population name used by Analysator.

    Returns
    -------
    numpy.ndarray
        Dense 3D VDF array with dtype `float32`. The returned axis order is
        ``[vx, vy, vz]``.
    """

    extractor = VdfExtractor(reader=reader, pop=pop)

    return extractor.extract(cid=cid, box=box)


def extract_vdf(file_location, cid, box=-1, pop="avgs"):
    """
    Extract a 3D VDF from one Vlasiator cell.

    Parameters
    ----------
    file_location : str
        Path to the `.vlsv` file.
    cid : int
        Spatial cell ID from which the VDF is extracted.
    box : int, optional
        If positive, crop the VDF around its maximum value using this value as
        the half-width of the crop in index space. If `-1`, return VDF.
    pop : str, optional
        Particle population name used by Analysator.

    Returns
    -------
    numpy.ndarray
        Dense 3D VDF array with dtype `float32`.
    """

    reader = pt.vlsvfile.VlsvReader(file_location)
    return extract_vdf_from_reader(
        reader=reader,
        cid=cid,
        box=box,
        pop=pop,
    )
