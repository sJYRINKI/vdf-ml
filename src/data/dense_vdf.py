"""Extract dense physical VDF arrays from sparse VLSV velocity cells.

Sparse values are reordered into the saved ``[vx, vy, vz]`` convention.
Missing velocity cells remain zero. The reusable extractor serves dataset
creation and prediction, so it is not assigned a dataset workflow stage.
"""

import numpy as np

from src.data.velocity_grid import (
    create_velocity_grid_descriptor,
    interpolate_volume,
)


EQUIVALENT_VELOCITY_POPULATIONS = ("avgs", "proton")


class VdfExtractor:
    """Extract dense VDFs or complete interpolated target-grid volumes.

    One instance is normally created for an open timestep VLSV reader and
    reused for every selected spatial cell in that file. It retains the
    velocity-grid description and, after the first extraction, the stable
    sparse-to-dense velocity-cell ordering; it does not own or close the
    supplied reader.

    Parameters
    ----------
    reader : analysator.vlsvfile.VlsvReader
        Open VLSV reader that owns the selected velocity mesh.
    pop : str, optional
        Explicit particle population. Automatic selection prefers ``avgs``
        and otherwise uses ``proton``.
    dtype : numpy.dtype, optional
        Floating-point dtype used for dense physical VDF arrays.
    velocity_mesh_shape : tuple of int, optional
        ``(vx, vy, vz)`` shape already resolved for this reader and
        population.

    Attributes
    ----------
    velocity_grid : dict
        Current velocity-grid descriptor with extents in metres per second.
    vdf_shape : tuple of int
        Dense VDF shape in ``[vx, vy, vz]`` order.
    pop : str
        Reader population used for sparse velocity-cell access.
    """

    def __init__(
        self,
        reader,
        pop=None,
        dtype=np.float32,
        *,
        velocity_mesh_shape=None,
    ):
        """Initialize extraction state for one open VLSV velocity mesh.

        Parameters
        ----------
        reader : analysator.vlsvfile.VlsvReader
            Open source reader retained for subsequent cell extractions.
        pop : str, optional
            Explicit particle population, or automatic equivalent-proton
            selection when omitted.
        dtype : numpy.dtype, optional
            Storage dtype for returned dense VDF values.
        velocity_mesh_shape : tuple of int, optional
            Precomputed ``(vx, vy, vz)`` velocity-cell shape for the selected
            population.
        """

        self.reader = reader
        self.dtype = dtype
        self.pop = resolve_velocity_population(reader=reader, pop=pop)
        self.velocity_grid = create_velocity_grid_descriptor(
            reader=reader,
            pop=self.pop,
            velocity_mesh_shape=velocity_mesh_shape,
        )
        self.vdf_shape = tuple(
            int(value) for value in self.velocity_grid["shape"]
        )
        self.n_velocity_cells = int(np.prod(self.vdf_shape))
        self.sorted_velocity_indices = None

    def extract(self, cid, box=-1):
        """Extract one sparse spatial-cell VDF into a dense physical cube.

        Dataset extraction calls this method for each planned cell after the
        timestep reader is opened. Sparse velocity-cell IDs are expanded,
        zero-filled, and reordered once into the project ``[vx, vy, vz]``
        convention before an optional peak-centred index crop is applied.

        Parameters
        ----------
        cid : int
            VLSV spatial cell identifier.
        box : int, optional
            Positive half-width for an index-space crop around the VDF peak;
            ``-1`` returns the complete velocity mesh.

        Returns
        -------
        numpy.ndarray
            Dense phase-space density in ``s^3 m^-6`` with axis order
            ``[vx, vy, vz]`` and the extractor dtype. Missing sparse cells
            contain zero.
        """

        velocity_cells = self.reader.read_velocity_cells(int(cid), self.pop)
        dist = np.zeros(self.n_velocity_cells, dtype=self.dtype)
        dist[list(velocity_cells.keys())] = list(velocity_cells.values())
        if self.sorted_velocity_indices is None:
            self.sorted_velocity_indices = create_sorted_velocity_indices(
                reader=self.reader,
                n_velocity_cells=self.n_velocity_cells,
                pop=self.pop,
            )
        nx, ny, nz = self.vdf_shape
        vdf = dist[self.sorted_velocity_indices].reshape((nz, ny, nx))
        vdf = np.transpose(vdf, (2, 1, 0))
        max_indices = np.unravel_index(np.argmax(vdf), vdf.shape)
        box = int(box)
        if box > 0:
            i, j, k = max_indices
            vdf = vdf[
                (i - box):(i + box),
                (j - box):(j + box),
                (k - box):(k + box),
            ]
        return np.asarray(vdf, dtype=self.dtype)

    def extract_interpolated_volume(self, cid, interpolation_plan):
        """Interpolate one complete sparse VDF onto the training grid.

        Raw prediction expands the selected source VDF in native
        ``(vx, vy, vz)`` order and applies the cached trilinear plan before
        logarithmic model preprocessing. All three velocity axes therefore
        reach the raw ``Conv3d`` input.

        Parameters
        ----------
        cid : int
            VLSV spatial cell identifier.
        interpolation_plan : dict
            Cached plan returned by
            :func:`src.data.velocity_grid.create_trilinear_volume_plan`.

        Returns
        -------
        numpy.ndarray
            Raw phase-space-density target volume in ``s^3 m^-6`` with axis
            order ``(vx, vy, vz)``. Missing source cells and out-of-coverage
            target centres are zero.

        Notes
        -----
        Interpolation acts on physical VDF values and does not conserve
        velocity moments.
        """

        return interpolate_volume(
            self.extract(int(cid)),
            interpolation_plan,
            self.dtype,
        )


def resolve_velocity_population(reader, pop=None):
    """Resolve the configured VDF population in one VLSV source.

    Dense extraction and VDF-cell discovery call this once per reader so all
    later velocity and fluid-moment operations use the same population
    namespace. Automatic selection preserves the established ``avgs``
    preference over ``proton``; an explicit active name supports
    multi-population files without substituting an equivalent proton species.

    Parameters
    ----------
    reader : analysator.vlsvfile.VlsvReader
        Open VLSV reader exposing active velocity populations.
    pop : str, optional
        Explicit active population; ``None`` or ``"auto"`` prefers ``avgs``
        and otherwise selects ``proton``.

    Returns
    -------
    str
        Population name accepted by the reader and reused for VDF extraction,
        velocity, number density, and pressure moments.
    """

    requested_population = None if pop is None else str(pop).strip()
    automatic = (
        not requested_population
        or requested_population.lower() == "auto"
    )
    candidates = (
        EQUIVALENT_VELOCITY_POPULATIONS
        if automatic
        else (requested_population,)
    )
    if hasattr(reader, "check_population"):
        return next(
            candidate
            for candidate in candidates
            if reader.check_population(candidate)
        )
    active_populations = tuple(
        str(population)
        for population in reader.active_populations
    )
    active_by_lower_name = {
        population.lower(): population
        for population in active_populations
    }
    return next(
        active_by_lower_name[candidate.lower()]
        for candidate in candidates
        if candidate.lower() in active_by_lower_name
    )


def create_sorted_velocity_indices(reader, n_velocity_cells, pop=None):
    """Reproduce the stable VLSV velocity-cell order used by extraction.

    The sparse VLSV ID order does not directly encode the saved dense-array
    axes. Three stable coordinate sorts construct the exact permutation used
    before reshaping and transposing values into ``[vx, vy, vz]`` order.

    Parameters
    ----------
    reader : analysator.vlsvfile.VlsvReader
        Open VLSV reader used to obtain velocity-cell coordinates.
    n_velocity_cells : int
        Total number of cells in the dense velocity mesh.
    pop : str, optional
        Explicit velocity population, or automatic equivalent-proton
        selection when omitted.

    Returns
    -------
    numpy.ndarray
        Integer indices ordered by three consecutive stable coordinate sorts,
        matching the established dense-array calculation order.
    """

    pop = resolve_velocity_population(reader=reader, pop=pop)
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


def extract_vdf_from_reader(reader, cid, box=-1, pop=None):
    """Extract one dense VDF from an already open VLSV reader.

    Plotting and prediction use this convenience entry point when they already
    own a reader. It creates a short-lived extractor and therefore does not
    retain the cached velocity-cell permutation across calls.

    Parameters
    ----------
    reader : analysator.vlsvfile.VlsvReader
        Open VLSV reader containing sparse velocity cells.
    cid : int
        Spatial cell identifier.
    box : int, optional
        Positive peak-centred crop half-width, or ``-1`` for the full mesh.
    pop : str, optional
        Explicit particle population or automatic equivalent-proton choice.

    Returns
    -------
    numpy.ndarray
        Float32 phase-space density in ``s^3 m^-6`` and ``[vx, vy, vz]``
        axis order.
    """

    return VdfExtractor(reader=reader, pop=pop).extract(cid=cid, box=box)


def extract_vdf(file_location, cid, box=-1, pop=None):
    """Open one VLSV file and extract a dense spatial-cell VDF.

    This path serves one-off plotting and inspection calls that do not already
    hold an Analysator reader. Multi-sample workflows use
    :func:`extract_vdf_from_reader` or :class:`VdfExtractor` to avoid repeated
    file opening.

    Parameters
    ----------
    file_location : str or pathlib.Path
        VLSV source path.
    cid : int
        Spatial cell identifier.
    box : int, optional
        Positive peak-centred crop half-width, or ``-1`` for the full mesh.
    pop : str, optional
        Explicit particle population or automatic equivalent-proton choice.

    Returns
    -------
    numpy.ndarray
        Float32 phase-space density in ``s^3 m^-6`` and ``[vx, vy, vz]``
        axis order.
    """

    import analysator as pt

    reader = pt.vlsvfile.VlsvReader(file_location)
    return extract_vdf_from_reader(
        reader=reader,
        cid=cid,
        box=box,
        pop=pop,
    )
