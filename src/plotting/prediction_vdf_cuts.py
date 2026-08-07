"""Extract and render raw VDF planes through each distribution peak.

Sparse source cells are mapped to the current regular ``[vx, vy, vz]`` mesh
without allocating a dense cube. The three plotted cuts use physical velocity
axes and never apply downsampling. Their source-resolved same-cell sparsity
threshold and historical transparent-mask rendering are shared with dataset
figures.
"""

from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg", force=True)

import numpy as np

from src.physics.physical_context import get_sparsity_threshold
from src.plotting.vdf_slices import (
    create_physical_vdf_plane,
    draw_vdf_plane,
    get_vdf_plot_velocity_cell_size_ms,
    prepare_physical_vdf_plane,
)


@dataclass(frozen=True)
class PeakVdfCuts:
    """
    Hold three raw VDF planes crossing the distribution maximum.

    The record is created for one source CID after prediction has completed.
    It owns copied two-dimensional arrays and physical velocity bounds, so
    later Matplotlib rendering does not retain the sparse VLSV cell mapping
    or mutate source values.

    Attributes
    ----------
    cid : int
        Source spatial cell ID.
    peak_index : tuple of int
        Peak index in ``[vx, vy, vz]`` order.
    vx_vy : numpy.ndarray
        Phase-space density in ``s^3 m^-6`` at the peak ``vz`` index.
    vx_vz : numpy.ndarray
        Phase-space density in ``s^3 m^-6`` at the peak ``vy`` index.
    vy_vz : numpy.ndarray
        Phase-space density in ``s^3 m^-6`` at the peak ``vx`` index.
    velocity_extent_mps : tuple of float
        Six velocity-grid bounds in metres per second.
    plotting_threshold : float
        Same-cell source sparsity threshold in ``s^3 m^-6`` used only for
        physical visualization.
    velocity_cell_size_ms : float
        Historical regular ``vy`` cell width in metres per second.
    """

    cid: int
    peak_index: tuple
    vx_vy: np.ndarray
    vx_vz: np.ndarray
    vy_vz: np.ndarray
    velocity_extent_mps: tuple
    plotting_threshold: float
    velocity_cell_size_ms: float


def extract_peak_vdf_cuts(
    extractor,
    cid,
    *,
    sparsity_threshold_variable,
):
    """
    Extract peak-crossing raw VDF planes without allocating a dense cube.

    Combined prediction figures need only three planes, not the complete
    dense velocity cube. This routine reproduces dense C-order
    :func:`numpy.argmax` semantics directly from sparse velocity cells and
    fills just the peak-crossing ``vx-vy``, ``vx-vz``, and ``vy-vz`` cuts.

    Parameters
    ----------
    extractor : VdfExtractor
        Prepared source extractor owning one open VLSV reader, population,
        dtype, and ``[vx, vy, vz]`` velocity grid.
    cid : int
        Positive representative spatial cell ID.
    sparsity_threshold_variable : str
        Exact producer selected once when the prediction source was opened.
        Every representative cell from that source reuses this name.

    Returns
    -------
    PeakVdfCuts
        Three copied raw planes and their physical velocity extent.

    Notes
    -----
    The selected peak is identical to ``numpy.argmax`` on a zero-filled
    dense C-order cube, including ties involving implicit zeros.
    """

    cid = int(cid)
    velocity_grid = extractor.velocity_grid
    shape = tuple(int(value) for value in velocity_grid["shape"])
    velocity_cells = extractor.reader.read_velocity_cells(
        cid,
        extractor.pop,
    )
    cellids = np.fromiter(
        velocity_cells.keys(),
        dtype=np.int64,
        count=len(velocity_cells),
    )
    values = np.fromiter(
        velocity_cells.values(),
        dtype=np.dtype(extractor.dtype),
        count=len(velocity_cells),
    )
    indices = _sparse_velocity_grid_indices(
        extractor,
        cellids,
        velocity_grid,
    )
    linear_indices = np.ravel_multi_index(
        tuple(indices[:, axis] for axis in range(3)),
        shape,
        order="C",
    )
    peak_linear_index = _sparse_dense_argmax(
        linear_indices,
        values,
        int(np.prod(shape)),
    )
    peak_index = tuple(
        int(value)
        for value in np.unravel_index(
            peak_linear_index,
            shape,
            order="C",
        )
    )
    vx_index, vy_index, vz_index = peak_index
    vx_vy = np.zeros(shape[:2], dtype=values.dtype)
    vx_vz = np.zeros((shape[0], shape[2]), dtype=values.dtype)
    vy_vz = np.zeros(shape[1:], dtype=values.dtype)
    if len(values):
        ix, iy, iz = indices.T
        mask = iz == vz_index
        vx_vy[ix[mask], iy[mask]] = values[mask]
        mask = iy == vy_index
        vx_vz[ix[mask], iz[mask]] = values[mask]
        mask = ix == vx_index
        vy_vz[iy[mask], iz[mask]] = values[mask]
    plotting_threshold = get_sparsity_threshold(
        extractor.reader,
        cid,
        variable=sparsity_threshold_variable,
    )[0]
    return PeakVdfCuts(
        cid=cid,
        peak_index=peak_index,
        vx_vy=vx_vy,
        vx_vz=vx_vz,
        vy_vz=vy_vz,
        velocity_extent_mps=tuple(
            float(value) for value in velocity_grid["extent_mps"]
        ),
        plotting_threshold=float(plotting_threshold),
        velocity_cell_size_ms=get_vdf_plot_velocity_cell_size_ms(
            velocity_grid
        ),
    )


def plot_peak_vdf_planes(
    figure,
    axes,
    cuts,
    *,
    velocity_limit_ms,
):
    """
    Plot three physical raw-VDF planes crossing one distribution peak.

    Prediction figures use the same active plane preparation and drawing
    owners as dataset figures, preserving peak slicing, physical axes, and
    logarithmic color normalization. This composition function adds three
    images and one shared colorbar to caller-owned axes but does not save or
    close the figure.

    Parameters
    ----------
    figure : matplotlib.figure.Figure
        Figure that owns the shared colorbar.
    axes : sequence of matplotlib.axes.Axes
        Three axes for ``vx-vy``, ``vx-vz``, and ``vy-vz``.
    cuts : object
        Peak-cut record with three planes and six velocity-grid bounds in
        metres per second.
    velocity_limit_ms : float or None
        Optional symmetric visible velocity limit in metres per second.

    Notes
    -----
    Each plane applies the same historical threshold formula as Stage 6 and
    standalone rendering. Masked values remain transparent over white axes,
    standard neutral Matplotlib spines replace the previous class-colored
    border, and source arrays are not modified.
    """

    extent = cuts.velocity_extent_mps
    planes = (
        (
            create_physical_vdf_plane(
                "xy",
                cuts.vx_vy,
                cuts.peak_index,
                extent,
            ),
            "vx-vy at peak vz",
        ),
        (
            create_physical_vdf_plane(
                "xz",
                cuts.vx_vz,
                cuts.peak_index,
                extent,
            ),
            "vx-vz at peak vy",
        ),
        (
            create_physical_vdf_plane(
                "yz",
                cuts.vy_vz,
                cuts.peak_index,
                extent,
            ),
            "vy-vz at peak vx",
        ),
    )
    images = []
    for axis, (plane, title) in zip(
        axes,
        planes,
    ):
        prepared = prepare_physical_vdf_plane(
            plane.values,
            cuts.plotting_threshold,
            cuts.velocity_cell_size_ms,
        )
        image = draw_vdf_plane(
            axis,
            plane,
            prepared=prepared,
            velocity_limit_ms=velocity_limit_ms,
        )
        axis.set_title(title)
        images.append(image)
    figure.colorbar(
        images[-1],
        ax=list(axes),
        label="f(v)",
        fraction=0.02,
        pad=0.02,
    )


def _sparse_velocity_grid_indices(extractor, cellids, velocity_grid):
    """Map stored velocity-cell IDs to regular ``[vx, vy, vz]`` indices.

    Cell-centre coordinates in metres per second are converted with the
    source mesh extent and cell size. The result supplies the sparse plane
    assembly and C-order peak calculation without constructing a dense cube.

    Parameters
    ----------
    extractor : VdfExtractor
        Open source extractor used to read velocity-cell coordinates.
    cellids : numpy.ndarray
        Stored velocity-space cell IDs for one spatial CID.
    velocity_grid : mapping
        Source shape and ordered extent in metres per second.

    Returns
    -------
    numpy.ndarray
        Integer indices with shape ``(n_velocity_cells, 3)`` and axis order
        ``[vx, vy, vz]``.
    """

    if len(cellids) == 0:
        return np.empty((0, 3), dtype=np.int64)
    shape = np.asarray(velocity_grid["shape"], dtype=np.int64)
    coordinates = np.asarray(
        extractor.reader.get_velocity_cell_coordinates(
            cellids,
            extractor.pop,
        ),
        dtype=np.float64,
    )
    extent = np.asarray(velocity_grid["extent_mps"], dtype=np.float64)
    cell_size = (extent[3:] - extent[:3]) / shape
    first_center = extent[:3] + 0.5 * cell_size
    indices = np.rint(
        (coordinates - first_center) / cell_size
    ).astype(np.int64)
    return indices


def _sparse_dense_argmax(linear_indices, values, total_cells):
    """Return dense C-order argmax semantics from sparse stored values.

    Missing velocity cells represent zeros. The comparison therefore
    includes the first implicit zero and preserves NumPy's first-index
    tie-breaking even when stored values are nonpositive.

    Parameters
    ----------
    linear_indices : numpy.ndarray
        Stored cells expressed as dense C-order indices.
    values : numpy.ndarray
        Sparse physical VDF values aligned with ``linear_indices``.
    total_cells : int
        Complete dense velocity-cell count.

    Returns
    -------
    int
        Linear C-order index of the dense zero-filled maximum.
    """

    if len(values) == 0:
        return 0
    implicit_zero_exists = len(values) < int(total_cells)
    stored_maximum = np.max(values)
    maximum = (
        max(stored_maximum, values.dtype.type(0))
        if implicit_zero_exists
        else stored_maximum
    )
    stored_candidates = linear_indices[values == maximum]
    peak_index = (
        int(np.min(stored_candidates))
        if len(stored_candidates)
        else int(total_cells)
    )
    if implicit_zero_exists and maximum == 0:
        peak_index = min(
            peak_index,
            _first_missing_linear_index(linear_indices, total_cells),
        )
    return peak_index


def _first_missing_linear_index(linear_indices, total_cells):
    """Return the first absent C-order index from sparse occupied indices.

    Parameters
    ----------
    linear_indices : numpy.ndarray
        Unique stored indices for one zero-filled dense cube.
    total_cells : int
        Complete dense cell count retained for call-site clarity.

    Returns
    -------
    int
        First implicit-zero position in dense C order.
    """

    ordered = np.sort(np.asarray(linear_indices, dtype=np.int64))
    if len(ordered) == 0 or ordered[0] != 0:
        return 0
    gaps = np.flatnonzero(np.diff(ordered) > 1)
    if len(gaps):
        return int(ordered[int(gaps[0])] + 1)
    return int(ordered[-1] + 1)
