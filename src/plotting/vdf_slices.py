"""Extract and render shared peak-centred physical dataset VDF planes.

The module owns deterministic dataset-row selection plus the dense physical
plane calculation and drawing reused by extraction Stage 6 and the standalone
VDF command. Saved distributions remain in ``(vx, vy, vz)`` order and
``s^3 m^-6`` units. Every requested plane crosses the same three-dimensional
VDF maximum. Historical plotting preparation uses the source's same-cell
sparsity threshold and velocity-cell width before logarithmic rendering;
masked numerical background remains transparent over the white physical axes
instead of being forced to black. Source arrays remain unchanged.
"""

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)

from matplotlib import colors as matplotlib_colors
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

from src.data.load_velocity_grid import load_velocity_grid
from src.data.dense_vdf import resolve_velocity_population
from src.physics.physical_context import (
    resolve_sparsity_threshold_variable,
)
from src.physics.vlsv_physical_values import read_variable_info
from src.plotting.dataset_colormap import save_figure


DEFAULT_VDF_PLANES = ("xz", "xy", "yz")
DEFAULT_VDF_VELOCITY_LIMIT_MS = 2.0e6
_VELOCITY_EXTENT_INDICES = {
    "x": (0, 3),
    "y": (1, 4),
    "z": (2, 5),
}
_VELOCITY_AXIS_LABELS = {
    "x": r"$v_x$ [km s$^{-1}$]",
    "y": r"$v_y$ [km s$^{-1}$]",
    "z": r"$v_z$ [km s$^{-1}$]",
}


@dataclass(frozen=True)
class PhysicalVdfPlane:
    """Hold one physical VDF plane through a sample's 3-D maximum.

    Attributes
    ----------
    name : str
        Plane abbreviation: ``xz``, ``xy``, or ``yz``.
    peak_index : tuple of int
        Maximum index in saved ``(vx, vy, vz)`` order.
    values : numpy.ndarray
        Copied phase-space density in horizontal-then-vertical axis order and
        ``s^3 m^-6`` units.
    horizontal_axis_name : str
        Physical horizontal velocity component: ``x`` or ``y``.
    vertical_axis_name : str
        Physical vertical velocity component: ``y`` or ``z``.
    horizontal_extent_mps : tuple of float
        Horizontal cell-boundary extent in metres per second.
    vertical_extent_mps : tuple of float
        Vertical cell-boundary extent in metres per second.
    """

    name: str
    peak_index: tuple
    values: np.ndarray
    horizontal_axis_name: str
    vertical_axis_name: str
    horizontal_extent_mps: tuple
    vertical_extent_mps: tuple


@dataclass(frozen=True)
class PreparedPhysicalVdfPlane:
    """Hold historical plot data for one physical VDF plane.

    Attributes
    ----------
    values : numpy.ma.MaskedArray
        Thresholded density multiplied by the velocity-cell width. Values
        below the same-cell source sparsity threshold and all nonpositive
        values are masked.
    norm : matplotlib.colors.LogNorm
        Logarithmic normalization autoscaled from visible values.
    colormap : matplotlib.colors.Colormap
        Unmodified global ``nipy_spectral`` colormap. Its default transparent
        bad color exposes the white axes below the mask.
    """

    values: np.ma.MaskedArray
    norm: matplotlib_colors.LogNorm
    colormap: matplotlib.colors.Colormap


@dataclass(frozen=True)
class VdfSlicePlotResult:
    """Describe one saved multi-sample physical VDF-plane figure.

    Attributes
    ----------
    output_file : pathlib.Path
        Saved combined PNG path.
    sample_indices : tuple of int
        Selected dataset sample indices in stored order.
    cids : tuple of int
        Selected spatial cell IDs in plotted order.
    peak_indices : tuple of tuple
        Per-sample maximum indices in ``(vx, vy, vz)`` order.
    planes : tuple of str
        Requested plane abbreviations in displayed column order.
    """

    output_file: Path
    sample_indices: tuple
    cids: tuple
    peak_indices: tuple
    planes: tuple


def plot_dataset_vdf_slices(
    dataset_dir,
    output_file,
    *,
    sample_indices=(),
    cids=(),
    class_names=(),
    number_of_samples=None,
    planes=DEFAULT_VDF_PLANES,
    velocity_limit_ms=None,
    dpi=150,
    reader_factory=None,
):
    """Save requested peak-crossing VDF planes for selected dataset rows.

    This standalone workflow opens ``X.npy`` read-only, selects metadata rows
    deterministically, and writes one combined PNG with one sample per row and
    one requested plane per column. Drawing is delegated to the same physical
    plane renderer used by extraction Stage 6; Stage 6 requests only ``xz``,
    while omission of the CLI option supplies ``xz``, ``xy``, and ``yz``.

    Parameters
    ----------
    dataset_dir : str or pathlib.Path
        Dataset containing ``X.npy``, ``metadata.csv``, and
        ``velocity_grid.npz``.
    output_file : str or pathlib.Path
        Combined PNG destination.
    sample_indices : sequence of int, optional
        Dataset ``sample_index`` values to select.
    cids : sequence of int, optional
        Spatial cell IDs to select.
    class_names : sequence of str, optional
        Physical class names to select.
    number_of_samples : int, optional
        Deterministic maximum number of selected rows.
    planes : sequence of str, optional
        Physical planes in displayed order. Supported values are ``xz``,
        ``xy``, and ``yz``.
    velocity_limit_ms : float, optional
        Shared symmetric velocity limit in metres per second. The Stage 6
        limit is used when omitted.
    dpi : int, optional
        PNG resolution in dots per inch.
    reader_factory : callable, optional
        VLSV reader constructor. The production path uses Analysator; focused
        tests may supply the same reader interface for bounded source data.

    Returns
    -------
    VdfSlicePlotResult
        Output path, selected identities, peak indices, and plane order.
    """

    dataset_dir = Path(dataset_dir)
    output_file = Path(output_file)
    X = np.load(
        dataset_dir / "X.npy",
        mmap_mode="r",
        allow_pickle=False,
    )
    try:
        return _plot_loaded_dataset_vdf_slices(
            X,
            dataset_dir,
            output_file,
            sample_indices=sample_indices,
            cids=cids,
            class_names=class_names,
            number_of_samples=number_of_samples,
            planes=tuple(planes),
            velocity_limit_ms=velocity_limit_ms,
            dpi=dpi,
            reader_factory=reader_factory,
        )
    finally:
        _close_memmap(X)


def select_vdf_samples(
    metadata,
    *,
    sample_indices=(),
    cids=(),
    class_names=(),
    number_of_samples=None,
):
    """Select deterministic dataset rows while retaining raw-array alignment.

    Selection works on saved metadata values, then stores each row's original
    array position in ``_array_row`` before sorting by ``sample_index``. This
    mapping prevents CID or class selection from pairing metadata with a
    different physical VDF.

    Parameters
    ----------
    metadata : pandas.DataFrame
        Saved metadata in raw-array row order.
    sample_indices : sequence of int, optional
        Dataset sample indices.
    cids : sequence of int, optional
        Spatial cell IDs.
    class_names : sequence of str, optional
        Physical class names.
    number_of_samples : int, optional
        Maximum selected rows.

    Returns
    -------
    pandas.DataFrame
        Rows ordered by ``sample_index`` with ``_array_row`` retaining the
        aligned raw-array row.
    """

    sample_indices = tuple(int(value) for value in sample_indices)
    cids = tuple(int(value) for value in cids)
    class_names = tuple(str(value) for value in class_names)
    if number_of_samples is not None:
        number_of_samples = int(number_of_samples)

    selected = metadata.copy()
    selected["_array_row"] = np.arange(len(selected), dtype=np.int64)
    if sample_indices:
        selected = selected[
            selected["sample_index"].astype(int).isin(sample_indices)
        ]
    elif cids:
        selected = selected[selected["cid"].astype(int).isin(cids)]
    elif class_names:
        selected = selected[
            selected["class_name"].astype(str).isin(class_names)
        ]
    elif number_of_samples is None:
        number_of_samples = 1

    if class_names and number_of_samples is None:
        selected = selected.drop_duplicates("class_name", keep="first")
    selected = selected.sort_values("sample_index", kind="stable")
    if number_of_samples is not None:
        selected = selected.iloc[:number_of_samples]
    return selected


def extract_peak_vdf_planes(vdf, velocity_extent_mps, planes):
    """Extract requested physical planes through one complete VDF maximum.

    The physical VDF has shape ``(vx, vy, vz)``. Its three-dimensional peak
    is calculated once. ``xz`` fixes ``vy`` at ``peak_vy`` and returns
    ``vdf[:, peak_vy, :]``; ``xy`` fixes ``vz`` at ``peak_vz`` and returns
    ``vdf[:, :, peak_vz]``; ``yz`` fixes ``vx`` at ``peak_vx`` and returns
    ``vdf[peak_vx, :, :]``. Peak slicing keeps displaced populations visible
    when a geometric mesh-centre plane would miss them.

    Parameters
    ----------
    vdf : numpy.ndarray
        Physical phase-space density in ``s^3 m^-6`` with shape
        ``(vx, vy, vz)``.
    velocity_extent_mps : sequence of float
        Cell-boundary extent ``[vxmin, vymin, vzmin, vxmax, vymax, vzmax]``
        in metres per second.
    planes : sequence of str
        Requested ``xz``, ``xy``, or ``yz`` planes in output order.

    Returns
    -------
    tuple of PhysicalVdfPlane
        Copied plane values, shared peak index, physical axis names, and
        matching velocity extents in requested order.

    Notes
    -----
    Tied maxima use the first stable C-order ``numpy.argmax`` position. The
    source VDF remains unchanged.
    """

    vdf = np.asanyarray(vdf)
    peak_index = tuple(
        int(value)
        for value in np.unravel_index(
            int(np.argmax(vdf)),
            vdf.shape,
            order="C",
        )
    )
    peak_vx, peak_vy, peak_vz = peak_index
    plane_slices = {
        "xz": (slice(None), peak_vy, slice(None)),
        "xy": (slice(None), slice(None), peak_vz),
        "yz": (peak_vx, slice(None), slice(None)),
    }
    return tuple(
        create_physical_vdf_plane(
            plane_name,
            np.array(vdf[plane_slices[plane_name]], copy=True),
            peak_index,
            velocity_extent_mps,
        )
        for plane_name in planes
    )


def create_physical_vdf_plane(
    plane_name,
    values,
    peak_index,
    velocity_extent_mps,
):
    """Describe one physical VDF plane in the shared display orientation.

    Dense dataset extraction and sparse prediction cuts both call this
    constructor so a plane name has one active mapping to horizontal and
    vertical velocity extents. Central ownership prevents the two figure
    paths from compensating for orientation with different transposes.

    Parameters
    ----------
    plane_name : {"xz", "xy", "yz"}
        Physical plane abbreviation. Its first component owns the horizontal
        array dimension and its second component owns the vertical dimension.
    values : numpy.ndarray
        Plane values in horizontal-then-vertical order.
    peak_index : tuple of int
        Full VDF maximum in saved ``(vx, vy, vz)`` index order.
    velocity_extent_mps : sequence of float
        Cell-boundary extent ``[vxmin, vymin, vzmin, vxmax, vymax, vzmax]``
        in metres per second.

    Returns
    -------
    PhysicalVdfPlane
        Copied values with physical axis names and matching velocity extents.
    """

    extent = tuple(float(value) for value in velocity_extent_mps)
    horizontal_name, vertical_name = tuple(str(plane_name))
    horizontal_indices = _VELOCITY_EXTENT_INDICES[horizontal_name]
    vertical_indices = _VELOCITY_EXTENT_INDICES[vertical_name]
    return PhysicalVdfPlane(
        name=str(plane_name),
        peak_index=tuple(int(value) for value in peak_index),
        values=np.array(values, copy=True),
        horizontal_axis_name=horizontal_name,
        vertical_axis_name=vertical_name,
        horizontal_extent_mps=tuple(
            extent[index] for index in horizontal_indices
        ),
        vertical_extent_mps=tuple(
            extent[index] for index in vertical_indices
        ),
    )


def get_vdf_plot_velocity_cell_size_ms(velocity_grid):
    """Return the historical VDF plotting cell width along ``vy``.

    Historical physical-slice preparation multiplied density by the regular
    velocity-cell width calculated from the ``vy`` bounds. The current saved
    grid stores the same extent and ``(vx, vy, vz)`` shape, so Stage 6,
    standalone plotting, and prediction panels can reproduce that calculation
    without changing physical coordinates.

    Parameters
    ----------
    velocity_grid : mapping
        Saved or source velocity grid with ``shape`` and ``extent_mps``;
        extents are in metres per second.

    Returns
    -------
    float
        Regular ``vy`` cell width in metres per second.
    """

    extent = np.asarray(velocity_grid["extent_mps"], dtype=np.float64)
    shape = np.asarray(velocity_grid["shape"], dtype=np.int64)
    return float((extent[4] - extent[1]) / shape[1])


def load_vdf_plotting_thresholds(metadata, reader_factory=None):
    """Read source-resolved sparsity thresholds for selected dataset rows.

    Historical VDF figures read the plotting threshold from the source VLSV
    cell rather than deriving it from the visible plane. Current metadata
    retains each source path and CID. Rows are grouped by source so one reader
    resolves the active velocity population and threshold producer once, then
    one batched read serves every unique CID in that file. The producer order
    is ``MinValue``, ``<population>/MinValue``,
    ``<population>/EffectiveSparsityThreshold``, then
    ``<population>/vg_effectivesparsitythreshold`` as used by Analysator.
    Thresholds return in exact metadata-row order.

    Parameters
    ----------
    metadata : pandas.DataFrame
        Selected rows containing ``file_location`` and ``cid``.
    reader_factory : callable, optional
        VLSV reader constructor. Analysator is used by production callers.

    Returns
    -------
    tuple of float
        Same-cell VDF sparsity thresholds in ``s^3 m^-6`` aligned with
        ``metadata``.

    Notes
    -----
    Stage 6 and standalone dataset VDF plots share this source-loading path.
    Producer selection and threshold reads do not modify saved VDF values or
    the historical physical-plane rendering calculation.
    """

    if reader_factory is None:
        import analysator as pt

        reader_factory = pt.vlsvfile.VlsvReader
    rows = tuple(metadata.itertuples(index=False))
    positions_by_file = {}
    for position, row in enumerate(rows):
        positions_by_file.setdefault(
            str(row.file_location),
            [],
        ).append(position)
    aligned = [None] * len(rows)
    for file_location, positions in positions_by_file.items():
        reader = reader_factory(file_location)
        population = resolve_velocity_population(reader)
        threshold_variable = resolve_sparsity_threshold_variable(
            reader,
            population,
        )
        unique_cids = tuple(
            dict.fromkeys(int(rows[position].cid) for position in positions)
        )
        values, _ = read_variable_info(
            reader,
            threshold_variable,
            np.asarray(unique_cids, dtype=np.int64),
        )
        thresholds_by_cid = dict(
            zip(
                unique_cids,
                np.asarray(values).reshape(-1),
            )
        )
        for position in positions:
            cid = int(rows[position].cid)
            aligned[position] = float(thresholds_by_cid[cid])
    return tuple(aligned)


def prepare_physical_vdf_plane(
    physical_plane,
    plotting_threshold,
    velocity_cell_size_ms,
):
    """Prepare one physical VDF plane for logarithmic display.

    This function ports the historical physical-slice transformation into
    the current plotting package. Density is converted to the historical
    plotted quantity by multiplying by the regular ``vy`` cell width. Values
    strictly below the sample's same-cell source sparsity threshold are
    replaced by zero, then all nonpositive values are masked. A value exactly
    at the threshold remains visible. The unmodified ``nipy_spectral`` bad
    color is transparent, so masked numerical background reveals white axes
    instead of becoming an opaque black field.

    Parameters
    ----------
    physical_plane : numpy.ndarray
        Peak-centred physical density plane in horizontal-then-vertical order
        and ``s^3 m^-6`` units.
    plotting_threshold : float
        Same-cell VLSV sparsity threshold in ``s^3 m^-6``.
    velocity_cell_size_ms : float
        Historical regular ``vy`` cell width in metres per second.

    Returns
    -------
    PreparedPhysicalVdfPlane
        Masked plot values, autoscaled logarithmic normalization, and an
        historical colormap.

    Notes
    -----
    This transformation is visualization-only. It never changes ``X.npy`` or
    a PCA, CNN, autoencoder, or Hermite representation.
    """

    velocity_cell_size_ms = float(velocity_cell_size_ms)
    plot_values = (
        np.asarray(physical_plane, dtype=np.float32)
        * velocity_cell_size_ms
    )
    thresholded = np.where(
        plot_values < float(plotting_threshold) * velocity_cell_size_ms,
        0,
        plot_values,
    )
    prepared = np.ma.masked_less_equal(thresholded, 0)
    norm = matplotlib_colors.LogNorm()
    norm.autoscale_None(prepared)
    return PreparedPhysicalVdfPlane(
        values=prepared,
        norm=norm,
        colormap=matplotlib.colormaps["nipy_spectral"],
    )


def draw_vdf_plane(
    axis,
    plane,
    *,
    prepared,
    velocity_limit_ms=None,
):
    """Draw one prepared physical VDF plane on caller-owned axes.

    Stage 6, standalone figures, and physical raw prediction panels use this
    renderer. The thresholded input is transposed once from the current
    horizontal-then-vertical convention for Matplotlib. Physical velocity
    extents are converted from metres per second to kilometres per second.
    Historical logarithmic colors leave masked background transparent over
    white axes, and the returned image supplies the caller's colorbar.

    Parameters
    ----------
    axis : matplotlib.axes.Axes
        Axes receiving the physical VDF image.
    plane : PhysicalVdfPlane
        Peak-crossing values and their physical velocity-axis ownership.
    prepared : PreparedPhysicalVdfPlane
        Historical thresholded values, logarithmic normalization, and
        unmodified colormap.
    velocity_limit_ms : float, optional
        Symmetric visible velocity limit in metres per second. The common
        Stage 6 limit is used when omitted.

    Returns
    -------
    matplotlib.image.AxesImage
        Image used by the caller to create an ``f(v)`` colorbar.
    """

    horizontal_extent = tuple(
        float(value) / 1000.0 for value in plane.horizontal_extent_mps
    )
    vertical_extent = tuple(
        float(value) / 1000.0 for value in plane.vertical_extent_mps
    )
    image = axis.imshow(
        prepared.values.T,
        origin="lower",
        extent=(*horizontal_extent, *vertical_extent),
        aspect="equal",
        cmap=prepared.colormap,
        norm=prepared.norm,
    )
    format_vdf_plane_axes(
        axis,
        plane,
        velocity_limit_ms=velocity_limit_ms,
    )
    return image


def format_vdf_plane_axes(axis, plane, *, velocity_limit_ms=None):
    """Format physical VDF axes with the shared extraction geometry.

    Stage 6, standalone VDF plots, prediction panels, and raw autoencoder
    reconstruction errors use this operation so all physical ``vx-vz``
    panels share kilometre-per-second coordinates, the configured symmetric
    velocity limit, square aspect, and grey Cartesian grid. Image drawing and
    color normalization remain the responsibility of the caller.

    Parameters
    ----------
    axis : matplotlib.axes.Axes
        Axes receiving physical velocity coordinates.
    plane : PhysicalVdfPlane
        Plane whose horizontal and vertical physical axes determine labels.
    velocity_limit_ms : float, optional
        Symmetric visible velocity limit in metres per second. The common
        Stage 6 limit is used when omitted.
    """

    if velocity_limit_ms is None:
        velocity_limit_ms = DEFAULT_VDF_VELOCITY_LIMIT_MS
    limit_kmps = float(velocity_limit_ms) / 1000.0
    axis.set_xlim(-limit_kmps, limit_kmps)
    axis.set_ylim(-limit_kmps, limit_kmps)
    axis.set_box_aspect(1)
    axis.set_xlabel(_VELOCITY_AXIS_LABELS[plane.horizontal_axis_name])
    axis.set_ylabel(_VELOCITY_AXIS_LABELS[plane.vertical_axis_name])
    axis.grid(color="gray", axis="both")


def _plot_loaded_dataset_vdf_slices(
    X,
    dataset_dir,
    output_file,
    *,
    sample_indices,
    cids,
    class_names,
    number_of_samples,
    planes,
    velocity_limit_ms,
    dpi,
    reader_factory,
):
    """Render selected rows from one already-open raw dataset array."""

    metadata = pd.read_csv(dataset_dir / "metadata.csv")
    selected = select_vdf_samples(
        metadata,
        sample_indices=sample_indices,
        cids=cids,
        class_names=class_names,
        number_of_samples=number_of_samples,
    )
    velocity_grid = load_velocity_grid(dataset_dir)
    extent_mps = tuple(
        float(value) for value in velocity_grid["extent_mps"]
    )
    velocity_cell_size_ms = get_vdf_plot_velocity_cell_size_ms(
        velocity_grid
    )
    plotting_thresholds = load_vdf_plotting_thresholds(
        selected,
        reader_factory=reader_factory,
    )
    figure, axes = plt.subplots(
        nrows=len(selected),
        ncols=len(planes),
        squeeze=False,
        figsize=(4.6 * len(planes), 4.1 * len(selected)),
        constrained_layout=True,
    )
    peak_indices = []
    try:
        for plot_row, ((_, row), plotting_threshold) in enumerate(
            zip(selected.iterrows(), plotting_thresholds)
        ):
            extracted = extract_peak_vdf_planes(
                X[int(row["_array_row"])],
                extent_mps,
                planes,
            )
            peak_indices.append(extracted[0].peak_index)
            images = []
            for axis, plane in zip(axes[plot_row], extracted):
                prepared = prepare_physical_vdf_plane(
                    plane.values,
                    plotting_threshold,
                    velocity_cell_size_ms,
                )
                images.append(
                    draw_vdf_plane(
                        axis,
                        plane,
                        prepared=prepared,
                        velocity_limit_ms=velocity_limit_ms,
                    )
                )
                axis.set_title(
                    f"plane={plane.name}, "
                    f"sample={int(row['sample_index'])}, "
                    f"timestep={int(row['timestep'])}, "
                    f"cid={int(row['cid'])}, "
                    f"class={row['class_name']}",
                    fontsize=9,
                )
            colorbar = figure.colorbar(
                images[-1],
                ax=list(axes[plot_row]),
                orientation="vertical",
                fraction=0.025,
                pad=0.02,
            )
            colorbar.set_label("f(v)")
        save_figure(figure, output_file, dpi=int(dpi))
    finally:
        plt.close(figure)

    return VdfSlicePlotResult(
        output_file=output_file,
        sample_indices=tuple(
            int(value) for value in selected["sample_index"]
        ),
        cids=tuple(int(value) for value in selected["cid"]),
        peak_indices=tuple(peak_indices),
        planes=tuple(planes),
    )


def _close_memmap(array):
    """Close one read-only NumPy memory map without writing it."""

    mmap_object = getattr(array, "_mmap", None)
    if mmap_object is not None:
        mmap_object.close()
