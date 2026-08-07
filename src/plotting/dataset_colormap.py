"""Render the shared Stage 6 and standalone dataset spatial colormap.

Both callers draw bulk ``v_x`` with a zero-centred red-white-blue scale,
thin black magnetic streamlines, yellow VDF-carrying cells, blue source
coordinates, red selected cells, and exact current X/O search geometry.
Stage 6 passes the point records retained during planning; the standalone
command reconstructs those records through the same active point-label path.
Plotting never changes sample membership or a saved scientific array.
"""

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

from src.data.step_01_open_vlsv_files import resolve_vlsv_source
from src.data.step_02_find_vdf_cells import (
    get_nearest_vdf_cellid,
    get_vdf_cells_with_coords_re,
)
from src.plotting.dataset_selection_colormap import (
    DATASET_CANDIDATE_MARKER_SIZE,
    add_dataset_selection_legend,
    plot_dataset_selected_cells,
    plot_dataset_source_coordinates,
    render_dataset_velocity_background,
)
from src.plotting.point_selection_areas import (
    draw_point_selection_areas,
)


_ALL_VDF_CELL_STYLE = {
    "marker": ".",
    "s": DATASET_CANDIDATE_MARKER_SIZE,
    "color": "gold",
    "linewidths": 0,
}
_DEFAULT_SPATIAL_BOX_RE = (-30.0, 30.0, -13.0, 13.0)
_DEFAULT_VELOCITY_LIMIT_MS = 1.5e6


@dataclass(frozen=True)
class DatasetColormapResult:
    """
    Describe one saved dataset colormap.

    The plotting command returns this immutable saved-result record after the
    figure has been encoded and closed. It owns paths and scalar counts only;
    it does not retain the VLSV reader, Matplotlib figure, or metadata table.

    Attributes
    ----------
    output_file : pathlib.Path
        Saved PNG path.
    source_vlsv : pathlib.Path
        VLSV source used for the background.
    timestep : int
        Plotted simulation timestep.
    labeled_cell_count : int
        Number of dataset rows shown.
    source_vdf_cell_count : int
        Number of source VDF cells shown.
    class_ids : tuple of int
        Physical class IDs in stable order.
    """

    output_file: Path
    source_vlsv: Path
    timestep: int
    labeled_cell_count: int
    source_vdf_cell_count: int
    class_ids: tuple


@dataclass(frozen=True)
class _SpatialPlotConfig:
    """Supply dataset colormap options to the shared spatial renderer.

    This private adapter presents the spatial values consumed by the shared
    Analysator x-velocity renderer without introducing a plotting context
    abstraction.
    """

    spatial_box_re: tuple | None
    color_limits: tuple | None
    show_streamlines: bool


def plot_dataset_colormap(
    dataset_dir,
    output_file,
    *,
    extraction_config,
    timestep=None,
    source_vlsv=None,
    spatial_box_re=None,
    velocity_limit_ms=None,
    spatial_slice_y_re=None,
    show_all_vdf_cells=True,
    show_labeled_cells=True,
    show_class_legend=True,
    show_magnetic_streamlines=True,
    dpi=150,
    reader_factory=None,
    spatial_renderer=None,
    metadata=None,
    x_point_records=None,
    o_point_records=None,
):
    """
    Plot one shared Stage 6-style dataset spatial colormap.

    Extraction Stage 6 and the standalone command both call this function.
    The fixed background is physical bulk ``v_x`` in metres per second with
    symmetric limits. Yellow, blue, and red encode candidate VDF cells,
    configured or detected source coordinates, and selected dataset cells;
    marker shapes retain current class or point identity. The exact active
    X/O selection geometry is drawn in the same x-z axes in Earth radii.

    Parameters
    ----------
    dataset_dir : str or pathlib.Path
        Dataset containing ``metadata.csv``.
    output_file : str or pathlib.Path
        PNG path written by Matplotlib.
    extraction_config : dict
        Extraction configuration used for source-path fallback, configured
        manual coordinates, and exact current X/O point records.
    timestep : int, optional
        Timestep to plot. The first stored timestep is used when omitted.
    source_vlsv : str or pathlib.Path, optional
        Explicit VLSV source, taking precedence over saved provenance.
    spatial_box_re : sequence of float, optional
        X-z bounds ``[xmin, xmax, zmin, zmax]`` in Earth radii. The fixed
        Stage 6 bounds are used when omitted.
    velocity_limit_ms : float, optional
        Positive symmetric bulk-x-velocity limit in metres per second. The
        fixed Stage 6 limit is used when omitted.
    spatial_slice_y_re : float, optional
        Y coordinate in Earth radii for a three-dimensional source.
    show_all_vdf_cells : bool, optional
        Whether to show every source VDF-carrying cell.
    show_labeled_cells : bool, optional
        Whether to show red selected cells and blue source coordinates with
        current class or point marker shapes.
    show_class_legend : bool, optional
        Whether to include role, current class/point, and search-area legend
        entries below the spatial axes.
    show_magnetic_streamlines : bool, optional
        Whether to request magnetic streamlines when available.
    dpi : int, optional
        Output resolution.
    reader_factory : callable, optional
        VLSV reader factory, primarily for focused tests.
    spatial_renderer : callable, optional
        Spatial rendering function with the shared dataset-renderer API.
    metadata : pandas.DataFrame, optional
        Already-loaded metadata retained by Stage 6 across timestep frames.
    x_point_records : sequence of dict or None, optional
        Accepted X points whose physical selection polygons are drawn.
        ``None`` reconstructs current accepted point records for the
        standalone command.
    o_point_records : sequence of dict or None, optional
        Accepted O points whose physical contours and manual boxes are drawn.
        ``None`` reconstructs current accepted point records for the
        standalone command.

    Returns
    -------
    DatasetColormapResult
        Published path and plotted-row summary.

    """

    dataset_dir = Path(dataset_dir)
    output_file = Path(output_file)
    if metadata is None:
        metadata = pd.read_csv(dataset_dir / "metadata.csv")
    timestep, rows = _select_timestep_rows(metadata, timestep)
    source_vlsv = resolve_dataset_source_vlsv(
        rows,
        timestep=timestep,
        source_vlsv=source_vlsv,
        extraction_config=extraction_config,
    )
    if reader_factory is None:
        reader_factory = _analysator_reader
    reader = reader_factory(str(source_vlsv))
    resolved_source = resolve_vlsv_source(reader)
    vdf_cellids, vdf_coords_re = get_vdf_cells_with_coords_re(
        reader,
        pop=resolved_source.population,
    )

    class_ids = tuple(
        sorted(metadata["class_id"].astype(int).unique())
    )
    if spatial_box_re is None:
        spatial_box_re = _DEFAULT_SPATIAL_BOX_RE
    if velocity_limit_ms is None:
        velocity_limit_ms = _DEFAULT_VELOCITY_LIMIT_MS
    velocity_limit_ms = float(velocity_limit_ms)
    plot_config = _SpatialPlotConfig(
        spatial_box_re=_optional_bounds(spatial_box_re),
        color_limits=(-velocity_limit_ms, velocity_limit_ms),
        show_streamlines=bool(show_magnetic_streamlines),
    )
    cut_y_re = (
        float(spatial_slice_y_re)
        if spatial_slice_y_re is not None
        else float(np.median(rows["vdf_y_re"].to_numpy(dtype=float)))
    )
    dpi = int(dpi)

    if spatial_renderer is None:
        import analysator as pt

        _ = pt.plot.plot_colormap
        spatial_renderer = render_dataset_velocity_background

    if x_point_records is None and o_point_records is None:
        x_point_records, o_point_records = _create_dataset_point_records(
            extraction_config,
            timestep,
            reader,
            vdf_cellids,
            vdf_coords_re,
        )
    x_point_records = tuple(x_point_records)
    o_point_records = tuple(o_point_records)

    figure, axis = plt.subplots(figsize=(8.0, 6.5))
    figure.subplots_adjust(bottom=0.22)
    source_vdf_cell_count = 0
    try:
        spatial_renderer(
            axis,
            reader=reader,
            resolved_source=resolved_source,
            plotting_config=plot_config,
            cut_y_re=cut_y_re,
        )
        if show_all_vdf_cells:
            coords_re = _inside_spatial_box(
                vdf_coords_re,
                plot_config.spatial_box_re,
            )
            source_vdf_cell_count = int(len(coords_re))
            if source_vdf_cell_count:
                axis.scatter(
                    coords_re[:, 0],
                    coords_re[:, 2],
                    zorder=3,
                    **_ALL_VDF_CELL_STYLE,
                )
        source_handles = ()
        selected_handles = ()
        if show_labeled_cells:
            selected_handles = plot_dataset_selected_cells(axis, rows)
            source_handles = plot_dataset_source_coordinates(
                axis,
                configured_sources=_accepted_configured_sources(
                    extraction_config,
                    rows,
                    vdf_cellids,
                    vdf_coords_re,
                ),
                x_point_records=x_point_records,
                o_point_records=o_point_records,
            )
        selection_handles = draw_point_selection_areas(
            axis,
            config=extraction_config,
            x_point_records=x_point_records,
            o_point_records=o_point_records,
        )
        if show_class_legend:
            add_dataset_selection_legend(
                figure,
                include_all_cells=bool(show_all_vdf_cells),
                all_cell_style=_ALL_VDF_CELL_STYLE,
                source_handles=source_handles,
                selected_handles=selected_handles,
                selection_handles=selection_handles,
            )
        axis.set_xlabel("x [$R_E$]")
        axis.set_ylabel("z [$R_E$]")
        if plot_config.spatial_box_re is not None:
            xmin, xmax, zmin, zmax = plot_config.spatial_box_re
            axis.set_xlim(xmin, xmax)
            axis.set_ylim(zmin, zmax)
        axis.set_aspect("equal", adjustable="box")
        axis.set_title(f"timestep = {timestep}")
        save_figure(
            figure,
            output_file,
            dpi=dpi,
            tight=False,
        )
    finally:
        plt.close(figure)

    return DatasetColormapResult(
        output_file=output_file,
        source_vlsv=source_vlsv,
        timestep=timestep,
        labeled_cell_count=int(len(rows)) if show_labeled_cells else 0,
        source_vdf_cell_count=source_vdf_cell_count,
        class_ids=class_ids,
    )


def resolve_dataset_source_vlsv(
    metadata_rows,
    *,
    timestep,
    source_vlsv=None,
    extraction_config=None,
):
    """
    Resolve one VLSV path from explicit and stored provenance.

    Colormap rendering needs the original bulk source only for the
    background and optional complete VDF-cell overlay. Resolution therefore
    follows a deterministic priority: explicit command argument, first
    nonblank metadata path, then the extraction template for the selected
    timestep.

    Parameters
    ----------
    metadata_rows : pandas.DataFrame
        Rows selected for one timestep.
    timestep : int
        Selected simulation timestep.
    source_vlsv : str or pathlib.Path, optional
        Explicit path with highest priority.
    extraction_config : dict, optional
        Extraction configuration used as the final fallback.

    Returns
    -------
    pathlib.Path
        VLSV source path selected by explicit, metadata, then config
        priority.
    """

    if source_vlsv is not None:
        return Path(source_vlsv)

    metadata_sources = _metadata_source_paths(metadata_rows)
    if metadata_sources:
        return metadata_sources[0]
    if extraction_config is not None:
        template = extraction_config.get("file_template_bulk")
        if template:
            return Path(str(template).format(timestep=timestep))
    return metadata_sources[0]


def _create_dataset_point_records(
    config,
    timestep,
    reader,
    vdf_cellids,
    vdf_coords_re,
):
    """Reconstruct accepted X/O records for standalone colormap rendering.

    The standalone command has the extraction configuration and open bulk
    reader but not Stage 3's in-memory plotting handoff. Calling the active
    point-label owner reproduces the same detection, conflict handling, and
    physical geometry without copying those scientific calculations into
    plotting code.

    Parameters
    ----------
    config : dict
        Current extraction configuration, including the matching flux source
        and X/O selection settings.
    timestep : int
        Simulation timestep represented by the open reader.
    reader : analysator.vlsvfile.VlsvReader
        Already-open bulk source selected by standalone source precedence.
    vdf_cellids : numpy.ndarray
        Stored VDF-carrying cell IDs in current source order.
    vdf_coords_re : numpy.ndarray
        Matching VDF-cell centres in Earth radii.

    Returns
    -------
    x_point_records : tuple of dict
        Accepted current X-point records with exact selection geometry.
    o_point_records : tuple of dict
        Accepted current O-point records with exact selection geometry.
    """

    from src.physics.point_labels import create_point_label_data
    from src.physics.point_selection import (
        create_accepted_point_selection_results,
    )

    label_data = create_point_label_data(
        config=config,
        timestep=timestep,
        reader=reader,
        include_physical_geometry=True,
    )
    selection_data = create_accepted_point_selection_results(
        config=config,
        point_records=label_data["point_labeled_coords"],
        vdf_cellids=vdf_cellids,
        vdf_coords_re=vdf_coords_re,
    )
    accepted = tuple(
        result["point_record"]
        for result in selection_data["accepted_results"]
    )
    return (
        tuple(
            record
            for record in accepted
            if record["point_kind"] == "x"
        ),
        tuple(
            record
            for record in accepted
            if record["point_kind"] == "o"
        ),
    )


def _accepted_configured_sources(config, rows, vdf_cellids, vdf_coords_re):
    """Return configured coordinates whose selected sample was retained.

    Configured plasma-region coordinates select their nearest VDF-carrying
    cell during Stage 3. This projection repeats that direct lookup against
    the already-loaded source cell arrays, then keeps only the first
    configured coordinate for each final ``(class_name, cid)`` row. It lets
    the shared colormap draw a blue coordinate only when its paired red VDF
    sample actually exists.

    Parameters
    ----------
    config : dict
        Extraction configuration containing manual class coordinates.
    rows : pandas.DataFrame
        Final saved metadata rows for one timestep.
    vdf_cellids : numpy.ndarray
        Stored VDF-carrying cell IDs in current source order.
    vdf_coords_re : numpy.ndarray
        Matching VDF-cell centres in Earth radii.

    Returns
    -------
    dict
        Class names mapped to accepted configured coordinates in source order.
    """

    saved_class_cellids = {
        (str(row.class_name), int(row.cid))
        for row in rows[["class_name", "cid"]].itertuples(index=False)
    }
    configured_sources = {}
    seen_class_cellids = set()
    for class_name, coords_re in config["class_coords_re"].items():
        for coord_re in coords_re:
            cid = get_nearest_vdf_cellid(
                coord_re,
                vdf_cellids,
                vdf_coords_re,
            )
            class_cellid = (str(class_name), int(cid))
            if (
                class_cellid in saved_class_cellids
                and class_cellid not in seen_class_cellids
            ):
                configured_sources.setdefault(str(class_name), []).append(
                    coord_re
                )
                seen_class_cellids.add(class_cellid)
    return configured_sources


def save_figure(figure, output_file, *, dpi, tight=True):
    """
    Save one Matplotlib figure to the requested path.

    Dataset colormap and VDF-slice commands share this direct PNG saver.
    It creates only the parent directory and delegates image encoding to
    Matplotlib; callers remain responsible for closing the figure.

    Parameters
    ----------
    figure : matplotlib.figure.Figure
        Populated figure to encode.
    output_file : str or pathlib.Path
        Requested ``.png`` path.
    dpi : int
        Output resolution in dots per inch.
    tight : bool, optional
        Whether to crop to artist bounds. Stage 6 disables cropping so every
        animation frame retains the same fixed canvas dimensions.

    Notes
    -----
    The parent directory is created before Matplotlib writes the image.
    Existing files follow Matplotlib's ordinary replacement behavior.
    """

    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    bbox_inches = "tight" if tight else None
    figure.savefig(output_file, dpi=dpi, bbox_inches=bbox_inches)


def _select_timestep_rows(metadata, timestep):
    """Return one timestep and its metadata rows in saved order.

    Parameters
    ----------
    metadata : pandas.DataFrame
        Complete saved dataset metadata.
    timestep : int or None
        Requested simulation timestep, or ``None`` for the first saved row.

    Returns
    -------
    selected_timestep : int
        Timestep represented by the returned rows.
    rows : pandas.DataFrame
        Independent table containing only that timestep.
    """

    available = metadata["timestep"].astype(int)
    if timestep is None:
        timestep = int(available.iloc[0])
    else:
        timestep = int(timestep)
    rows = metadata.loc[available == timestep].copy()
    return timestep, rows


def _metadata_source_paths(rows):
    """Return unique nonblank VLSV paths in first-row order.

    Parameters
    ----------
    rows : pandas.DataFrame
        Metadata rows selected for one timestep.

    Returns
    -------
    list of pathlib.Path
        Source paths without reordering or duplication.
    """

    paths = []
    seen = set()
    for value in rows["file_location"]:
        text = str(value).strip()
        if text and text not in seen:
            paths.append(Path(text))
            seen.add(text)
    return paths


def _inside_spatial_box(coords_re, spatial_box_re):
    """Select coordinates inside optional x-z Earth-radius bounds.

    Parameters
    ----------
    coords_re : numpy.ndarray
        Cell-centre coordinates with shape ``(n_cells, 3)`` in Earth radii.
    spatial_box_re : sequence of float or None
        Bounds ``(xmin, xmax, zmin, zmax)`` in Earth radii.

    Returns
    -------
    numpy.ndarray
        Original or selected ``(n_visible, 3)`` coordinate rows.
    """

    coords_re = np.asarray(coords_re, dtype=float)
    if spatial_box_re is None or len(coords_re) == 0:
        return coords_re
    xmin, xmax, zmin, zmax = spatial_box_re
    selected = (
        (coords_re[:, 0] >= xmin)
        & (coords_re[:, 0] <= xmax)
        & (coords_re[:, 2] >= zmin)
        & (coords_re[:, 2] <= zmax)
    )
    return coords_re[selected]


def _optional_bounds(values):
    """Return immutable floating-point plot bounds or ``None``.

    Parameters
    ----------
    values : sequence or None
        Optional limits in the caller-defined order.

    Returns
    -------
    tuple of float or None
        Converted limit values.
    """

    if values is None:
        return None
    return tuple(float(value) for value in values)


def _analysator_reader(file_path):
    """Open one VLSV source with the installed Analysator reader.

    Parameters
    ----------
    file_path : str or pathlib.Path
        Bulk VLSV file used as the spatial background.

    Returns
    -------
    object
        Open Analysator ``VlsvReader`` consumed by the renderer.
    """

    import analysator as pt

    return pt.vlsvfile.VlsvReader(str(file_path))
