"""Compose bulk-velocity prediction maps and peak-centred source-VDF panels.

Coordinate and region workflows call this module only after CSV output. The
spatial map uses physical bulk ``v_x`` in metres per second and a dedicated
legend row sits directly below the map. Coordinate figures retain their CID,
class, and probability title plus scalar X/O distance circles. Region figures
use a concise title and compact class markers only for representatives with
matching physical VDF panels. Vector topology values remain in CSV rows and
are not rendered as arrows or endpoints.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)

from matplotlib import pyplot as plt
import numpy as np

from src.plotting.dataset_selection_colormap import (
    render_dataset_velocity_background,
)
from src.plotting.prediction_plot_geometry import (
    build_class_color_map,
    select_prediction_representatives,
)
from src.plotting.prediction_spatial_overlays import (
    _add_class_legend,
    _plot_prediction_distance_circles,
    _plot_prediction_cells,
    _plot_requested_coordinate,
)
from src.plotting.prediction_vdf_cuts import (
    extract_peak_vdf_cuts,
    plot_peak_vdf_planes,
)


def save_coordinate_prediction_figure(
    output_path,
    *,
    reader,
    prepared_source,
    row,
    requested_coordinate_re,
    class_ids,
    class_names,
    plotting_config,
    spatial_renderer=None,
    cut_loader=None,
):
    """Save one coordinate map and its selected cell's physical VDF panels.

    Coordinate orchestration calls this after its one-row CSV is written.
    The shared bulk-x-velocity map keeps the requested Earth-radius position
    as a small blue star separate from the predicted-class marker at the
    selected VDF cell. Scalar X/O distances are unfilled circles centred on
    that cell. Three lower panels show physical planes through the VDF peak
    with the historical source threshold and logarithmic colors.

    Parameters
    ----------
    output_path : str or pathlib.Path
        Combined PNG path written by Matplotlib.
    reader : object
        Open VLSV reader used for the spatial background.
    prepared_source : PreparedPredictionSource
        File-scoped producers, threshold name, and raw VDF extractor.
    row : mapping
        One saved prediction row with coordinates in Earth radii.
    requested_coordinate_re : array-like
        Requested ``[x, y, z]`` coordinate in Earth radii.
    class_ids : sequence of int
        Physical class IDs in checkpoint output order.
    class_names : sequence of str
        Physical class names in checkpoint output order.
    plotting_config : PredictionPlotConfig
        Fixed velocity map, spatial bounds, and VDF-panel settings.
    spatial_renderer : callable or None, optional
        Test substitute for the shared bulk-x-velocity renderer.
    cut_loader : callable or None, optional
        Test substitute returning peak-centred raw VDF cuts for one CID.

    Returns
    -------
    dict
        Background details and selected-cell diagnostics for the saved PNG.
    """

    requested_coordinate_re = np.asarray(
        requested_coordinate_re,
        dtype=np.float64,
    )
    return _save_combined_prediction_figure(
        output_path,
        reader=reader,
        prepared_source=prepared_source,
        representative_rows=[row],
        requested_coordinate_re=requested_coordinate_re,
        class_ids=class_ids,
        class_names=class_names,
        plotting_config=plotting_config,
        spatial_renderer=spatial_renderer,
        cut_loader=cut_loader,
        figure_kind="coordinate",
    )


def save_region_prediction_figure(
    output_path,
    *,
    reader,
    prepared_source,
    rows,
    class_ids,
    class_names,
    plotting_config,
    spatial_renderer=None,
    cut_loader=None,
):
    """Save one region map and a bounded set of physical VDF panels.

    Region orchestration passes rows already written to the timestep CSV.
    This function chooses deterministic representatives, marks only those
    cells with their predicted-class colors and marker shapes, and displays
    the matching peak-centred source VDF panels without rerunning CNN
    inference. The legend below the map is the class-name key, so no text or
    distance circles obscure the spatial background.

    Parameters
    ----------
    output_path : str or pathlib.Path
        Combined PNG path written by Matplotlib.
    reader : object
        Open VLSV reader used for the spatial background.
    prepared_source : PreparedPredictionSource
        File-scoped producers, threshold name, and raw VDF extractor.
    rows : sequence of mapping
        Saved region prediction rows in stable selected-cell order.
    class_ids : sequence of int
        Physical class IDs in checkpoint output order.
    class_names : sequence of str
        Physical class names in checkpoint output order.
    plotting_config : PredictionPlotConfig
        Fixed velocity map, representative selection, and VDF settings.
    spatial_renderer : callable or None, optional
        Test substitute for the shared bulk-x-velocity renderer.
    cut_loader : callable or None, optional
        Test substitute returning peak-centred raw VDF cuts for one CID.

    Returns
    -------
    dict
        Background details and representative diagnostics for the saved PNG.

    Notes
    -----
    Undisplayed region rows remain in CSV output but do not receive map
    markers. No requested-coordinate star or distance circle exists in a
    region figure.
    """

    representatives = select_prediction_representatives(
        list(rows),
        class_ids,
        strategy=plotting_config.representative_selection,
        maximum_examples=plotting_config.maximum_examples,
        explicit_cids=plotting_config.explicit_cids,
    )
    return _save_combined_prediction_figure(
        output_path,
        reader=reader,
        prepared_source=prepared_source,
        representative_rows=representatives,
        requested_coordinate_re=None,
        class_ids=class_ids,
        class_names=class_names,
        plotting_config=plotting_config,
        spatial_renderer=spatial_renderer,
        cut_loader=cut_loader,
        figure_kind="region",
    )


def _save_combined_prediction_figure(
    output_path,
    *,
    reader,
    prepared_source,
    representative_rows,
    requested_coordinate_re,
    class_ids,
    class_names,
    plotting_config,
    spatial_renderer,
    cut_loader,
    figure_kind,
):
    """Compose one velocity map and its peak-centred physical VDF panels.

    The spatial panel uses the x component of bulk velocity in metres per
    second so predictions appear against plasma flow rather than density.
    Coordinate figures mark the requested Earth-radius position with a compact
    blue star and retain a title reporting CID, predicted physical class, and
    probability. Region figures show compact class markers without nearby text
    and use a concise region title. The complete legend occupies a dedicated
    frameless row immediately below the colormap.

    Coordinate-only positive scalar X/O distances remain unfilled circles
    centred on the selected VDF cell. Region figures omit those circles.
    Displacement arrows, predicted endpoints, and connectors remain absent,
    while all topology numbers stay in serialized prediction rows. VDF panels
    retain the historical threshold, ``LogNorm``, unmodified
    ``nipy_spectral``, transparent masks, white axes, grey grid, physical km/s
    axes, and colorbar without blue panel boundaries.

    Parameters
    ----------
    output_path : str or pathlib.Path
        Combined PNG destination.
    reader : object
        Already-open VLSV reader reused by background rendering.
    prepared_source : PreparedPredictionSource
        Reusable source extractor and threshold producer selected once.
    representative_rows : sequence of mapping
        Rows whose map markers and physical VDF panels are displayed.
    requested_coordinate_re : array-like or None
        Coordinate-only ``[x, y, z]`` position in Earth radii.
    class_ids : sequence of int
        Physical class IDs in checkpoint output order.
    class_names : sequence of str
        Physical class names aligned with ``class_ids``.
    plotting_config : PredictionPlotConfig
        Spatial, representative, and VDF rendering settings.
    spatial_renderer : callable or None
        Optional test substitute for bulk-x-velocity rendering.
    cut_loader : callable or None
        Optional test substitute for CID-to-peak-cut extraction.
    figure_kind : {"coordinate", "region"}
        Controls coordinate-only circles, requested star, and spatial title.

    Returns
    -------
    dict
        Background renderer details, representative identities, and VDF peak
        indices corresponding to the saved image.
    """

    color_map = build_class_color_map(class_ids, class_names)
    if spatial_renderer is None:
        import analysator as pt

        _ = pt.plot.plot_colormap
        spatial_renderer = render_dataset_velocity_background
    if cut_loader is None:

        def cut_loader(cid):
            """Load cuts with the source-scoped physical threshold."""

            return extract_peak_vdf_cuts(
                prepared_source.extractor,
                cid,
                sparsity_threshold_variable=(
                    prepared_source.sparsity_threshold_variable
                ),
            )

    cut_y_re = _resolve_cut_y_re(
        representative_rows,
        plotting_config.spatial_slice_y_re,
    )
    row_count = 1 + len(representative_rows)
    figure = plt.figure(
        figsize=(15.0, 5.0 + 3.6 * len(representative_rows)),
        constrained_layout=True,
    )
    grid = figure.add_gridspec(
        row_count,
        3,
        height_ratios=[1.5, *([1.0] * len(representative_rows))],
    )
    spatial_grid = grid[0, :].subgridspec(
        2,
        1,
        height_ratios=[1.0, 0.12],
        hspace=0.02,
    )
    spatial_axis = figure.add_subplot(spatial_grid[0])
    legend_axis = figure.add_subplot(spatial_grid[1])
    legend_axis.axis("off")
    try:
        background = spatial_renderer(
            spatial_axis,
            reader=reader,
            resolved_source=prepared_source.resolved_source,
            plotting_config=plotting_config,
            cut_y_re=cut_y_re,
        )
        xmin, xmax, zmin, zmax = plotting_config.spatial_box_re
        spatial_axis.set_xlim(xmin, xmax)
        spatial_axis.set_ylim(zmin, zmax)
        distance_circle_artists = []
        if figure_kind == "coordinate":
            for row in representative_rows:
                distance_circle_artists.extend(
                    _plot_prediction_distance_circles(spatial_axis, row)
                )
        _plot_prediction_cells(
            spatial_axis,
            representative_rows,
            color_map,
        )
        if requested_coordinate_re is not None:
            _plot_requested_coordinate(
                spatial_axis,
                requested_coordinate_re,
            )
        _add_class_legend(
            legend_axis,
            representative_rows,
            color_map,
            include_requested=requested_coordinate_re is not None,
            distance_circle_artists=distance_circle_artists,
        )
        if figure_kind == "coordinate":
            spatial_axis.set_title(_prediction_title(representative_rows))
        else:
            spatial_axis.set_title("Region predictions")

        diagnostics = []
        for row_index, row in enumerate(representative_rows, start=1):
            cuts = cut_loader(int(row["cid"]))
            axes = [
                figure.add_subplot(grid[row_index, column])
                for column in range(3)
            ]
            _plot_peak_cuts(
                figure,
                axes,
                cuts,
                row,
                color_map[int(row["predicted_class_id"])]["color"],
                plotting_config.velocity_limit_ms,
            )
            diagnostics.append(
                {
                    "cid": int(row["cid"]),
                    "predicted_class_id": int(
                        row["predicted_class_id"]
                    ),
                    "predicted_class_name": str(
                        row["predicted_class_name"]
                    ),
                    "predicted_probability": float(
                        row["predicted_probability"]
                    ),
                    "peak_index": list(cuts.peak_index),
                }
            )
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(
            output_path,
            dpi=plotting_config.dpi,
            bbox_inches="tight",
        )
    finally:
        plt.close(figure)
    return {
        "figure_kind": figure_kind,
        "representative_cids": [
            int(row["cid"]) for row in representative_rows
        ],
        "representative_count": len(representative_rows),
        "background": background,
        "representatives": diagnostics,
    }


def _plot_peak_cuts(
    figure,
    axes,
    cuts,
    row,
    class_color,
    velocity_limit_ms,
):
    """Plot one representative's physical VDF cuts and class heading.

    Parameters
    ----------
    figure : matplotlib.figure.Figure
        Combined figure owning the shared physical VDF colorbar.
    axes : sequence of matplotlib.axes.Axes
        Axes ordered as ``vx-vy``, ``vx-vz``, and ``vy-vz``.
    cuts : PeakVdfCuts
        Physical source planes crossing one VDF peak.
    row : mapping
        Saved CID, physical class, and predicted probability.
    class_color : matplotlib color
        Current physical-class color used only for heading text.
    velocity_limit_ms : float or None
        Symmetric visible velocity limit in metres per second.
    """

    plot_peak_vdf_planes(
        figure,
        axes,
        cuts,
        velocity_limit_ms=velocity_limit_ms,
    )
    axes[0].text(
        0.0,
        1.18,
        _prediction_title([row]),
        transform=axes[0].transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        color=class_color,
    )


def _prediction_title(rows):
    """Format displayed CID, physical class, and probability title lines.

    Parameters
    ----------
    rows : sequence of mapping
        Displayed prediction rows in panel order.

    Returns
    -------
    str
        One newline-separated ``cid | class | p`` line per displayed row.
    """

    return "\n".join(
        f"cid={int(row['cid'])} | "
        f"class={row['predicted_class_name']} | "
        f"p={float(row['predicted_probability']):.3f}"
        for row in rows
    )


def _resolve_cut_y_re(rows, configured):
    """Resolve the three-dimensional velocity-map y cut in Earth radii.

    Parameters
    ----------
    rows : sequence of mapping
        Displayed rows containing ``selected_y_re`` in Earth radii.
    configured : float or None
        Optional explicit source-slice y coordinate in Earth radii.

    Returns
    -------
    float
        Y coordinate passed to the three-dimensional background renderer.
    """

    if configured is not None:
        return float(configured)
    values = np.asarray(
        [float(row["selected_y_re"]) for row in rows],
        dtype=np.float64,
    )
    if values.size == 0:
        return 0.0
    return float(np.median(values))
