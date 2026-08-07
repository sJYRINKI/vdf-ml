"""Own the shared dataset spatial-selection colormap design.

This focused plotting module supplies the bulk-``v_x`` Analysator producer,
zero-centred red-white-blue background, black magnetic streamlines, and
yellow/blue/red candidate/source/selected marker layers used by extraction
Stage 6 and prediction maps. Exact X/O search geometry remains owned by
``point_selection_areas``.
"""

from matplotlib.lines import Line2D
import numpy as np

from src.physics.vlsv_physical_values import resolve_bulk_velocity_source
from src.plotting.prediction_spatial_background import (
    _resolve_streamline_variable,
    spatial_mesh_shape,
)


DATASET_SOURCE_COLOR = "tab:blue"
DATASET_SELECTED_COLOR = "red"
DATASET_CANDIDATE_MARKER_SIZE = 2
DATASET_SOURCE_MARKER_SIZE = 10
DATASET_SELECTED_MARKER_SIZE = 8
DATASET_LEGEND_MARKER_SIZE = 3
DATASET_CLASS_MARKERS = {
    "lobe": "^",
    "current_layer": "s",
    "o_point": "o",
    "x_point": "X",
    "solar_wind": ">",
    "magnetosheath": "D",
    "inner_magnetosphere": "p",
}
def render_dataset_velocity_background(
    axis,
    *,
    reader,
    resolved_source,
    plotting_config,
    cut_y_re,
):
    """Render the shared dataset bulk-velocity and streamline background.

    Newer population-based Vlasiator sources provide bulk velocity directly,
    while older sources store number flux as ``rho_v`` and number density as
    ``rho``. The renderer prefers the stored velocity and otherwise evaluates
    the established ``rho_v / rho`` expression. Analysator selects the x
    component in metres per second. Symmetric ``bwr`` limits keep zero white,
    and thin black magnetic streamlines retain the current spatial context.

    Parameters
    ----------
    axis : matplotlib.axes.Axes
        Existing x-z spatial axes in Earth radii.
    reader : analysator.vlsvfile.VlsvReader
        Already-open timestep source used for both velocity and magnetic data.
    resolved_source : ResolvedVlsvSource
        Active file-scoped source containing the VDF population name.
    plotting_config : object
        Shared spatial bounds, symmetric velocity limits, and streamline
        flag.
    cut_y_re : float
        Physical y coordinate in Earth radii for a three-dimensional source.

    Returns
    -------
    dict
        Renderer name, velocity-source mode, selected physical producers,
        and magnetic streamline variable.
    """

    import analysator as pt

    velocity_mode, primary_variable, density_variable = (
        resolve_bulk_velocity_source(
            reader,
            resolved_source.population,
        )
    )
    streamlines = (
        _resolve_streamline_variable(reader)
        if plotting_config.show_streamlines
        else None
    )
    options = {
        "vlsvobj": reader,
        "axes": axis,
        "operator": "x",
        "colormap": "bwr",
        "lin": 1,
        "title": "",
        "cbtitle": r"$v_x$ [m s$^{-1}$]",
        "streamlines": streamlines,
        "streamlinecolor": "black",
        "streamlinedensity": 1,
        "streamlinethick": 1.0,
    }
    if velocity_mode == "direct":
        options["var"] = primary_variable
        momentum_variable = None
    else:
        options["expression"] = create_bulk_velocity_expression(
            density_variable,
            primary_variable,
        )
        momentum_variable = primary_variable
    if plotting_config.spatial_box_re is not None:
        options["boxre"] = list(plotting_config.spatial_box_re)
    if plotting_config.color_limits is not None:
        options["vmin"], options["vmax"] = plotting_config.color_limits
    mesh_shape = spatial_mesh_shape(reader)
    if mesh_shape is not None and mesh_shape[1] > 1:
        pt.plot.plot_colormap3dslice(
            normal="y",
            cutpointre=float(cut_y_re),
            **options,
        )
        renderer = "plot_colormap3dslice"
    else:
        pt.plot.plot_colormap(**options)
        renderer = "plot_colormap"
    return {
        "renderer": renderer,
        "velocity_mode": velocity_mode,
        "velocity_variable": (
            primary_variable if velocity_mode == "direct" else None
        ),
        "density_variable": density_variable,
        "momentum_variable": momentum_variable,
        "streamline_variable": streamlines,
    }


def create_bulk_velocity_expression(density_variable, momentum_variable):
    """Create an Analysator expression for physical bulk velocity.

    Parameters
    ----------
    density_variable : str
        Active number-density producer in ``m^-3``.
    momentum_variable : str
        Matching number-flux producer whose ratio to density is velocity.

    Returns
    -------
    callable
        Analysator expression returning ``[vx, vy, vz]`` in metres per
        second, or the two required variable names when queried.
    """

    def bulk_velocity(expression_maps, requestvariables=False):
        if requestvariables:
            return [density_variable, momentum_variable]
        density = np.asarray(expression_maps[density_variable])
        momentum = np.asarray(expression_maps[momentum_variable])
        return momentum / density[..., np.newaxis]

    return bulk_velocity


def plot_dataset_selected_cells(axis, rows):
    """Draw selected VDF cells in red with current class marker shapes.

    Red identifies the stored VDF cell selected for a dataset sample. Marker
    shape identifies its current class and matches the blue source-coordinate
    shape drawn by :func:`plot_dataset_source_coordinates`. The
    ``current_layer`` class is red-only because it has no configured source
    coordinate. Empty and conflict-ejected points have no saved rows and are
    therefore absent.

    Parameters
    ----------
    axis : matplotlib.axes.Axes
        Spatial x-z axes in Earth radii.
    rows : pandas.DataFrame
        Saved rows for one timestep; duplicate spatial cell IDs are drawn
        only once.

    Returns
    -------
    tuple of matplotlib.lines.Line2D
        Red class-marker handles in first-appearance order.
    """

    handles = []
    selected_rows = rows.drop_duplicates("cid", keep="first")
    for class_name in selected_rows["class_name"].astype(str).unique():
        class_rows = selected_rows[
            selected_rows["class_name"].astype(str) == class_name
        ]
        marker = DATASET_CLASS_MARKERS.get(class_name, "o")
        axis.scatter(
            class_rows["vdf_x_re"].to_numpy(dtype=float),
            class_rows["vdf_z_re"].to_numpy(dtype=float),
            marker=marker,
            s=DATASET_SELECTED_MARKER_SIZE,
            color=DATASET_SELECTED_COLOR,
            edgecolors="black",
            linewidths=0.35,
            zorder=5,
        )
        handles.append(
            Line2D(
                [],
                [],
                linestyle="none",
                marker=marker,
                markersize=DATASET_LEGEND_MARKER_SIZE,
                markerfacecolor=DATASET_SELECTED_COLOR,
                markeredgecolor="black",
                label=f"{class_name} selected VDF",
            )
        )
    return tuple(handles)


def plot_dataset_source_coordinates(
    axis,
    *,
    configured_sources,
    x_point_records,
    o_point_records,
):
    """Draw configured and detected sample-source coordinates in blue.

    Blue identifies the physical or configured source coordinate, while the
    paired selected VDF cell is red. Both roles use the identical marker from
    ``DATASET_CLASS_MARKERS`` so shape carries class or point identity and
    color carries plotting role. ``configured_sources`` has already been
    matched to saved class/CID pairs, and only accepted X/O records reach this
    function, so empty and conflict-ejected sources are absent.

    Parameters
    ----------
    axis : matplotlib.axes.Axes
        Spatial x-z axes in Earth radii.
    configured_sources : mapping
        Manual class names mapped to accepted ``[x_re, y_re, z_re]`` source
        coordinates in Earth radii.
    x_point_records, o_point_records : sequence of dict
        Accepted point records with source ``coord_re`` and class identity.

    Returns
    -------
    tuple of matplotlib.lines.Line2D
        Blue source-marker handles in configured then X/O order.
    """

    handles = []
    for class_name, coordinates_re in configured_sources.items():
        if not coordinates_re:
            continue
        coordinates_re = np.asarray(coordinates_re, dtype=float)
        marker = DATASET_CLASS_MARKERS.get(str(class_name), "o")
        axis.scatter(
            coordinates_re[:, 0],
            coordinates_re[:, 2],
            marker=marker,
            s=DATASET_SOURCE_MARKER_SIZE,
            color=DATASET_SOURCE_COLOR,
            edgecolors="black",
            linewidths=0.35,
            zorder=6,
        )
        handles.append(_source_handle(str(class_name), marker))
    for point_kind, records in (
        ("x", x_point_records),
        ("o", o_point_records),
    ):
        records = tuple(records)
        if not records:
            continue
        class_name = str(
            records[0].get("class_name", f"{point_kind}_point")
        )
        marker = DATASET_CLASS_MARKERS.get(class_name, "o")
        coordinates_re = np.asarray(
            [record["coord_re"] for record in records],
            dtype=float,
        )
        axis.scatter(
            coordinates_re[:, 0],
            coordinates_re[:, 2],
            marker=marker,
            s=DATASET_SOURCE_MARKER_SIZE,
            color=DATASET_SOURCE_COLOR,
            edgecolors="black",
            linewidths=0.35,
            zorder=6,
        )
        handles.append(_source_handle(class_name, marker))
    return tuple(handles)


def add_dataset_selection_legend(
    figure,
    *,
    include_all_cells,
    all_cell_style,
    source_handles,
    selected_handles,
    selection_handles,
):
    """Place the shared dataset role and geometry legend below the axes.

    The four-column historical layout keeps candidate, source, selected, and
    exact search-area identities outside the physical plot box. Labels are
    deduplicated in first-appearance order before the figure legend is built.

    Parameters
    ----------
    figure : matplotlib.figure.Figure
        Dataset colormap figure that owns the outside legend.
    include_all_cells : bool
        Whether the yellow VDF-cell layer appears in the frame.
    all_cell_style : mapping
        Marker and yellow role color used by the candidate-cell layer.
    source_handles : sequence
        Blue class/source marker handles returned by source plotting.
    selected_handles : sequence
        Red class marker handles returned by selected-cell plotting.
    selection_handles : sequence
        Exact physical/manual X/O area handles.
    """

    handles = []
    if include_all_cells:
        handles.append(
            Line2D(
                [],
                [],
                linestyle="none",
                marker=all_cell_style["marker"],
                markersize=DATASET_LEGEND_MARKER_SIZE,
                color=all_cell_style["color"],
                label="VDF cell",
            )
        )
    source_by_class = {
        handle.get_label().removesuffix(" coordinate"): handle
        for handle in source_handles
    }
    selected_by_class = {
        handle.get_label().removesuffix(" selected VDF"): handle
        for handle in selected_handles
    }
    class_names = tuple(
        dict.fromkeys((*source_by_class, *selected_by_class))
    )
    for class_name in class_names:
        if class_name in source_by_class:
            handles.append(source_by_class[class_name])
        if class_name in selected_by_class:
            handles.append(selected_by_class[class_name])
    handles.extend(selection_handles)
    unique_handles = {
        handle.get_label(): handle
        for handle in handles
    }
    figure.legend(
        handles=tuple(unique_handles.values()),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        ncol=4,
        fontsize=8,
        frameon=False,
    )


def _source_handle(class_name, marker):
    """Create one blue configured/detected source-coordinate legend handle."""

    return Line2D(
        [],
        [],
        linestyle="none",
        marker=marker,
        markersize=DATASET_LEGEND_MARKER_SIZE,
        markerfacecolor=DATASET_SOURCE_COLOR,
        markeredgecolor="black",
        label=f"{class_name} coordinate",
    )

