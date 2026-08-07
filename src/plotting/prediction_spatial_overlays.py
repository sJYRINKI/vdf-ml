"""Draw displayed physical-class predictions over one spatial background.

Coordinate figures draw the selected VDF cell, a compact requested-point star,
and scalar X/O distance circles. Region figures draw only compact physical-class
markers for the bounded representatives whose physical VDF panels appear in the
same figure. Class names remain in the legend rather than obscuring the map.
"""

from matplotlib.lines import Line2D
from matplotlib.patches import Circle

from src.plotting.dataset_selection_colormap import (
    DATASET_SOURCE_COLOR,
)


def _plot_prediction_cells(
    axis,
    rows,
    color_map,
):
    """Draw compact predicted-VDF markers on one spatial map.

    Coordinate prediction supplies its selected VDF row, while region
    prediction supplies only representatives that also own physical VDF
    panels. The small fixed scatter area preserves the x-velocity background
    and magnetic streamlines; the external legend uses independently sized
    handles so physical classes remain readable.

    Parameters
    ----------
    axis : matplotlib.axes.Axes
        Spatial x-z axes expressed in Earth radii.
    rows : sequence of mapping
        Coordinate row or bounded region representatives already selected for
        physical VDF-panel display.
    color_map : mapping
        Checkpoint class IDs mapped to current colors, markers, and names.

    Notes
    -----
    The renderer does not annotate individual region cells or alter prediction
    rows, class probabilities, topology values, marker shapes, or colors.
    """

    for row in rows:
        class_record = color_map[int(row["predicted_class_id"])]
        x_re = float(row["selected_x_re"])
        z_re = float(row["selected_z_re"])
        axis.scatter(
            [x_re],
            [z_re],
            marker=class_record["marker"],
            s=8,
            color=class_record["color"],
            edgecolors="black",
            linewidths=0.3,
            zorder=5,
        )


def _plot_requested_coordinate(axis, requested_coordinate_re):
    """Mark one user-requested coordinate with a blue star.

    Parameters
    ----------
    axis : matplotlib.axes.Axes
        Spatial x-z axes expressed in Earth radii.
    requested_coordinate_re : array-like
        Requested ``[x, y, z]`` position in Earth radii. Only x and z are
        projected onto the spatial panel.
    """

    axis.scatter(
        [float(requested_coordinate_re[0])],
        [float(requested_coordinate_re[2])],
        marker="*",
        s=12,
        color=DATASET_SOURCE_COLOR,
        edgecolors="black",
        linewidths=0.4,
        zorder=7,
    )


def _plot_prediction_distance_circles(axis, prediction_row):
    """Draw predicted X- and O-point distance circles around one VDF cell.

    The CNN predicts scalar distances from the selected VDF cell to the
    nearest physical X and O points. Each positive distance is shown as an
    unfilled circle centred on the VDF cell in the spatial x-z map. The
    circles communicate distance magnitude without asserting a direction,
    which remains ambiguous when vector arrows and predicted point endpoints
    are intentionally omitted.

    Parameters
    ----------
    axis : matplotlib.axes.Axes
        Spatial x-z axes whose coordinates are Earth radii.
    prediction_row : pandas.Series or mapping
        Common coordinate/region prediction row containing selected VDF
        coordinates and physical X/O distances in Earth radii.

    Returns
    -------
    list of matplotlib.patches.Circle
        Circle patches actually added to the spatial axes.

    Notes
    -----
    The numerical X/O vector components remain in the prediction CSV. This
    renderer uses only the scalar distance values and does not alter model
    output.
    """

    center = (
        float(prediction_row["selected_x_re"]),
        float(prediction_row["selected_z_re"]),
    )
    circle_specs = (
        (
            "predicted_distance_to_x_point_re",
            "red",
            "--",
            "Predicted distance to X point",
        ),
        (
            "predicted_distance_to_o_point_re",
            "blue",
            ":",
            "Predicted distance to O point",
        ),
    )
    artists = []
    for column, color, linestyle, label in circle_specs:
        radius = float(prediction_row[column])
        if radius > 0.0:
            circle = Circle(
                center,
                radius,
                fill=False,
                color=color,
                linestyle=linestyle,
                linewidth=1.2,
                alpha=0.9,
                label=label,
                zorder=4,
            )
            axis.add_patch(circle)
            artists.append(circle)
    return artists


def _add_class_legend(
    axis,
    rows,
    color_map,
    *,
    include_requested,
    distance_circle_artists,
):
    """Place displayed prediction identities in the map's legend row.

    Parameters
    ----------
    axis : matplotlib.axes.Axes
        Dedicated frameless axes directly below the spatial colormap.
    rows : sequence of mapping
        Rows whose predicted VDF markers are visible on the map.
    color_map : mapping
        Checkpoint class IDs mapped to current colors, markers, and names.
    include_requested : bool
        Whether the coordinate-only blue star is visible.
    distance_circle_artists : sequence of matplotlib.patches.Circle
        Coordinate-only X/O circles. Repeated labels are collapsed into one
        legend entry; region figures pass an empty sequence.

    Returns
    -------
    matplotlib.legend.Legend
        Compact legend containing only artists drawn on the map.
    """

    displayed_class_ids = tuple(
        dict.fromkeys(int(row["predicted_class_id"]) for row in rows)
    )
    handles = [
        Line2D(
            [],
            [],
            linestyle="none",
            marker=color_map[class_id]["marker"],
            markersize=7,
            markerfacecolor=color_map[class_id]["color"],
            markeredgecolor="black",
            label=color_map[class_id]["name"],
        )
        for class_id in displayed_class_ids
    ]
    if include_requested:
        handles.append(
            Line2D(
                [],
                [],
                linestyle="none",
                marker="*",
                markersize=10,
                markerfacecolor=DATASET_SOURCE_COLOR,
                markeredgecolor="black",
                label="requested coordinate",
            )
        )
    handles.extend(distance_circle_artists)
    handles = list(
        {
            handle.get_label(): handle
            for handle in handles
        }.values()
    )
    return axis.legend(
        handles=handles,
        loc="center",
        ncol=min(4, len(handles)),
        fontsize="small",
        frameon=False,
    )
