from matplotlib.patches import Rectangle
import numpy as np

from src.vdf_helpers import R_EARTH


SOURCE_POINT_STYLES = {
    "reconnection": {
        "color": "blue",
        "marker": "x",
        "label": "reconnection",
        "s": 10,
    },
    "o_point": {
        "edgecolor": "blue",
        "facecolor": "none",
        "marker": "o",
        "label": "o point",
        "s": 14,
        "linewidths": 1.0,
    },
    "other": {
        "color": "red",
        "marker": "s",
        "label": "other",
        "s": 10,
    },
    "lobe": {"color": "gold", "marker": "2", "label": "lobe", "s": 16},
}


def expr_velocity(exprmaps, requestvariables=False):
    """
    Return bulk velocity for Analysator colormap plotting.

    Parameters
    ----------
    exprmaps : dict
        Variables requested from the VLSV file.
    requestvariables : bool, optional
        If true, return the required variable names.

    Returns
    -------
    list[str] or numpy.ndarray
        Required variable names, or the bulk velocity vector field.
    """

    if requestvariables is True:
        return ["rho", "rho_v"]

    rho = exprmaps["rho"][:, :]
    rhov = exprmaps["rho_v"][:, :, :]

    return rhov / rho[:, :, None]


def draw_point_boxes(ax, metadata_rows, box_config, box_classes):
    """
    Draw configured xz boxes around source X/O point coordinates.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes where boxes are drawn.
    metadata_rows : pandas.DataFrame
        Metadata rows for one timestep.
    box_config : dict
        Box half-widths in Earth radii.
    box_classes : iterable of str
        Classes whose source coordinates should get boxes.
    """

    if not box_config or not box_classes:
        return

    half_width_x = float(box_config["x_half_width_re"])
    half_width_z = float(box_config["z_half_width_re"])
    box_classes = set(box_classes)
    box_rows = metadata_rows[
        metadata_rows["class_name"].isin(box_classes)
    ].drop_duplicates(["class_name", "x_re", "z_re"])

    for box_index, (_, row) in enumerate(box_rows.iterrows()):
        label = "box" if box_index == 0 else None

        rectangle = Rectangle(
            (row["x_re"] - half_width_x, row["z_re"] - half_width_z),
            2 * half_width_x,
            2 * half_width_z,
            fill=False,
            edgecolor="red",
            linewidth=1.0,
            label=label,
        )
        ax.add_patch(rectangle)


def scatter_label_points(ax, reader, metadata_rows):
    """
    Scatter source label points and sampled VDF cells on an xz colormap.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes where points are drawn.
    reader : analysator.vlsvfile.VlsvReader
        Reader for the timestep VLSV file.
    metadata_rows : pandas.DataFrame
        Metadata rows for one timestep.
    """

    marker_rows = metadata_rows.drop_duplicates(["cid"])

    for cell_index, (_, row) in enumerate(marker_rows.iterrows()):
        label = "cell" if cell_index == 0 else None
        cell_coord_re = np.asarray(
            reader.get_cell_coordinates(int(row["cid"])),
            dtype=float,
        ) / R_EARTH

        ax.scatter(
            cell_coord_re[0],
            cell_coord_re[2],
            label=label,
            marker=".",
            s=18,
            color="red",
            linewidths=0,
        )

    plotted_source_classes = set()
    source_rows = metadata_rows.drop_duplicates(["class_name", "x_re", "z_re"])

    for _, row in source_rows.iterrows():
        class_name = row["class_name"]
        style = SOURCE_POINT_STYLES.get(
            class_name,
            {"color": "black", "marker": ".", "label": class_name},
        )
        label = style["label"] if class_name not in plotted_source_classes else None

        ax.scatter(
            row["x_re"],
            row["z_re"],
            label=label,
            s=style.get("s", 16),
            color=style.get("color"),
            edgecolors=style.get("edgecolor"),
            facecolors=style.get("facecolor"),
            linewidths=style.get("linewidths"),
            marker=style["marker"],
        )

        plotted_source_classes.add(class_name)
