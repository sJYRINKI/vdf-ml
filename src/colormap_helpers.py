import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle
import numpy as np

from src.point_topology import find_smallest_closed_contour, read_smoothed_flux_grid
from src.timesteps import create_timestep_path
from src.vdf_helpers import R_EARTH, get_vdf_cellid_set


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
    "lobe": {"color": "blue", "marker": "2", "label": "lobe", "s": 16},
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


def scatter_all_vdf_cells(ax, reader, boxre=None):
    """
    Scatter all VDF-containing spatial cells on an xz colormap.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes where points are drawn.
    reader : analysator.vlsvfile.VlsvReader
        Reader for the timestep VLSV file.
    boxre : list of float, optional
        Plot box in Earth radii: ``[xmin, xmax, zmin, zmax]``. If provided,
        only VDF cells inside the visible xz range are drawn.
    """

    cellids = sorted(get_vdf_cellid_set(reader))
    if not cellids:
        return

    try:
        coords_re = np.asarray(
            reader.get_cell_coordinates(cellids),
            dtype=float,
        ) / R_EARTH
    except Exception:
        coords_re = np.asarray(
            [reader.get_cell_coordinates(int(cid)) for cid in cellids],
            dtype=float,
        ) / R_EARTH

    if boxre is not None:
        x_min, x_max, z_min, z_max = [float(value) for value in boxre]
        visible = (
            (coords_re[:, 0] >= x_min)
            & (coords_re[:, 0] <= x_max)
            & (coords_re[:, 2] >= z_min)
            & (coords_re[:, 2] <= z_max)
        )
        coords_re = coords_re[visible]

    if len(coords_re) == 0:
        return

    ax.scatter(
        coords_re[:, 0],
        coords_re[:, 2],
        label="VDF cell",
        marker=".",
        s=8,
        color="gold",
        linewidths=0,
        zorder=3,
    )


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


def draw_manual_point_search_boxes(ax, metadata_rows):
    """
    Draw manual fixed search boxes from point-selection metadata.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes where boxes are drawn.
    metadata_rows : pandas.DataFrame
        Metadata rows for one timestep.
    """

    required_columns = {
        "selection_method",
        "point_kind",
        "source_point_x_re",
        "source_point_z_re",
        "selection_box_x_half_width_re",
        "selection_box_z_half_width_re",
    }
    if not required_columns.issubset(metadata_rows.columns):
        return

    box_rows = metadata_rows[
        metadata_rows["selection_method"] == "manual"
    ].drop_duplicates(
        [
            "point_kind",
            "source_point_x_re",
            "source_point_z_re",
            "selection_box_x_half_width_re",
            "selection_box_z_half_width_re",
        ]
    )
    if box_rows.empty:
        return

    plotted_labels = set()
    colors = {"x": "tab:blue", "o": "tab:blue"}

    for _, row in box_rows.iterrows():
        values = np.asarray(
            [
                row["source_point_x_re"],
                row["source_point_z_re"],
                row["selection_box_x_half_width_re"],
                row["selection_box_z_half_width_re"],
            ],
            dtype=float,
        )
        if np.any(np.isnan(values)):
            continue

        point_kind = row["point_kind"]
        label = None
        if point_kind not in plotted_labels:
            label = f"manual {point_kind.upper()} search box"
            plotted_labels.add(point_kind)

        center_x_re, center_z_re, half_width_x_re, half_width_z_re = values
        color = colors.get(point_kind, "tab:red")
        rectangle = Rectangle(
            (center_x_re - half_width_x_re, center_z_re - half_width_z_re),
            2.0 * half_width_x_re,
            2.0 * half_width_z_re,
            facecolor=color,
            edgecolor=color,
            alpha=0.18,
            linewidth=1.5,
            label=label,
            zorder=2.2,
        )
        ax.add_patch(rectangle)


def draw_x_point_search_areas(ax, metadata_rows, x_selection_config=None):
    """
    Draw Hessian-aligned X-point search boxes used for VDF-cell selection.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes where X-point search areas are drawn.
    metadata_rows : pandas.DataFrame
        Metadata rows for one timestep.
    x_selection_config : dict, optional
        X-point selection config containing ``half_width_di``.
    """

    required_columns = {
        "point_kind",
        "source_point_x_re",
        "source_point_z_re",
        "di_re",
        "hessian_e0_x",
        "hessian_e0_z",
        "hessian_e1_x",
        "hessian_e1_z",
    }
    if not x_selection_config or not required_columns.issubset(metadata_rows.columns):
        return

    half_width_di = x_selection_config.get("half_width_di", {})
    if not half_width_di:
        return

    half_width_0_di = float(half_width_di["eigenvector_0"])
    half_width_1_di = float(half_width_di["eigenvector_1"])

    x_point_rows = metadata_rows[metadata_rows["point_kind"] == "x"]
    if "selection_method" in x_point_rows.columns:
        x_point_rows = x_point_rows[
            x_point_rows["selection_method"].fillna("physical") == "physical"
        ]
    if x_point_rows.empty:
        return

    box_rows = x_point_rows.drop_duplicates(
        [
            "source_point_x_re",
            "source_point_z_re",
            "di_re",
            "hessian_e0_x",
            "hessian_e0_z",
            "hessian_e1_x",
            "hessian_e1_z",
        ]
    )

    for box_index, (_, row) in enumerate(box_rows.iterrows()):
        values = np.asarray(
            [
                row["source_point_x_re"],
                row["source_point_z_re"],
                row["di_re"],
                row["hessian_e0_x"],
                row["hessian_e0_z"],
                row["hessian_e1_x"],
                row["hessian_e1_z"],
            ],
            dtype=float,
        )
        if np.any(np.isnan(values)):
            continue

        center = values[:2]
        di_re = values[2]
        e0 = values[3:5]
        e1 = values[5:7]
        half_width_0_re = half_width_0_di * di_re
        half_width_1_re = half_width_1_di * di_re

        vertices = np.asarray(
            [
                center + half_width_0_re * e0 + half_width_1_re * e1,
                center - half_width_0_re * e0 + half_width_1_re * e1,
                center - half_width_0_re * e0 - half_width_1_re * e1,
                center + half_width_0_re * e0 - half_width_1_re * e1,
            ],
            dtype=float,
        )

        label = "X search area" if box_index == 0 else None
        polygon = Polygon(
            vertices,
            closed=True,
            facecolor="tab:blue",
            edgecolor="tab:blue",
            alpha=0.18,
            linewidth=1.5,
            label=label,
            zorder=2.1,
        )
        ax.add_patch(polygon)


def draw_o_point_search_areas(
    ax,
    reader,
    metadata_rows,
    flux_file_template,
    o_core_fraction=None,
):
    """
    Draw closed O-point flux contours used for VDF-cell selection.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes where island areas are drawn.
    reader : analysator.vlsvfile.VlsvReader
        Reader for the timestep VLSV file.
    metadata_rows : pandas.DataFrame
        Metadata rows for one timestep.
    flux_file_template : str
        Flux file template from dataset creation config.
    o_core_fraction : float, optional
        Fraction from O-point flux to boundary flux used when metadata does not
        contain ``search_flux``.
    """

    required_columns = {
        "point_kind",
        "source_point_x_re",
        "source_point_z_re",
        "source_point_flux",
        "boundary_flux",
        "timestep",
    }
    if not flux_file_template or not required_columns.issubset(metadata_rows.columns):
        return

    o_point_rows = metadata_rows[metadata_rows["point_kind"] == "o"]
    if "selection_method" in o_point_rows.columns:
        o_point_rows = o_point_rows[
            o_point_rows["selection_method"].fillna("physical") == "physical"
        ]
    if o_point_rows.empty:
        return

    contour_key_columns = ["source_point_x_re", "source_point_z_re", "boundary_flux"]
    if "search_flux" in o_point_rows.columns:
        contour_key_columns.append("search_flux")
    contour_rows = o_point_rows.drop_duplicates(contour_key_columns)

    timestep = int(contour_rows.iloc[0]["timestep"])
    flux_file_location = create_timestep_path(
        path_template=flux_file_template,
        timestep=timestep,
    )
    x_array_m, z_array_m, flux_function_zx = read_smoothed_flux_grid(
        reader=reader,
        flux_file_location=flux_file_location,
    )
    x_array_re = x_array_m / R_EARTH
    z_array_re = z_array_m / R_EARTH

    for contour_index, (_, row) in enumerate(contour_rows.iterrows()):
        search_flux = get_o_point_search_flux(
            row=row,
            o_core_fraction=o_core_fraction,
        )
        contour = find_smallest_closed_contour(
            x_array=x_array_re,
            z_array=z_array_re,
            flux_function_zx=flux_function_zx,
            contour_flux=search_flux,
            point_xz=(
                float(row["source_point_x_re"]),
                float(row["source_point_z_re"]),
            ),
        )
        if contour is None:
            continue

        _, vertices_re = contour

        label = "O search area" if contour_index == 0 else None
        polygon = Polygon(
            vertices_re,
            closed=True,
            facecolor="tab:blue",
            edgecolor="tab:blue",
            alpha=0.18,
            linewidth=1.5,
            label=label,
            zorder=2,
        )
        ax.add_patch(polygon)


def get_o_point_search_flux(row, o_core_fraction=None):
    """
    Return O-point contour flux from metadata or configured core fraction.

    Parameters
    ----------
    row : pandas.Series
        O-point metadata row.
    o_core_fraction : float, optional
        Fraction from O-point flux to boundary flux.

    Returns
    -------
    float
        Flux level used for drawing the O-point search area.
    """

    if "search_flux" in row:
        try:
            search_flux = float(row["search_flux"])
        except (TypeError, ValueError):
            search_flux = float("nan")

        if not np.isnan(search_flux):
            return search_flux

    if o_core_fraction is None:
        o_core_fraction = 1.0

    o_core_fraction = min(1.0, max(0.0, float(o_core_fraction)))
    source_flux = float(row["source_point_flux"])
    boundary_flux = float(row["boundary_flux"])

    return source_flux + o_core_fraction * (boundary_flux - source_flux)


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
        label = "Used VDF cell" if cell_index == 0 else None
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
            zorder=5,
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
            zorder=6,
        )

        plotted_source_classes.add(class_name)
