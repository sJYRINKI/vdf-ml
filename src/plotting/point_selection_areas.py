"""Draw visible sample-selection areas used by dataset colormap frames.

Stage 6 receives accepted X/O source-point records from sample planning and
adds their exact x-z selection geometry to the existing spatial colormap.
Physical areas reuse the Hessian polygon or saved island contour that selected
samples, while the X- and O-point manual areas use their configured
Earth-radius boxes. For union selection, the physical outline replaces the
manual outline when the two methods selected any common VDF cell; disjoint
contributing areas remain visible. This plotting preference never changes the
union-selected cells. Keeping these artists separate from background
rendering avoids duplicating either the scientific geometry or the current
VLSV colormap implementation.
"""

from matplotlib.patches import Polygon, Rectangle
import numpy as np

from src.physics.point_selection_geometry import (
    get_hessian_di_box_vertices_re,
    get_manual_config_re,
)
from src.physics.point_selection import get_point_selection_method


_PHYSICAL_AREA_STYLE = {
    "facecolor": "tab:blue",
    "edgecolor": "tab:blue",
    "alpha": 0.18,
    "linewidth": 1.5,
}
_X_MANUAL_AREA_STYLE = {
    "fill": False,
    "edgecolor": "tab:blue",
    "alpha": 0.9,
    "linewidth": 1.5,
    "linestyle": "--",
}
_O_MANUAL_AREA_STYLE = {
    "fill": False,
    "edgecolor": "navy",
    "alpha": 0.9,
    "linewidth": 1.5,
    "linestyle": "-.",
}


def get_visible_selection_areas(
    selection_method,
    manual_cellids,
    physical_cellids,
):
    """Choose which manual and physical search areas to display.

    The decision affects only colormap rendering after point-cell selection
    has produced its final result. For union selection, a manual outline is
    omitted when manual and physical geometry selected at least one common
    VDF cell, because the physical area is the preferred visual explanation
    of that overlap. Manual-only cells remain part of the union dataset and
    remain plotted even when their manual outline is hidden. Existing display
    behavior for manual, physical, and consensus selection is unchanged.

    Parameters
    ----------
    selection_method : str
        Point-selection method used to create the final VDF-cell set.
    manual_cellids : collection of int
        VDF cells inside the configured manual search area.
    physical_cellids : collection of int
        VDF cells inside the physical X or O search area.

    Returns
    -------
    draw_manual : bool
        Whether the manual search-area geometry should be rendered.
    draw_physical : bool
        Whether the physical search-area geometry should be rendered.

    Notes
    -----
    This display rule does not alter selected cell IDs, metadata, point
    acceptance, class assignment, or source order.
    """

    if selection_method != "union":
        return True, True
    manual_cellids = set(manual_cellids)
    physical_cellids = set(physical_cellids)
    shared_cellids = manual_cellids & physical_cellids
    return (
        bool(manual_cellids) and not bool(shared_cellids),
        bool(physical_cellids),
    )


def draw_point_selection_areas(
    axis,
    *,
    config,
    x_point_records,
    o_point_records,
):
    """Draw exact physical and manual X/O search areas.

    Dataset-frame rendering calls this after the VLSV background and cell
    markers are available. Each accepted source point contributes at most one
    physical artist and one configured manual x-z box. Under union selection,
    the manual box is hidden when its selected cells overlap the physical
    selection, while every union-selected cell remains unchanged. Duplicate
    records sharing one source cell ID are drawn once, which keeps the search
    geometry readable when one source point contributes several VDF cells.

    Parameters
    ----------
    axis : matplotlib.axes.Axes
        Spatial x-z axes whose coordinates are Earth radii.
    config : dict
        Extraction configuration containing the X physical half-widths and
        the X/O manual half-widths in Earth radii.
    x_point_records : iterable of dict
        Accepted X-point records containing source cell IDs, Hessian
        eigenvectors, ion inertial lengths in metres, and centres in Earth
        radii.
    o_point_records : iterable of dict
        Accepted O-point records containing source cell IDs, saved physical
        contour vertices, and centres in Earth radii.

    Returns
    -------
    tuple of matplotlib.artist.Artist
        One legend handle for each existing area type, ordered as X physical,
        X manual, O physical, then O manual. All source-point artists remain
        attached to ``axis``.

    Notes
    -----
    The function draws saved physical point geometry directly and uses the
    same manual box configuration as sample selection. It never recomputes
    critical points, flux contours, or selected VDF-cell membership.
    """

    x_point_records = _deduplicate_point_records(x_point_records)
    o_point_records = _deduplicate_point_records(o_point_records)
    x_manual_records, x_physical_records = _visible_point_records_by_area(
        config,
        "x",
        x_point_records,
    )
    o_manual_records, o_physical_records = _visible_point_records_by_area(
        config,
        "o",
        o_point_records,
    )
    artists = []

    for index, point_record in enumerate(x_physical_records):
        polygon = Polygon(
            get_hessian_di_box_vertices_re(config, point_record),
            closed=True,
            label="physical X search area" if index == 0 else "_nolegend_",
            zorder=2.1,
            **_PHYSICAL_AREA_STYLE,
        )
        axis.add_patch(polygon)
        if index == 0:
            artists.append(polygon)

    x_manual_config = (
        get_manual_config_re(config, "x")
        if x_manual_records
        else None
    )
    for index, point_record in enumerate(x_manual_records):
        center_re = np.asarray(point_record["coord_re"], dtype=float)
        x_half_width_re = x_manual_config["x_half_width_re"]
        z_half_width_re = x_manual_config["z_half_width_re"]
        rectangle = Rectangle(
            (
                center_re[0] - x_half_width_re,
                center_re[2] - z_half_width_re,
            ),
            2.0 * x_half_width_re,
            2.0 * z_half_width_re,
            label="manual X search area" if index == 0 else "_nolegend_",
            zorder=2.2,
            **_X_MANUAL_AREA_STYLE,
        )
        axis.add_patch(rectangle)
        if index == 0:
            artists.append(rectangle)

    physical_o_point_records = [
        point_record
        for point_record in o_physical_records
        if point_record.get("contour_vertices_re") is not None
    ]
    for index, point_record in enumerate(physical_o_point_records):
        polygon = Polygon(
            np.asarray(point_record["contour_vertices_re"], dtype=float),
            closed=True,
            label="physical O search area" if index == 0 else "_nolegend_",
            zorder=2.0,
            **_PHYSICAL_AREA_STYLE,
        )
        axis.add_patch(polygon)
        if index == 0:
            artists.append(polygon)

    o_manual_config = (
        get_manual_config_re(config, "o")
        if o_manual_records
        else None
    )
    for index, point_record in enumerate(o_manual_records):
        center_re = np.asarray(point_record["coord_re"], dtype=float)
        x_half_width_re = o_manual_config["x_half_width_re"]
        z_half_width_re = o_manual_config["z_half_width_re"]
        rectangle = Rectangle(
            (
                center_re[0] - x_half_width_re,
                center_re[2] - z_half_width_re,
            ),
            2.0 * x_half_width_re,
            2.0 * z_half_width_re,
            label="manual O search area" if index == 0 else "_nolegend_",
            zorder=2.2,
            **_O_MANUAL_AREA_STYLE,
        )
        axis.add_patch(rectangle)
        if index == 0:
            artists.append(rectangle)

    return tuple(artists)


def _visible_point_records_by_area(config, point_kind, point_records):
    """Split accepted records by the search-area outlines they display."""

    if not point_records:
        return [], []
    selection_method = get_point_selection_method(config, point_kind)
    manual_records = []
    physical_records = []
    for point_record in point_records:
        draw_manual, draw_physical = get_visible_selection_areas(
            selection_method,
            point_record["manual_cellids"],
            point_record["physical_cellids"],
        )
        if draw_manual:
            manual_records.append(point_record)
        if draw_physical:
            physical_records.append(point_record)
    return manual_records, physical_records


def _deduplicate_point_records(point_records):
    """Keep the first accepted record for each stable source-point identity."""

    records = []
    seen_identities = set()
    for point_record in point_records:
        identity = (
            str(point_record["point_kind"]),
            int(point_record["cellid"]),
            tuple(float(value) for value in point_record["coord_re"]),
        )
        if identity in seen_identities:
            continue
        seen_identities.add(identity)
        records.append(point_record)
    return records
