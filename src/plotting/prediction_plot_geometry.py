"""Construct physical-class styles and bounded prediction representatives.

Prediction figures use checkpoint class order for colors and current project
marker shapes, then select only the rows whose physical source VDF panels are
displayed. Topology values remain serialized prediction data and are not
converted into figure geometry here.
"""

from matplotlib import pyplot as plt

from src.plotting.dataset_selection_colormap import (
    DATASET_CLASS_MARKERS,
)


CLASS_COLOR_INDEX_BY_NAME = {
    "lobe": 0,
    "current_layer": 1,
    "o_point": 2,
    "x_point": 3,
    "solar_wind": 4,
    "magnetosheath": 5,
    "inner_magnetosphere": 6,
}


def build_class_color_map(class_ids, class_names):
    """
    Create stable colors in the supplied physical-class order.

    Dataset, PCA, and prediction plots call this single palette owner so one
    project class keeps the same color across figures. The physical name
    selects its fixed tabular color, while the supplied order determines only
    display and model-output indices; names are never sorted alphabetically.

    Parameters
    ----------
    class_ids : sequence of int
        Project class IDs in the display order used by the caller.
    class_names : sequence of str
        Physical class names in the same order.

    Returns
    -------
    dict
        Mapping from project class ID to class name, RGBA color, marker, and
        supplied order index.
    """

    class_ids = tuple(int(value) for value in class_ids)
    class_names = tuple(str(value) for value in class_names)
    colors = plt.get_cmap("tab10").colors
    return {
        class_id: {
            "name": class_name,
            "color": tuple(
                colors[CLASS_COLOR_INDEX_BY_NAME[class_name]]
            ),
            "marker": DATASET_CLASS_MARKERS[class_name],
            "output_index": index,
        }
        for index, (class_id, class_name) in enumerate(
            zip(class_ids, class_names)
        )
    }


def select_prediction_representatives(
    rows,
    class_ids,
    *,
    strategy,
    maximum_examples,
    explicit_cids=(),
):
    """
    Select bounded deterministic representatives from existing rows.

    Region figures call this after inference and CSV creation. Selection
    chooses a small, reproducible subset for source-VDF panels without
    changing the complete spatial prediction set or rerunning the CNN.

    Parameters
    ----------
    rows : sequence of mapping
        Stable prediction rows already written or ready to save.
    class_ids : sequence of int
        Checkpoint class IDs in model-output order.
    strategy : str
        Supported representative-selection strategy.
    maximum_examples : int
        Maximum number of returned rows.
    explicit_cids : sequence of int, optional
        Requested CIDs for the explicit strategy.

    Returns
    -------
    list of mapping
        Original row objects in deterministic selection order.

    """

    rows = list(rows)
    class_ids = tuple(int(value) for value in class_ids)
    maximum_examples = int(maximum_examples)
    strategy = str(strategy).strip().lower()
    if not rows:
        return []

    if strategy == "highest_confidence_overall":
        selected = sorted(
            enumerate(rows),
            key=lambda item: (
                -_finite_confidence(item[1]),
                item[0],
            ),
        )
        return [row for _, row in selected[:maximum_examples]]

    if strategy == "explicit_cids":
        explicit_cids = tuple(int(value) for value in explicit_cids)
        by_cid = {int(row["cid"]): row for row in rows}
        return [by_cid[cid] for cid in explicit_cids[:maximum_examples]]

    selected = []
    for class_id in class_ids:
        candidates = [
            (row_index, row)
            for row_index, row in enumerate(rows)
            if int(row["predicted_class_id"]) == class_id
        ]
        if not candidates:
            continue
        if strategy == "first_per_class":
            selected.append(candidates[0][1])
        else:
            selected.append(
                min(
                    candidates,
                    key=lambda item: (
                        -_finite_confidence(item[1]),
                        item[0],
                    ),
                )[1]
            )
        if len(selected) >= maximum_examples:
            break
    return selected


def _finite_confidence(row):
    """Return one row's maximum checkpoint-ordered class probability.

    Parameters
    ----------
    row : mapping
        Saved or pending prediction row.

    Returns
    -------
    float
        Maximum probability used only for deterministic representative
        ranking.
    """

    return float(row["predicted_probability"])
