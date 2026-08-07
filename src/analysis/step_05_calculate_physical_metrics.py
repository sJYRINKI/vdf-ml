"""Stage 5: calculate physical-label diagnostics from PCA geometry.

KMeans and nearest-neighbor search operate only in the shared leading PCA
score space selected by Stage 0. Physical class names are introduced after
those label-blind geometric calculations to measure how well the
representation groups known plasma populations. Split and class values are
reporting groups rather than fitting inputs.

This stage returns in-memory sample-count, KMeans-purity,
physical-class-concentration, and nearest-neighbor tables. Stage 7 writes
them into one human-readable metrics report; no diagnostic labels or
sample-level purity arrays are persisted.
"""

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


CLUSTER_PURITY_COLUMNS = (
    "cluster",
    "count",
    "dominant_class",
    "dominant_count",
    "purity",
)
CLASS_CONCENTRATION_COLUMNS = (
    "class_name",
    "count",
    "best_cluster",
    "count_in_best_cluster",
    "fraction_in_best_cluster",
)
NEIGHBOR_PURITY_COLUMNS = (
    "split",
    "class_name",
    "count",
    "mean",
    "median",
    "min",
)
_STANDARD_SPLIT_ORDER = ("train", "validation", "test", "all")


def load_physical_classes(metadata):
    """Read physical class IDs and names in saved sample order.

    Stage 5 calls this only after PCA and KMeans fitting. The returned arrays
    remain aligned with score rows and are used for diagnostic tables and
    physical-class figure styles, never as estimator inputs.

    Parameters
    ----------
    metadata : pandas.DataFrame
        Dataset metadata containing ``class_id`` and ``class_name``.

    Returns
    -------
    class_ids : numpy.ndarray
        Int64 physical class IDs in metadata row order.
    class_names : numpy.ndarray
        Physical class names in the same row order.
    """

    class_ids = metadata["class_id"].to_numpy(dtype=np.int64)
    class_names = metadata["class_name"].astype(str).to_numpy()
    return class_ids, class_names


def create_sample_count_table(metadata):
    """Count samples by saved split and current physical class.

    The table uses the metadata split directly when present. Datasets without
    split information receive one in-memory ``all`` reporting group; their
    metadata file is not changed. Standard split names precede any additional
    names, and physical columns follow current class-ID order.

    Parameters
    ----------
    metadata : pandas.DataFrame
        Sample-aligned metadata containing ``class_id``, ``class_name``, and
        optionally ``split``.

    Returns
    -------
    pandas.DataFrame
        Cross-tabulation with an ``All`` margins row and column.
    """

    split_values = _get_split_values(metadata)
    split_order = _get_split_order(split_values)
    class_order = _get_physical_class_order(metadata)
    return pd.crosstab(
        pd.Categorical(
            split_values,
            categories=split_order,
            ordered=True,
        ),
        pd.Categorical(
            metadata["class_name"].astype(str),
            categories=class_order,
            ordered=True,
        ),
        margins=True,
        margins_name="All",
        dropna=False,
        rownames=["split"],
        colnames=["class_name"],
    )


def calculate_kmeans_physical_metrics(
    cluster_labels,
    class_ids,
    class_names,
):
    """Measure an internal KMeans partition against physical classes.

    KMeans has already fitted on PCA scores without labels. Cluster purity is
    the dominant physical-class fraction inside each partition, while
    weighted purity aggregates dominant counts over the complete dataset.
    Physical-class concentration asks the complementary question: which one
    KMeans partition contains the largest share of each physical class.

    Parameters
    ----------
    cluster_labels : array-like
        In-memory KMeans partition IDs in saved sample order.
    class_ids : array-like
        Integer physical class IDs aligned with ``cluster_labels``.
    class_names : array-like
        Physical class names aligned with ``class_ids``.

    Returns
    -------
    cluster_purity : pandas.DataFrame
        Count, dominant class, dominant count, and purity for each cluster.
    composition : pandas.DataFrame
        Per-cluster physical-class counts with dynamic class columns.
    class_concentration : pandas.DataFrame
        Best cluster and retained fraction for every physical class.
    weighted_purity : float
        Complete-sample dominant-class fraction across KMeans clusters.

    Notes
    -----
    Cluster IDs remain diagnostic partitions. They are not remapped to class
    IDs and are not saved or plotted as another classification.
    """

    cluster_labels = np.asarray(cluster_labels, dtype=np.int64)
    class_ids = np.asarray(class_ids, dtype=np.int64)
    class_names = np.asarray(class_names, dtype=object)
    class_rows = pd.DataFrame(
        {"class_id": class_ids, "class_name": class_names}
    ).drop_duplicates("class_id")
    class_rows = class_rows.sort_values("class_id", kind="stable")
    class_order = class_rows["class_name"].astype(str).tolist()
    class_name_by_id = dict(
        zip(
            class_rows["class_id"].astype(int),
            class_rows["class_name"].astype(str),
        )
    )
    cluster_ids = np.unique(cluster_labels)
    purity_rows = []
    composition_rows = []
    dominant_total = 0
    for cluster_id in cluster_ids:
        cluster_mask = cluster_labels == cluster_id
        cluster_size = int(np.count_nonzero(cluster_mask))
        cluster_class_ids, cluster_counts = np.unique(
            class_ids[cluster_mask],
            return_counts=True,
        )
        dominant_count = int(np.max(cluster_counts))
        dominant_class_id = int(
            np.min(cluster_class_ids[cluster_counts == dominant_count])
        )
        dominant_total += dominant_count
        purity_rows.append(
            {
                "cluster": int(cluster_id),
                "count": cluster_size,
                "dominant_class": class_name_by_id[dominant_class_id],
                "dominant_count": dominant_count,
                "purity": dominant_count / cluster_size,
            }
        )
        row = {"cluster": int(cluster_id)}
        for class_name in class_order:
            row[class_name] = int(
                np.count_nonzero(class_names[cluster_mask] == class_name)
            )
        composition_rows.append(row)

    concentration_rows = []
    for class_name in class_order:
        class_mask = class_names == class_name
        class_cluster_ids, class_cluster_counts = np.unique(
            cluster_labels[class_mask],
            return_counts=True,
        )
        best_index = int(np.argmax(class_cluster_counts))
        total_count = int(np.count_nonzero(class_mask))
        best_count = int(class_cluster_counts[best_index])
        concentration_rows.append(
            {
                "class_name": class_name,
                "count": total_count,
                "best_cluster": int(class_cluster_ids[best_index]),
                "count_in_best_cluster": best_count,
                "fraction_in_best_cluster": best_count / total_count,
            }
        )

    cluster_purity = pd.DataFrame(
        purity_rows,
        columns=CLUSTER_PURITY_COLUMNS,
    )
    composition = pd.DataFrame(
        composition_rows,
        columns=("cluster", *class_order),
    )
    class_concentration = pd.DataFrame(
        concentration_rows,
        columns=CLASS_CONCENTRATION_COLUMNS,
    )
    weighted_purity = dominant_total / len(cluster_labels)
    return (
        cluster_purity,
        composition,
        class_concentration,
        weighted_purity,
    )


def calculate_nearest_neighbor_purity(
    pca_scores,
    metadata,
    neighbor_count,
):
    """Calculate local physical-label agreement in PCA score space.

    The historical metric fits one nearest-neighbor model on training PCA
    scores and queries every split against that reference. Training queries
    remove their own exact row before retaining ``neighbor_count`` results;
    validation and test rows are not members of the training reference. When
    metadata has no split column, every row belongs to one in-memory ``all``
    split, the complete score set is the reference, and every query removes
    itself.

    Neighbor search receives only the shared leading PCA score columns.
    Physical class names are read after all neighbor indices are known, then
    used to calculate one matching-label fraction per sample and group those
    fractions for reporting.

    Parameters
    ----------
    pca_scores : numpy.ndarray
        Label-blind leading PCA scores with shape
        ``(n_samples, n_analysis_components)``.
    metadata : pandas.DataFrame
        Sample-aligned metadata containing ``class_name`` and an optional
        ``split`` column.
    neighbor_count : int
        Number of distance-ordered reference neighbors retained per sample.

    Returns
    -------
    sample_purity : numpy.ndarray
        Per-sample matching-label fractions in saved row order.
    summary : pandas.DataFrame
        Count, mean, median, and minimum purity grouped by split and physical
        class in the historical lexical split-and-class table order.

    Notes
    -----
    Split and class groups are reporting categories. Labels do not influence
    the fitted neighbor geometry or the returned neighbor indices.
    """

    pca_scores = np.asarray(pca_scores)
    split_values = _get_split_values(metadata)
    if "split" in metadata.columns:
        reference_indices = np.flatnonzero(split_values == "train")
    else:
        reference_indices = np.arange(len(pca_scores), dtype=np.int64)
    model = NearestNeighbors(n_neighbors=int(neighbor_count) + 1)
    model.fit(pca_scores[reference_indices])
    neighbor_positions = model.kneighbors(
        pca_scores,
        return_distance=False,
    )
    neighbor_indices = reference_indices[neighbor_positions]
    self_matches = (
        neighbor_indices
        == np.arange(len(pca_scores), dtype=np.int64)[:, np.newaxis]
    )
    self_last = np.argsort(self_matches, axis=1, kind="stable")
    neighbor_indices = np.take_along_axis(
        neighbor_indices,
        self_last,
        axis=1,
    )[:, : int(neighbor_count)]

    class_names = metadata["class_name"].astype(str).to_numpy()
    sample_purity = np.mean(
        class_names[neighbor_indices] == class_names[:, np.newaxis],
        axis=1,
    )
    sample_rows = pd.DataFrame(
        {
            "split": split_values,
            "class_name": class_names,
            "label_purity": sample_purity,
        }
    )
    summary = (
        sample_rows.groupby(
            ["split", "class_name"],
            sort=True,
        )["label_purity"]
        .agg(["count", "mean", "median", "min"])
        .reset_index()
    )
    return sample_purity, summary.loc[:, list(NEIGHBOR_PURITY_COLUMNS)]


def _get_split_values(metadata):
    """Return reporting splits without changing saved metadata.

    Parameters
    ----------
    metadata : pandas.DataFrame
        Sample-aligned metadata with an optional ``split`` column.

    Returns
    -------
    numpy.ndarray
        Saved split names, or one in-memory ``all`` value per sample when
        the dataset has no established split information.
    """

    if "split" in metadata.columns:
        return metadata["split"].astype(str).to_numpy()
    return np.full(len(metadata), "all", dtype=object)


def _get_split_order(split_values):
    """Order reporting splits consistently in every metrics table.

    Parameters
    ----------
    split_values : array-like
        Split name for every sample in saved row order.

    Returns
    -------
    tuple of str
        Present standard splits in train, validation, test, all order,
        followed by additional names in first-appearance order.
    """

    first_seen = tuple(dict.fromkeys(str(value) for value in split_values))
    return tuple(
        value for value in _STANDARD_SPLIT_ORDER if value in first_seen
    ) + tuple(
        value for value in first_seen if value not in _STANDARD_SPLIT_ORDER
    )


def _get_physical_class_order(metadata):
    """Return current physical classes in stable dataset class-ID order.

    Parameters
    ----------
    metadata : pandas.DataFrame
        Sample metadata containing physical ``class_id`` and ``class_name``.

    Returns
    -------
    tuple of str
        Names of classes present in the dataset, ordered by their current
        physical class IDs rather than alphabetically.
    """

    rows = metadata[["class_id", "class_name"]].drop_duplicates("class_id")
    rows = rows.sort_values("class_id", kind="stable")
    return tuple(rows["class_name"].astype(str))
