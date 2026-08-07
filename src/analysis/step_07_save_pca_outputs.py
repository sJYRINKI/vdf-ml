"""Stage 7: save two physical-class figures and one PCA metrics report.

This final stage owns every PCA filesystem and Matplotlib side effect. PCA
scores, KMeans partition IDs, t-SNE coordinates, and per-sample diagnostics
remain in memory; only physical-class PCA and t-SNE views plus one plain-text
metrics report are persisted.

The renderer reuses the current project class colors and marker symbols.
Color and shape therefore identify only physical classes, while the smaller
translucent rasterized points expose density without inventing separation in
the label-blind embeddings.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.plotting.dataset_selection_colormap import (
    DATASET_CLASS_MARKERS,
)
from src.plotting.prediction_plot_geometry import (
    build_class_color_map,
)


PCA_OUTPUT_FILENAMES = (
    "pca_physical_classes.png",
    "tsne_physical_classes.png",
    "metrics.txt",
)
_SECTION_RULE = "=" * 70
_PCA_PLOT_ALPHA = 0.30


def save_pca_outputs(
    *,
    output_dir,
    dataset_dir,
    representation,
    source_shape,
    metadata,
    pca_scores,
    explained_variance_ratio,
    tsne_embedding,
    sample_counts,
    cluster_purity,
    cluster_composition,
    class_concentration,
    neighbor_purity,
    kmeans_silhouette,
    weighted_purity,
    kmeans_cluster_count,
    config,
    selected_device,
    analysis_component_count,
):
    """
    Save the complete three-file physical-class PCA output set.

    The two figures use physical metadata as their only category mapping.
    KMeans IDs and all numerical coordinate arrays remain internal while
    their physical-purity diagnostics are formatted into ``metrics.txt``.
    No NumPy array, CSV table, estimator, scaler, or explained-variance plot
    is written.

    Parameters
    ----------
    output_dir : str or pathlib.Path
        Fresh analysis directory receiving exactly three files.
    dataset_dir : str or pathlib.Path
        Dataset identity recorded in the metrics report.
    representation : {"raw", "hermite"}
        Saved representation used for the PCA fit and plot titles.
    source_shape : sequence of int
        Memory-mapped source shape beginning with saved sample order.
    metadata : pandas.DataFrame
        Sample-aligned physical classes, timesteps, cell IDs, and optional
        split values.
    pca_scores : numpy.ndarray
        In-memory PCA scores with shape ``(n_samples, n_components)``.
    explained_variance_ratio : numpy.ndarray
        Per-component explained-variance fractions.
    tsne_embedding : numpy.ndarray
        In-memory label-blind t-SNE coordinates with two columns.
    sample_counts : pandas.DataFrame
        Physical-class counts by reporting split with margins.
    cluster_purity : pandas.DataFrame
        Per-cluster physical-class purity diagnostics.
    cluster_composition : pandas.DataFrame
        Per-cluster physical-class composition counts.
    class_concentration : pandas.DataFrame
        Best diagnostic KMeans partition for every physical class.
    neighbor_purity : pandas.DataFrame
        Nearest-neighbor label-purity summary by split and class.
    kmeans_silhouette : float
        Silhouette score in the shared leading PCA score space.
    weighted_purity : float
        Complete-sample dominant-class fraction across KMeans clusters.
    kmeans_cluster_count : int
        Number of internal partitions derived from physical classes present.
    config : dict
        Resolved PCA, KMeans, t-SNE, plotting, and reporting settings.
    selected_device : str
        PyTorch device that performed scaling and PCA.
    analysis_component_count : int
        Leading PCA width shared by downstream diagnostics and t-SNE.

    Returns
    -------
    dict
        Output filename to :class:`pathlib.Path` mapping for the two PNGs and
        ``metrics.txt``.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        filename: output_dir / filename
        for filename in PCA_OUTPUT_FILENAMES
    }
    class_ids = metadata["class_id"].to_numpy(dtype=np.int64)
    class_names = metadata["class_name"].astype(str).to_numpy()
    ratios = np.asarray(explained_variance_ratio, dtype=np.float64)
    marker_size = float(config["tsne"]["marker_size"])

    save_physical_class_scatter_plot(
        coordinates=np.asarray(pca_scores)[:, :2],
        class_ids=class_ids,
        class_names=class_names,
        path=paths["pca_physical_classes.png"],
        title=(
            f"{representation.capitalize()} PCA by physical class "
            f"(n={len(pca_scores)})"
        ),
        coordinate_labels=(
            f"PC1 ({100.0 * ratios[0]:.1f}$\\%$)",
            f"PC2 ({100.0 * ratios[1]:.1f}$\\%$)",
        ),
        dpi=config["plot"]["dpi"],
        marker_size=marker_size,
        alpha=_PCA_PLOT_ALPHA,
    )
    save_physical_class_scatter_plot(
        coordinates=tsne_embedding,
        class_ids=class_ids,
        class_names=class_names,
        path=paths["tsne_physical_classes.png"],
        title=(
            "t-SNE of leading "
            f"{int(analysis_component_count)} PCA components"
        ),
        coordinate_labels=("t-SNE 1", "t-SNE 2"),
        dpi=config["plot"]["dpi"],
        marker_size=marker_size,
        alpha=float(config["tsne"]["alpha"]),
    )
    metrics_text = create_metrics_text(
        output_dir=output_dir,
        dataset_dir=dataset_dir,
        representation=representation,
        source_shape=source_shape,
        metadata=metadata,
        explained_variance_ratio=ratios,
        sample_counts=sample_counts,
        cluster_purity=cluster_purity,
        cluster_composition=cluster_composition,
        class_concentration=class_concentration,
        neighbor_purity=neighbor_purity,
        kmeans_silhouette=kmeans_silhouette,
        weighted_purity=weighted_purity,
        kmeans_cluster_count=kmeans_cluster_count,
        config=config,
        selected_device=selected_device,
        analysis_component_count=analysis_component_count,
    )
    paths["metrics.txt"].write_text(metrics_text, encoding="utf-8")
    return paths


def save_physical_class_scatter_plot(
    *,
    coordinates,
    class_ids,
    class_names,
    path,
    title,
    coordinate_labels,
    dpi,
    marker_size,
    alpha,
):
    """
    Save one PCA or t-SNE scatter categorized only by physical class.

    The function shares the project's active physical-class palette and
    marker symbols with dataset and prediction figures. Points are small,
    translucent, borderless, and rasterized so an eleven-thousand-sample
    cloud exposes density instead of becoming an opaque overplot. KMeans IDs
    are not accepted by this renderer.

    Parameters
    ----------
    coordinates : numpy.ndarray
        Two-dimensional PCA or t-SNE coordinates in saved sample order.
    class_ids : array-like
        Physical class IDs aligned with ``coordinates``.
    class_names : array-like
        Physical class names aligned with ``class_ids``.
    path : str or pathlib.Path
        PNG path written and closed by this function.
    title : str
        Figure title describing the label-blind coordinate system.
    coordinate_labels : sequence of str
        Horizontal and vertical axis labels.
    dpi : int
        Output resolution in dots per inch.
    marker_size : float
        Matplotlib scatter area used for every physical class.
    alpha : float
        Point opacity used to reveal overlapping sample density.
    """

    figure, axis = plt.subplots(figsize=(8.2, 5.6))
    draw_physical_class_scatter(
        axis,
        coordinates,
        class_ids,
        class_names,
        marker_size=marker_size,
        alpha=alpha,
    )
    axis.set_xlabel(str(coordinate_labels[0]))
    axis.set_ylabel(str(coordinate_labels[1]))
    axis.set_title(str(title))
    axis.grid(alpha=0.15)
    figure.tight_layout(rect=(0.0, 0.0, 0.80, 1.0))
    figure.savefig(path, dpi=int(dpi), bbox_inches="tight")
    plt.close(figure)


def draw_physical_class_scatter(
    axis,
    coordinates,
    class_ids,
    class_names,
    *,
    marker_size,
    alpha,
):
    """Draw physical classes with the current shared colors and markers.

    PCA and t-SNE call this one artist owner after their label-blind fits.
    Marker shape and color both identify the physical class, while small
    rasterized points, zero edge width, and partial opacity expose density.
    KMeans partition IDs are neither accepted nor rendered.

    Parameters
    ----------
    axis : matplotlib.axes.Axes
        Existing PCA or t-SNE axes.
    coordinates : numpy.ndarray
        Two-dimensional coordinates in saved sample order.
    class_ids : array-like
        Current physical class IDs aligned with ``coordinates``.
    class_names : array-like
        Current physical class names aligned with ``class_ids``.
    marker_size : float
        Matplotlib scatter area in points squared.
    alpha : float
        Point opacity used for every physical class.

    Returns
    -------
    tuple of matplotlib.collections.PathCollection
        One point collection per physical class in class-ID order.
    """

    coordinates = np.asarray(coordinates, dtype=np.float64)
    class_ids = np.asarray(class_ids, dtype=np.int64)
    class_names = np.asarray(class_names, dtype=object)
    class_rows = pd.DataFrame(
        {"class_id": class_ids, "class_name": class_names}
    ).drop_duplicates("class_id")
    class_rows = class_rows.sort_values("class_id", kind="stable")
    physical_ids = tuple(class_rows["class_id"].astype(int))
    physical_names = tuple(class_rows["class_name"].astype(str))
    color_map = build_class_color_map(physical_ids, physical_names)
    collections = []
    for class_id, class_name in zip(physical_ids, physical_names):
        class_mask = class_ids == class_id
        collections.append(
            axis.scatter(
                coordinates[class_mask, 0],
                coordinates[class_mask, 1],
                marker=DATASET_CLASS_MARKERS[class_name],
                s=float(marker_size),
                alpha=float(alpha),
                linewidths=0,
                color=color_map[class_id]["color"],
                label=class_name,
                rasterized=True,
            )
        )
    legend = axis.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        fontsize="small",
        markerscale=3.0,
        frameon=True,
    )
    for handle in legend.legend_handles:
        handle.set_alpha(1.0)
    return tuple(collections)


def create_explained_variance_table(explained_variance_ratio):
    """
    Format retained PCA variance for the consolidated text report.

    Parameters
    ----------
    explained_variance_ratio : array-like
        Per-component explained-variance fractions in retained order.

    Returns
    -------
    pandas.DataFrame
        One-based component number, individual explained variance, and
        cumulative explained variance for every retained component.
    """

    ratios = np.asarray(explained_variance_ratio, dtype=np.float64)
    return pd.DataFrame(
        {
            "component": np.arange(1, len(ratios) + 1, dtype=np.int64),
            "explained_variance_ratio": ratios,
            "cumulative_explained_variance_ratio": np.cumsum(ratios),
        }
    )


def create_metrics_text(
    *,
    output_dir,
    dataset_dir,
    representation,
    source_shape,
    metadata,
    explained_variance_ratio,
    sample_counts,
    cluster_purity,
    cluster_composition,
    class_concentration,
    neighbor_purity,
    kmeans_silhouette,
    weighted_purity,
    kmeans_cluster_count,
    config,
    selected_device,
    analysis_component_count,
):
    """
    Create the seven-section physical-class PCA metrics report.

    The report combines run settings, dataset identity, dynamic physical
    sample counts, retained variance, internal KMeans diagnostics, and the
    historical nearest-neighbor label-purity summary. Tables are formatted
    directly from in-memory pandas objects; no array or CSV companion file is
    created.

    Parameters
    ----------
    output_dir, dataset_dir : str or pathlib.Path
        Analysis and dataset locations recorded for reproducibility.
    representation : {"raw", "hermite"}
        Saved feature representation used by PCA.
    source_shape : sequence of int
        Shape of the memory-mapped saved representation.
    metadata : pandas.DataFrame
        Sample-aligned physical class, timestep, cell-ID, and split fields.
    explained_variance_ratio : array-like
        Retained PCA explained-variance fractions.
    sample_counts : pandas.DataFrame
        Split-by-physical-class count table with margins.
    cluster_purity, cluster_composition : pandas.DataFrame
        Internal KMeans physical-purity and composition tables.
    class_concentration : pandas.DataFrame
        Best KMeans partition for every physical class.
    neighbor_purity : pandas.DataFrame
        Nearest-neighbor purity grouped by split and class.
    kmeans_silhouette, weighted_purity : float
        Scalar KMeans diagnostics in physical-label reporting space.
    kmeans_cluster_count : int
        Number of internal KMeans partitions fitted.
    config : dict
        Resolved PCA workflow settings.
    selected_device : str
        PyTorch device that fitted scaling and PCA.
    analysis_component_count : int
        Leading PCA width shared by downstream methods.

    Returns
    -------
    str
        UTF-8-compatible report text ending with a newline.
    """

    source_shape = tuple(int(value) for value in source_shape)
    feature_count = int(np.prod(source_shape[1:], dtype=np.int64))
    class_rows = (
        metadata.loc[:, ["class_id", "class_name"]]
        .drop_duplicates("class_id")
        .sort_values("class_id", kind="stable")
    )
    physical_class_names = tuple(class_rows["class_name"].astype(str))
    class_counts = (
        metadata["class_name"]
        .astype(str)
        .value_counts(sort=False)
        .reindex(physical_class_names)
        .rename_axis("class_name")
        .reset_index(name="count")
    )
    explained_variance = create_explained_variance_table(
        explained_variance_ratio
    )
    tsne_config = config["tsne"]
    pca_config = config["pca"]

    sections = (
        (
            "PCA analysis run",
            [
                f"dataset_dir: {Path(dataset_dir).resolve()}",
                f"output_dir: {Path(output_dir).resolve()}",
                f"representation: {representation}",
                f"source array shape: {source_shape}",
                f"number of samples: {len(metadata)}",
                f"number of features: {feature_count}",
                "number of unique timesteps: "
                f"{metadata['timestep'].nunique()}",
                f"number of unique cell IDs: {metadata['cid'].nunique()}",
                f"number of physical classes: {len(physical_class_names)}",
                "physical class names: "
                f"{', '.join(physical_class_names)}",
                f"PCA method: {pca_config['method']}",
                f"PCA device: {selected_device}",
                f"PCA sample batch size: {pca_config['batch_size']}",
                f"retained PCA components: {pca_config['n_components']}",
                "PCA low-rank oversampling: "
                f"{pca_config['lowrank_oversampling']}",
                f"PCA low-rank iterations: {pca_config['lowrank_niter']}",
                "downstream analysis components: "
                f"{int(analysis_component_count)}",
                f"random seed: {config['random_state']}",
                f"KMeans cluster count: {kmeans_cluster_count}",
                f"KMeans algorithm: {config['kmeans']['algorithm']}",
                f"KMeans initializations: {config['kmeans']['n_init']}",
                f"KMeans maximum iterations: {config['kmeans']['max_iter']}",
                f"nearest-neighbor count: {config['nearest_neighbors']}",
                f"t-SNE output dimensions: {tsne_config['n_components']}",
                f"t-SNE perplexity: {tsne_config['perplexity']}",
                "t-SNE input component count: "
                f"{int(analysis_component_count)}",
                f"t-SNE initialization: {tsne_config['init']}",
                f"t-SNE learning rate: {tsne_config['learning_rate']}",
                "t-SNE early exaggeration: "
                f"{tsne_config['early_exaggeration']}",
                f"t-SNE iteration count: {tsne_config['max_iter']}",
                f"scatter marker size: {tsne_config['marker_size']}",
                f"t-SNE scatter alpha: {tsne_config['alpha']}",
                f"plot DPI: {config['plot']['dpi']}",
            ],
        ),
        (
            "Dataset summary",
            [
                f"samples: {len(metadata)}",
                f"unique timesteps: {metadata['timestep'].nunique()}",
                f"minimum timestep: {metadata['timestep'].min()}",
                f"maximum timestep: {metadata['timestep'].max()}",
                f"unique cell IDs: {metadata['cid'].nunique()}",
                f"physical class count: {len(physical_class_names)}",
                "",
                _format_table(class_counts),
            ],
        ),
        (
            "Sample counts by split and class",
            [sample_counts.to_string()],
        ),
        (
            "Explained variance",
            [_format_table(explained_variance)],
        ),
        (
            "KMeans purity against physical classes",
            [
                f"PCA components used: {int(analysis_component_count)}",
                f"KMeans clusters: {kmeans_cluster_count}",
                f"Silhouette score: {float(kmeans_silhouette):.6f}",
                f"Weighted purity: {float(weighted_purity):.6f}",
                "",
                "Per-cluster purity",
                _format_table(cluster_purity),
                "",
                "Physical-class composition counts",
                _format_table(cluster_composition),
            ],
        ),
        (
            "KMeans physical-class concentration",
            [_format_table(class_concentration)],
        ),
        (
            "Nearest-neighbor label purity",
            [
                f"Neighbor count: {config['nearest_neighbors']}",
                _format_table(neighbor_purity),
            ],
        ),
    )
    return "\n\n".join(
        "\n".join((heading, _SECTION_RULE, *lines))
        for heading, lines in sections
    ) + "\n"


def _format_table(frame):
    """Render one in-memory metrics table for the plain-text report.

    Parameters
    ----------
    frame : pandas.DataFrame
        Complete table for one report section.

    Returns
    -------
    str
        Untruncated table text with floating-point values formatted to six
        decimal places. No CSV companion is created.
    """

    return frame.to_string(
        index=False,
        float_format=lambda value: f"{value:.6f}",
    )
