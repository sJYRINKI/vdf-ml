"""Stage 4: fit an internal KMeans diagnostic in PCA score space.

This stage follows PCA fitting and precedes physical-class diagnostics. It
keeps unsupervised partitioning isolated from the labels used only to choose
the requested partition count and evaluate the fitted result.

The stage receives the selected leading PCA scores, KMeans numerical
settings, the number of physical classes present, and the shared seed. It
returns the fitted estimator, in-memory partition IDs, and their silhouette
score. No KMeans labels are saved or presented as another classification.
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def fit_kmeans(pca_scores, config, random_state, n_clusters):
    """
    Fit label-blind KMeans to the selected leading PCA score space.

    Stage 0 has already selected the shared leading-component view used by
    KMeans, nearest-neighbor diagnostics, and t-SNE. The physical class count
    determines only how many KMeans partitions are requested; class labels
    never enter ``fit_predict`` or the silhouette calculation. Stage 5 uses
    the returned IDs only to measure post-fit physical composition.

    Parameters
    ----------
    pca_scores : numpy.ndarray
        PCA coordinates with shape ``(n_samples, n_pca_components)``.
    config : dict
        KMeans initialization count, iteration limit, and algorithm.
    random_state : int
        Shared deterministic workflow seed.
    n_clusters : int
        Number of KMeans partitions, equal to the number of physical classes
        present in the current metadata.

    Returns
    -------
    model : sklearn.cluster.KMeans
        Fitted internal diagnostic estimator.
    labels : numpy.ndarray
        Int64 KMeans partition IDs in unchanged sample order.
    silhouette : float
        Mean silhouette coefficient in the same PCA score space used by
        KMeans.

    Notes
    -----
    Cluster IDs are retained exactly as assigned by scikit-learn while the
    workflow remains in memory. They are not remapped, plotted, or saved.
    """

    model = KMeans(
        n_clusters=n_clusters,
        n_init=config["n_init"],
        max_iter=config["max_iter"],
        algorithm=config["algorithm"],
        random_state=random_state,
    )
    labels = np.asarray(model.fit_predict(pca_scores), dtype=np.int64)
    silhouette = float(silhouette_score(pca_scores, labels))
    return model, labels, silhouette
