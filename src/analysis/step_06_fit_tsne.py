"""Stage 6: fit label-blind visualization-only t-SNE in PCA space.

This stage follows post-fit cluster evaluation and precedes output saving.
It is separate because the embedding is only a visualization coordinate
system and must never influence KMeans or purity.

The stage receives the shared leading-component score view, large-dataset
t-SNE settings, and the workflow random seed. It returns the fitted
estimator and a float64 two-dimensional embedding that remains in memory
until the physical-class figure is rendered.
"""

import numpy as np
from sklearn.manifold import TSNE


def fit_tsne(pca_scores, config, random_state):
    """
    Fit t-SNE to the selected leading PCA score columns.

    Stage 0 supplies only the first shared ``analysis_components`` PCA score
    columns. The default perplexity of 50, PCA initialization, and increased
    early exaggeration target broader neighborhoods in the current large
    dataset, while Stage 7's smaller translucent markers reduce visual
    overplotting. Physical and KMeans labels are absent from fitting.

    Parameters
    ----------
    pca_scores : numpy.ndarray
        Leading PCA coordinates with shape
        ``(n_samples, n_analysis_components)``.
    config : dict
        t-SNE settings including perplexity, learning rate, early
        exaggeration, iteration count, initialization, and output dimension.
    random_state : int
        Shared deterministic workflow seed.

    Returns
    -------
    model : sklearn.manifold.TSNE
        Fitted visualization estimator.
    embedding : numpy.ndarray
        Float64 t-SNE coordinates with shape ``(n_samples, 2)`` retained only
        for physical-class plotting.

    Notes
    -----
    t-SNE distances are visualization coordinates, not quantitative global
    distances. The method does not force physical classes apart, and the
    embedding is not saved as a numerical artifact.
    """

    model = TSNE(
        n_components=config["n_components"],
        perplexity=config["perplexity"],
        learning_rate=config["learning_rate"],
        early_exaggeration=config["early_exaggeration"],
        max_iter=config["max_iter"],
        init=config["init"],
        random_state=random_state,
    )
    embedding = np.asarray(
        model.fit_transform(pca_scores),
        dtype=np.float64,
    )
    return model, embedding
