import numpy as np

from src.batches import iter_array_batches


def print_dataset_info(X, y, metadata):
    """
    Print info about a loaded VDF dataset.

    Parameters
    ----------
    X : numpy.ndarray
        VDF samples.
    y : numpy.ndarray
        Integer labels.
    metadata : pandas.DataFrame
        Metadata table for the samples.
    """
    print("Dataset information:")
    print(f"X shape: {X.shape}")
    print(f"X dtype: {X.dtype}")
    print(f"y shape: {y.shape}")
    print(f"y dtype: {y.dtype}")
    print(f"metadata shape: {metadata.shape}")
    print("\n")

    print("Metadata:")
    print(metadata)
    print("\n")

    if "class_name" in metadata.columns:
        print("Class counts:")
        print(metadata["class_name"].value_counts())

    print("\n")


def print_vdf_statistics(X, batch_size=64):
    """
    Print statistics for VDF samples.

    Parameters
    ----------
    X : numpy.ndarray
        VDF samples.
    batch_size : int
        Size of batches to process.
    """
    global_min = np.inf
    global_max = -np.inf
    total_sum = 0.0
    total_sum_sq = 0.0
    total_count = 0
    
    for _, batch in iter_array_batches(X, batch_size=batch_size):
        batch = np.asarray(batch, dtype=np.float64)

        global_min = min(global_min, batch.min())
        global_max = max(global_max, batch.max())

        total_sum += batch.sum()
        total_sum_sq += (batch ** 2).sum()
        total_count += batch.size
    
    mean = total_sum / total_count
    variance = (total_sum_sq / total_count) - (mean ** 2)
    variance = max(variance, 0.0)
    std = np.sqrt(variance)

    print("VDF statistic:")
    print(f"min: {global_min}")
    print(f"max: {global_max}")
    print(f"mean: {mean}")
    print(f"std: {std}")
