import numpy as np

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

def print_vdf_statisctics(X):
    """
    Print statistics fo VDF samples.

    Parameters
    ----------
    X : numpy.ndarray
        VDF samples.
    """

    print("VDF statistic:")
    print(f"min: {X.min()}")
    print(f"max: {X.max()}")
    print(f"mean: {X.mean()}")
    print(f"std: {X.std()}")