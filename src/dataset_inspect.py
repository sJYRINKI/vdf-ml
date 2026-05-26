from pathlib import Path
import numpy as np
import pandas as pd

def load_dataset(dataset_dir):
    """
    Load saved VDF dataset.

    Parameters
    ----------
    dataset_ddir : str
        Directory containing ``vdf.npz`` and ``metadata.csv``.
    
    Returns
    -------
    X : numpy.ndarray
        VDF samples.
    y : numpy.ndarray
        Integer labels.
    metadata : pandas.DataFrame
        Metadata table for the samples.
    """

    dataset_dir = Path(dataset_dir)

    npz_path = dataset_dir / "vdf.npz"
    metadata_path = dataset_dir / "metadata.csv"

    data = np.load(npz_path)
    X = data["X"]
    y = data["y"]

    metadata = pd.read_csv(metadata_path)

    return X, y, metadata