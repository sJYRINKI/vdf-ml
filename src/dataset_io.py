from pathlib import Path
import numpy as np
import pandas as pd

def create_dataset_output_dir(output_dir, start_timestep, n_timestep):
    """
    Create the output directory for dataset.

    Parameters
    ----------
    output_dir : str
        Base directory where dataset folder are saved.
    start_timestep : int
        First timestep in the dataset.
    n_timesteps : int
        Number of timesteps in dataset.

    Returns
    -------
    pathlib.Path
        Directory path for dataset output.
    """

    outdir = Path(output_dir) / f"timesteps_{int(start_timestep)}_{int(n_timestep)}"
    outdir.mkdir(parents=True, exist_ok=True)

    return outdir

def save_dataset(outdir, X, y, metadata):
    """
    Save VDF data, labels, and metadata.

    Parameters
    ----------
    outdir : str
        output directory for the dataset.
    X : numpy.ndarray
        VDF samples.
    y : numpy.ndarray
        Integer labels.
    metadata : list of dict
        Metadata rows, one dictionary per sample.
    """

    outdir = Path(outdir)

    np.savez_compressed(
        outdir / "vdf.npz",
        X=X,
        y=y,
    )

    pd.DataFrame(metadata).to_csv(
        outdir / "metadata.csv",
        index=False
    )