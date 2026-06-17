from pathlib import Path
import mmap
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


def create_memmap_dataset(outdir,n_samples, sample_shape, dtype=np.float32):
    """
    Create memory-mapped dataset file.

    Parameters
    ----------
    outdir : str
        Directory where dataset file is saved.
    n_samples : int
        Number of samples in the dataset.
    sample_shape : tuple of int
        Shape of one VDF sample.
    dtype : data-type, optional
        Desired data type for the array.

    Returns
    -------
    X : numpy.memmap
        Memory mapped array for the VDF.
    y : numpy.memmap
        Memory mapped array for the labels.
    """

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    X = np.lib.format.open_memmap(
        outdir / "X.npy",
        mode="w+",
        dtype=dtype,
        shape=(int(n_samples), *sample_shape)
    )

    y = np.lib.format.open_memmap(
        outdir / "y.npy",
        mode="w+",
        dtype=np.int64,
        shape=(int(n_samples),),
    )

    return X, y


def release_memmap_pages(array):
    """
    Ask the operating system to drop cached pages for a memmap.

    Parameters
    ----------
    array : numpy.memmap
        Memory-mapped array whose resident pages should be released.

    Returns
    -------
    bool
        Whether page release was requested successfully.
    """

    mmap_object = getattr(array, "_mmap", None)
    madvise = getattr(mmap_object, "madvise", None)
    dontneed = getattr(mmap, "MADV_DONTNEED", None)

    if madvise is None or dontneed is None:
        return False

    try:
        madvise(dontneed)
    except (OSError, ValueError):
        return False

    return True


def flush_and_release_memmaps(*arrays):
    """
    Flush memmap arrays and request release of their resident pages.

    Parameters
    ----------
    *arrays : numpy.memmap
        Memory-mapped arrays to flush and release.
    """

    for array in arrays:
        array.flush()

    for array in arrays:
        release_memmap_pages(array)


def save_metadata(outdir, metadata):
    """
    Save dataset metadata.

    Parameters
    ----------
    outdir : str
        Directory where metadata file is saved.
    metadata : list of dict
        Metadata rows, one row per sample.
    """
    
    outdir = Path(outdir)

    pd.DataFrame(metadata).to_csv(
        outdir / "metadata.csv",
        index=False
    )


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
    outdir.mkdir(parents=True, exist_ok=True)

    np.save(outdir / "X.npy", np.asanyarray(X, dtype=np.float32))
    np.save(outdir / "y.npy", np.asanyarray(y, dtype=np.int64))

    save_metadata(outdir, metadata)


def load_dataset(dataset_dir, mmap=True):
    """
    Load saved VDF dataset.

    Parameters
    ----------
    dataset_dir : str
        Directory containing dataset arrays and metadata.
    mmap : bool, optional
        Whether to load the dataset as a memory-mapped array.

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

    X_path = dataset_dir / "X.npy"
    y_path = dataset_dir / "y.npy"
    metadata_path = dataset_dir / "metadata.csv"

    if mmap:
        X = np.load(X_path, mmap_mode="r")
        y = np.load(y_path, mmap_mode="r")
    else:
        X = np.load(X_path)
        y = np.load(y_path)

    metadata = pd.read_csv(metadata_path)

    return X, y, metadata
