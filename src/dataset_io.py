from pathlib import Path
import mmap
import tempfile
import numpy as np
import pandas as pd

from src.vdf_extract import (
    create_velocity_grid_descriptor,
    normalize_velocity_grid_descriptor,
    velocity_grid_descriptors_match,
)


VELOCITY_GRID_FILENAME = "velocity_grid.npz"
VELOCITY_GRID_PREPROCESSING_KEYS = {
    "shape": "velocity_grid_shape",
    "extent_mps": "velocity_grid_extent_mps",
    "population": "velocity_grid_population",
    "axis_order": "velocity_grid_axis_order",
    "slice_axis": "velocity_grid_slice_axis",
    "slice_index": "velocity_grid_slice_index",
}


def create_dataset_output_dir(
    output_dir,
    start_timestep,
    n_timestep,
    name_suffix=None,
):
    """
    Create the output directory for dataset.

    Parameters
    ----------
    output_dir : str
        Base directory where dataset folder are saved.
    start_timestep : int
        First timestep in the dataset.
    n_timestep : int
        Number of timesteps in dataset.
    name_suffix : str, optional
        Extra suffix appended to the dataset directory name.

    Returns
    -------
    pathlib.Path
        Directory path for dataset output.
    """

    dirname = f"timesteps_{int(start_timestep)}_{int(n_timestep)}"
    if name_suffix is not None:
        name_suffix = str(name_suffix).strip()
        if name_suffix:
            if "/" in name_suffix or "\\" in name_suffix:
                raise ValueError("Dataset name suffix must not contain path separators")
            if any(character.isspace() for character in name_suffix):
                raise ValueError("Dataset name suffix must not contain whitespace")
            dirname = f"{dirname}_{name_suffix}"

    outdir = Path(output_dir) / dirname
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


def save_velocity_grid_descriptor(dataset_dir, velocity_grid):
    """
    Save one dataset-level velocity-grid descriptor.

    Parameters
    ----------
    dataset_dir : str or pathlib.Path
        Dataset directory containing the saved VDF arrays.
    velocity_grid : dict
        Velocity-grid descriptor to save.

    Returns
    -------
    pathlib.Path
        Saved descriptor path.
    """

    dataset_dir = Path(dataset_dir)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    velocity_grid = normalize_velocity_grid_descriptor(velocity_grid)
    output_path = dataset_dir / VELOCITY_GRID_FILENAME
    _save_npz_atomic(output_path, velocity_grid)

    return output_path


def load_velocity_grid_descriptor(dataset_dir):
    """
    Load one dataset-level velocity-grid descriptor.

    Parameters
    ----------
    dataset_dir : str or pathlib.Path
        Dataset directory containing ``velocity_grid.npz``.

    Returns
    -------
    dict
        Normalized velocity-grid descriptor.
    """

    velocity_grid_path = Path(dataset_dir) / VELOCITY_GRID_FILENAME
    if not velocity_grid_path.is_file():
        raise FileNotFoundError(
            f"Dataset velocity grid does not exist: {velocity_grid_path}. "
            "Run scripts/backfill_dataset_velocity_grid.py first."
        )

    with np.load(velocity_grid_path, allow_pickle=False) as values:
        velocity_grid = {
            key: values[key]
            for key in VELOCITY_GRID_PREPROCESSING_KEYS
            if key in values.files
        }

    return normalize_velocity_grid_descriptor(velocity_grid)


def create_velocity_grid_preprocessing_values(velocity_grid):
    """
    Create model preprocessing values from a velocity-grid descriptor.

    Parameters
    ----------
    velocity_grid : dict
        Dataset velocity-grid descriptor.

    Returns
    -------
    dict
        Values ready to add to ``preprocessing.npz``.
    """

    velocity_grid = normalize_velocity_grid_descriptor(velocity_grid)

    return {
        preprocessing_key: velocity_grid[descriptor_key]
        for descriptor_key, preprocessing_key
        in VELOCITY_GRID_PREPROCESSING_KEYS.items()
    }


def load_velocity_grid_from_preprocessing(preprocessing):
    """
    Load the training velocity grid from model preprocessing values.

    Parameters
    ----------
    preprocessing : numpy.lib.npyio.NpzFile or dict
        Loaded model preprocessing values.

    Returns
    -------
    dict
        Normalized training velocity-grid descriptor.
    """

    if hasattr(preprocessing, "files"):
        available_keys = set(preprocessing.files)
    else:
        available_keys = set(preprocessing)
    required_keys = set(VELOCITY_GRID_PREPROCESSING_KEYS.values())
    missing_keys = required_keys - available_keys
    if missing_keys:
        raise ValueError(
            "Model preprocessing has no complete training velocity grid. "
            "Backfill the dataset and model preprocessing with "
            "scripts/backfill_dataset_velocity_grid.py. Missing keys: "
            f"{sorted(missing_keys)}"
        )

    velocity_grid = {
        descriptor_key: preprocessing[preprocessing_key]
        for descriptor_key, preprocessing_key
        in VELOCITY_GRID_PREPROCESSING_KEYS.items()
    }

    return normalize_velocity_grid_descriptor(velocity_grid)


def backfill_dataset_velocity_grid(
    dataset_dir,
    model_dirs=None,
    overwrite=False,
):
    """
    Backfill dataset and model velocity-grid information.

    The first source VLSV file supplies the physical velocity extent. The
    stored dataset array shape is checked against that source grid without
    loading VDF samples into memory.

    Parameters
    ----------
    dataset_dir : str or pathlib.Path
        Dataset directory containing ``X.npy`` and ``metadata.csv``.
    model_dirs : iterable of str or pathlib.Path, optional
        Model directories whose existing ``preprocessing.npz`` files should
        receive the backfilled training grid.
    overwrite : bool, optional
        Whether conflicting existing grid information may be replaced.

    Returns
    -------
    dict
        Dataset grid path and model preprocessing paths.
    """

    dataset_dir = Path(dataset_dir)
    X_path = dataset_dir / "X.npy"
    metadata_path = dataset_dir / "metadata.csv"
    if not X_path.is_file():
        raise FileNotFoundError(f"Dataset array does not exist: {X_path}")
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Dataset metadata does not exist: {metadata_path}")

    velocity_grid_path = dataset_dir / VELOCITY_GRID_FILENAME
    keep_existing_grid = velocity_grid_path.is_file() and not overwrite
    if keep_existing_grid:
        velocity_grid = load_velocity_grid_descriptor(dataset_dir)
    else:
        metadata = pd.read_csv(
            metadata_path,
            usecols=["file_location"],
            nrows=1,
        )
        if metadata.empty:
            raise ValueError(f"Dataset metadata has no rows: {metadata_path}")
        file_location_value = metadata.iloc[0]["file_location"]
        if pd.isna(file_location_value):
            raise ValueError("First dataset metadata row has no file_location")
        file_location = str(file_location_value).strip()
        if not file_location:
            raise ValueError("First dataset metadata row has no file_location")
        if not Path(file_location).is_file():
            raise FileNotFoundError(
                f"Training VLSV file does not exist: {file_location}"
            )

        import analysator as pt

        reader = pt.vlsvfile.VlsvReader(file_location)
        velocity_grid = create_velocity_grid_descriptor(reader=reader)

    X = np.load(X_path, mmap_mode="r")
    dataset_shape = tuple(int(value) for value in X.shape[1:])
    grid_shape = tuple(int(value) for value in velocity_grid["shape"])
    if dataset_shape != grid_shape:
        raise ValueError(
            "Dataset VDF shape does not match source velocity grid: "
            f"{dataset_shape} != {grid_shape}"
        )

    if not keep_existing_grid:
        velocity_grid_path = save_velocity_grid_descriptor(
            dataset_dir=dataset_dir,
            velocity_grid=velocity_grid,
        )
    if isinstance(model_dirs, (str, Path)):
        model_dirs = [model_dirs]
    dataset_id = dataset_dir.name.removeprefix("timesteps_")
    preprocessing_paths = []
    for model_dir in model_dirs or []:
        preprocessing_paths.append(
            _backfill_model_velocity_grid(
                model_dir=model_dir,
                velocity_grid=velocity_grid,
                dataset_id=dataset_id,
                overwrite=overwrite,
            )
        )

    return {
        "velocity_grid_path": velocity_grid_path,
        "preprocessing_paths": preprocessing_paths,
    }


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


def _backfill_model_velocity_grid(
    model_dir,
    velocity_grid,
    dataset_id,
    overwrite,
):
    """Add a training velocity grid to one model preprocessing archive."""

    preprocessing_path = Path(model_dir) / "preprocessing.npz"
    if not preprocessing_path.is_file():
        raise FileNotFoundError(
            f"Model preprocessing does not exist: {preprocessing_path}"
        )

    with np.load(preprocessing_path, allow_pickle=False) as preprocessing:
        values = {
            key: preprocessing[key]
            for key in preprocessing.files
        }
    if "dataset_id" not in values:
        raise ValueError(
            f"Model preprocessing has no dataset_id: {preprocessing_path}"
        )
    model_dataset_id = str(np.asarray(values["dataset_id"]).item())
    if model_dataset_id != str(dataset_id):
        raise ValueError(
            "Model dataset_id does not match the selected dataset: "
            f"{model_dataset_id!r} != {str(dataset_id)!r}"
        )
    existing_grid_keys = (
        set(values) & set(VELOCITY_GRID_PREPROCESSING_KEYS.values())
    )
    if existing_grid_keys:
        if existing_grid_keys != set(VELOCITY_GRID_PREPROCESSING_KEYS.values()):
            if not overwrite:
                raise ValueError(
                    "Model preprocessing contains an incomplete velocity grid: "
                    f"{preprocessing_path}"
                )
        else:
            existing_grid = load_velocity_grid_from_preprocessing(values)
            if not velocity_grid_descriptors_match(
                existing_grid,
                velocity_grid,
            ) and not overwrite:
                raise ValueError(
                    "Model preprocessing contains a conflicting velocity grid: "
                    f"{preprocessing_path}"
                )
            if velocity_grid_descriptors_match(existing_grid, velocity_grid):
                return preprocessing_path

    values.update(create_velocity_grid_preprocessing_values(velocity_grid))
    _save_npz_atomic(preprocessing_path, values)

    return preprocessing_path


def _save_npz_atomic(output_path, values):
    """Write one NumPy archive through a same-directory temporary file."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_mode = (
        output_path.stat().st_mode & 0o777
        if output_path.exists()
        else 0o644
    )
    with tempfile.NamedTemporaryFile(
        dir=output_path.parent,
        prefix=f".{output_path.stem}.",
        suffix=".npz",
        delete=False,
    ) as temporary_file:
        temporary_path = Path(temporary_file.name)

    try:
        np.savez(temporary_path, **values)
        temporary_path.chmod(output_mode)
        temporary_path.replace(output_path)
    finally:
        temporary_path.unlink(missing_ok=True)
