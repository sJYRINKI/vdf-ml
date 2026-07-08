from pathlib import Path
import time

from joblib import Parallel, delayed
import analysator as pt
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np

from src.config import load_config
from src.colormap_helpers import (
    draw_manual_point_search_boxes,
    draw_o_point_search_areas,
    draw_point_boxes,
    draw_x_point_search_areas,
    expr_velocity,
    scatter_all_vdf_cells,
    scatter_label_points,
)
from src.timesteps import create_path
from src.vdf_helpers import get_vdf_plot_axes_parameters, get_vdf_plot_threshold


_MEMMAP_CACHE = {}


def add_dataset_sampling_plot_config(colormap_config, dataset_config_path):
    """
    Add dataset sampling settings needed for colormap overlays.

    The plotting config keeps visual options such as color scale and plot bounds,
    while the dataset-creation config owns the point class names and, for old
    datasets, may also contain the VDF box size used for X-point and O-point
    samples. This helper copies those sampling settings into a colormap config
    dictionary when they are available.

    Parameters
    ----------
    colormap_config : dict
        Colormap plotting options from the plotting config.
    dataset_config_path : str or pathlib.Path
        Path to the dataset-creation YAML config containing ``points`` and
        optionally ``vdf_box`` sections.

    Returns
    -------
    dict
        Copy of ``colormap_config`` with ``box_classes`` and optional
        ``vdf_box`` added.
    """

    dataset_config = load_config(dataset_config_path)
    points_config = dataset_config.get("points", {})
    box_classes = [
        points_config[class_key]
        for class_key in ("x_class_name", "o_class_name")
        if class_key in points_config
    ]

    colormap_config = dict(colormap_config)
    if "vdf_box" in dataset_config:
        colormap_config["vdf_box"] = dataset_config["vdf_box"]
    colormap_config["box_classes"] = box_classes
    colormap_config["file_template_flux"] = dataset_config.get("file_template_flux")
    colormap_config["x_selection"] = points_config.get("x_selection")
    colormap_config["o_core_fraction"] = (
        points_config.get("o_selection", {}).get("core_fraction")
    )

    return colormap_config


def create_colormap_plot_jobs(metadata, output_dir, colormap_config):
    """
    Create plot jobs for one labeled colormap per timestep.

    Parameters
    ----------
    metadata : pandas.DataFrame
        Dataset metadata containing timestep, class, coordinate, and file columns.
    output_dir : pathlib.Path
        Base plot output directory for the dataset.
    colormap_config : dict
        Colormap plotting options. This may include ``boxre``, ``vmin``,
        ``vmax``, ``vdf_box``, and ``box_classes``.

    Returns
    -------
    list of dict
        Keyword argument dictionaries for ``plot_labeled_colormap``.
    """

    colormap_jobs = []

    for frame_index, (_, timestep_metadata) in enumerate(metadata.groupby("timestep")):
        colormap_output_path = output_dir / "colormaps" / f"colormap_{frame_index:04d}.png"

        colormap_jobs.append(
            {
                "metadata_rows": timestep_metadata,
                "output_path": colormap_output_path,
                "boxre": colormap_config.get("boxre", [-40, -1, -6, 6]),
                "vmin": float(colormap_config.get("vmin", -1.5e6)),
                "vmax": float(colormap_config.get("vmax", 1.5e6)),
                "vdf_box_config": colormap_config.get("vdf_box"),
                "box_classes": colormap_config.get("box_classes", []),
                "x_selection_config": colormap_config.get("x_selection"),
                "flux_file_template": colormap_config.get("file_template_flux"),
                "o_core_fraction": colormap_config.get("o_core_fraction"),
            }
        )

    return colormap_jobs


def iter_vdf_plot_jobs(X, y, metadata, output_dir, vdflim, X_plot=None):
    """
    Yield lightweight plot jobs for saved VDF samples.

    Parameters
    ----------
    X : numpy.ndarray
        Memory-mapped VDF sample array.
    y : numpy.ndarray
        Integer labels for VDF samples.
    metadata : pandas.DataFrame
        Dataset metadata with one row per sample.
    output_dir : pathlib.Path
        Base plot output directory for the dataset.
    vdflim : float
        Velocity axis limit in m/s for VDF plots.
    X_plot : numpy.ndarray, optional
        Memory-mapped cache of physical plot-oriented xz slices with shape
        ``(n_samples, vz, vx)``. If omitted, plots are made from full VDFs in
        ``X``.

    Yields
    ------
    dict
        Keyword argument dictionary for ``plot_vdf_sample_from_dataset``.
    """

    X_path = get_memmap_path(X, "X")
    X_plot_path = (
        None
        if X_plot is None
        else get_memmap_path(X_plot, "X_plot")
    )
    class_frame_counts = {}
    plot_axes_cache = {}
    plot_threshold_cache = {}
    reader_cache = {}
    vdf_shape = tuple(X.shape[1:])

    for sample_index in range(X.shape[0]):
        metadata_row = metadata.iloc[sample_index].to_dict()
        class_name = metadata_row["class_name"]
        file_location = metadata_row["file_location"]
        cid = int(metadata_row["cid"])

        if class_name not in class_frame_counts:
            class_frame_counts[class_name] = 0

        class_frame_index = class_frame_counts[class_name]
        file_key = str(file_location)
        extent, dv, threshold = _get_cached_vdf_plot_parameters(
            reader_cache=reader_cache,
            plot_axes_cache=plot_axes_cache,
            plot_threshold_cache=plot_threshold_cache,
            file_location=file_key,
            cid=cid,
            vdf_shape=vdf_shape,
        )

        class_output_dir = output_dir / class_name
        output_path = class_output_dir / f"sample_{class_frame_index:04d}_xz.png"

        yield {
            "X_path": X_path,
            "X_plot_path": X_plot_path,
            "sample_index": int(sample_index),
            "y_label": int(y[sample_index]),
            "metadata_row": metadata_row,
            "extent": extent,
            "output_path": output_path,
            "dv": dv,
            "threshold": threshold,
            "vdflim": vdflim,
        }

        class_frame_counts[class_name] += 1


def run_plot_jobs(plot_function, plot_jobs, n_jobs):
    """
    Run plotting jobs serially or in parallel.

    Parameters
    ----------
    plot_function : callable
        Plotting function called with each job dictionary as keyword arguments.
    plot_jobs : iterable of dict
        Plot job dictionaries. Each dictionary is expanded into keyword
        arguments for ``plot_function``.
    n_jobs : int
        Number of parallel workers. Use 1 for serial plotting.
    """

    if n_jobs == 1:
        for plot_job in plot_jobs:
            plot_function(**plot_job)
        return

    Parallel(n_jobs=n_jobs)(
        delayed(plot_function)(**plot_job)
        for plot_job in plot_jobs
    )


def plot_labeled_colormap(
        metadata_rows,
        output_path,
        boxre,
        vmin,
        vmax,
        vdf_box_config=None,
        box_classes=None,
        x_selection_config=None,
        flux_file_template=None,
        o_core_fraction=None,
):
    """
    Plot and save a spatial colormap with saved label points overlaid.

    Parameters
    ----------
    metadata_rows : pandas.DataFrame
        Metadata rows for one timestep.
    output_path : str
        Output PNG path.
    boxre : list of float
        Plot box in Earth radii: ``[xmin, xmax, zmin, zmax]``.
    vmin : float
        Minimum color scale value for the selected velocity component.
    vmax : float
        Maximum color scale value for the selected velocity component.
    vdf_box_config : dict, optional
        VDF sampling box config from dataset creation. Expected keys are
        ``x_half_width_re`` and ``z_half_width_re``. The boxes are drawn around
        the source X/O point coordinates stored in metadata.
    box_classes : iterable of str, optional
        Class names whose source coordinates should get boxes, normally the
        configured X-point and O-point classes.
    x_selection_config : dict, optional
        X-point selection config used to redraw Hessian-aligned search boxes.
    flux_file_template : str, optional
        Flux file template used to redraw O-point island search contours.
    o_core_fraction : float, optional
        Fraction from O-point flux to boundary flux used for old metadata that
        does not contain ``search_flux``.
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    file_location = metadata_rows.iloc[0]["file_location"]

    fig, ax1 = plt.subplots(figsize=(8, 5))

    pt.plot.plot_colormap(
        filename=file_location,
        axes=ax1,
        boxre=boxre,
        expression=expr_velocity,
        operator="x",
        vmin=vmin,
        vmax=vmax,
        streamlines="B",
        streamlinecolor="black",
    )

    reader = pt.vlsvfile.VlsvReader(str(file_location))
    scatter_all_vdf_cells(
        ax=ax1,
        reader=reader,
        boxre=boxre,
    )
    draw_point_boxes(
        ax=ax1,
        metadata_rows=metadata_rows,
        box_config=vdf_box_config,
        box_classes=box_classes,
    )
    draw_manual_point_search_boxes(
        ax=ax1,
        metadata_rows=metadata_rows,
    )
    draw_x_point_search_areas(
        ax=ax1,
        metadata_rows=metadata_rows,
        x_selection_config=x_selection_config,
    )
    draw_o_point_search_areas(
        ax=ax1,
        reader=reader,
        metadata_rows=metadata_rows,
        flux_file_template=flux_file_template,
        o_core_fraction=o_core_fraction,
    )
    scatter_label_points(
        ax=ax1,
        reader=reader,
        metadata_rows=metadata_rows,
    )
    ax1.legend(loc=2)

    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def get_memmap_path(array, array_name):
    """
    Return the backing filename for a memory-mapped array.

    Parameters
    ----------
    array : numpy.ndarray
        Array expected to be backed by a ``.npy`` memmap.
    array_name : str
        Human-readable array name used in error messages.

    Returns
    -------
    str
        Memmap backing filename.
    """

    filename = getattr(array, "filename", None)
    if filename is None:
        raise ValueError(f"{array_name} must be loaded as a memory-mapped array")

    return str(filename)


def load_memmap_array(array_path):
    """
    Load and cache a read-only memory-mapped array.

    Parameters
    ----------
    array_path : str
        Path to a ``.npy`` array.

    Returns
    -------
    numpy.ndarray
        Read-only memory-mapped array.
    """

    array_path = str(array_path)
    array = _MEMMAP_CACHE.get(array_path)
    if array is None:
        array = np.load(array_path, mmap_mode="r")
        _MEMMAP_CACHE[array_path] = array

    return array


def create_or_load_plot_xz_slice_cache(X, dataset_dir, dataset_id, cache_config):
    """
    Create or load cached physical plot-oriented xz VDF slices.

    Parameters
    ----------
    X : numpy.ndarray
        VDF samples with shape ``(n_samples, vx, vy, vz)``.
    dataset_dir : str or pathlib.Path
        Dataset directory containing ``X.npy``.
    dataset_id : str
        Dataset identifier used in cache path templates.
    cache_config : dict
        Plot cache settings.

    Returns
    -------
    numpy.ndarray or None
        Memory-mapped xz-slice cache, or ``None`` when caching is disabled.
    """

    cache_config = resolve_plot_xz_slice_cache_config(
        cache_config=cache_config,
        dataset_dir=dataset_dir,
        dataset_id=dataset_id,
    )
    if not cache_config["enabled"]:
        return None

    cache_path = cache_config["cache_path"]
    metadata_path = cache_config["metadata_path"]

    if (
            not cache_config["rebuild"]
            and is_plot_xz_slice_cache_valid(
                X=X,
                cache_path=cache_path,
                metadata_path=metadata_path,
            )
    ):
        print(f"Using plot xz-slice cache: {cache_path}")
        return np.load(cache_path, mmap_mode="r")

    create_plot_xz_slice_cache(
        X=X,
        cache_config=cache_config,
    )

    return np.load(cache_path, mmap_mode="r")


def resolve_plot_xz_slice_cache_config(cache_config, dataset_dir, dataset_id):
    """
    Resolve plot xz-slice cache paths and settings.

    Parameters
    ----------
    cache_config : dict
        Raw cache settings from plotting config.
    dataset_dir : str or pathlib.Path
        Dataset directory containing ``X.npy``.
    dataset_id : str
        Dataset identifier used in cache path templates.

    Returns
    -------
    dict
        Resolved cache settings.
    """

    cache_config = cache_config or {}
    dataset_dir = Path(dataset_dir)
    enabled = bool(cache_config.get("enabled", False))

    cache_dir_template = cache_config.get("dir")
    if cache_dir_template is None:
        cache_dir = dataset_dir / "cache"
    else:
        cache_dir = create_path(
            path_template=cache_dir_template,
            timestep=dataset_id,
            dataset_id=dataset_id,
        )

    batch_size = int(cache_config.get("batch_size", 128))
    n_jobs = int(cache_config.get("n_jobs", 1))

    if batch_size <= 0:
        raise ValueError("plot.cache.batch_size must be positive")

    if n_jobs == 0:
        raise ValueError("plot.cache.n_jobs must be non-zero")

    return {
        "enabled": enabled,
        "cache_dir": Path(cache_dir),
        "cache_path": Path(cache_dir) / cache_config.get(
            "filename",
            "plot_xz_slice.npy",
        ),
        "metadata_path": Path(cache_dir) / cache_config.get(
            "metadata_filename",
            "plot_xz_slice_metadata.npz",
        ),
        "rebuild": bool(cache_config.get("rebuild", False)),
        "batch_size": batch_size,
        "n_jobs": n_jobs,
    }


def create_plot_xz_slice_cache(X, cache_config):
    """
    Create a memory-mapped cache of physical plot-oriented xz VDF slices.

    Parameters
    ----------
    X : numpy.ndarray
        VDF samples with shape ``(n_samples, vx, vy, vz)``.
    cache_config : dict
        Resolved cache settings.

    Returns
    -------
    dict
        Cache metadata.
    """

    cache_path = cache_config["cache_path"]
    metadata_path = cache_config["metadata_path"]
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    cache_shape = (int(X.shape[0]), *infer_plot_xz_slice_shape(X))
    X_plot = np.lib.format.open_memmap(
        cache_path,
        mode="w+",
        dtype=np.float32,
        shape=cache_shape,
    )

    start = time.perf_counter()
    sample_indices = np.arange(int(X.shape[0]), dtype=int)
    batches = list(
        iter_index_batches(
            sample_indices,
            batch_size=cache_config["batch_size"],
        )
    )

    print(f"Creating plot xz-slice cache: {cache_path}")
    print(f"Cache shape: {cache_shape}")
    print(f"Cache jobs: {cache_config['n_jobs']}")

    if cache_config["n_jobs"] == 1:
        for batch_indices in batches:
            write_plot_xz_slice_cache_batch(
                X=X,
                X_plot=X_plot,
                batch_indices=batch_indices,
            )
    else:
        Parallel(
            n_jobs=cache_config["n_jobs"],
            prefer="threads",
            require="sharedmem",
        )(
            delayed(write_plot_xz_slice_cache_batch)(
                X=X,
                X_plot=X_plot,
                batch_indices=batch_indices,
            )
            for batch_indices in batches
        )

    X_plot.flush()
    elapsed = time.perf_counter() - start
    metadata = {
        "enabled": True,
        "cache_path": str(cache_path),
        "metadata_path": str(metadata_path),
        "raw_vdf_shape": tuple(int(value) for value in X.shape),
        "cache_shape": tuple(int(value) for value in cache_shape),
        "slice": "xz",
        "orientation": "plot",
        "physical_units": "vdf",
        "elapsed_seconds": float(elapsed),
    }
    save_plot_xz_slice_cache_metadata(
        metadata_path=metadata_path,
        metadata=metadata,
    )

    print(f"Created plot xz-slice cache in {elapsed:.2f} s")

    return metadata


def iter_index_batches(indices, batch_size):
    """
    Iterate fixed-size index batches.

    Parameters
    ----------
    indices : array-like of int
        Indices to split.
    batch_size : int
        Number of indices per batch.

    Yields
    ------
    numpy.ndarray
        Index batch.
    """

    indices = np.asarray(indices, dtype=int)

    for start in range(0, len(indices), int(batch_size)):
        yield indices[start:start + int(batch_size)]


def write_plot_xz_slice_cache_batch(X, X_plot, batch_indices):
    """
    Write one batch of physical plot-oriented xz slices into the cache.

    Parameters
    ----------
    X : numpy.ndarray
        VDF samples with shape ``(n_samples, vx, vy, vz)``.
    X_plot : numpy.ndarray
        Plot xz-slice cache.
    batch_indices : array-like of int
        Sample indices to write.
    """

    for sample_index in batch_indices:
        X_plot[int(sample_index)] = extract_plot_xz_slice_from_dataset(
            X=X,
            sample_index=int(sample_index),
        )


def extract_plot_xz_slice_from_dataset(X, sample_index):
    """
    Extract the plot-oriented middle xz slice from a saved sample.

    Parameters
    ----------
    X : numpy.ndarray
        VDF samples with shape ``(n_samples, vx, vy, vz)``.
    sample_index : int
        Dataset sample index.

    Returns
    -------
    numpy.ndarray
        Plot-oriented xz slice with shape ``(vz, vx)``.
    """

    mid_y = X.shape[2] // 2
    return np.asarray(X[int(sample_index), :, mid_y, :].T, dtype=np.float32)


def infer_plot_xz_slice_shape(X):
    """
    Infer plot-oriented xz-slice shape from a VDF dataset.

    Parameters
    ----------
    X : numpy.ndarray
        VDF samples with shape ``(n_samples, vx, vy, vz)``.

    Returns
    -------
    tuple of int
        Plot-oriented xz-slice shape ``(vz, vx)``.
    """

    return int(X.shape[3]), int(X.shape[1])


def is_plot_xz_slice_cache_valid(X, cache_path, metadata_path):
    """
    Return whether an existing plot xz-slice cache matches the dataset.

    Parameters
    ----------
    X : numpy.ndarray
        VDF samples with shape ``(n_samples, vx, vy, vz)``.
    cache_path : str or pathlib.Path
        Cache array path.
    metadata_path : str or pathlib.Path
        Cache metadata path.

    Returns
    -------
    bool
        Whether the cache can be reused.
    """

    cache_path = Path(cache_path)
    metadata_path = Path(metadata_path)
    if not cache_path.exists() or not metadata_path.exists():
        return False

    try:
        X_plot = np.load(cache_path, mmap_mode="r")
        metadata = load_plot_xz_slice_cache_metadata(metadata_path)
    except Exception:
        return False

    expected_shape = (int(X.shape[0]), *infer_plot_xz_slice_shape(X))

    return (
        tuple(X_plot.shape) == expected_shape
        and tuple(metadata.get("raw_vdf_shape", ())) == tuple(X.shape)
        and tuple(metadata.get("cache_shape", ())) == expected_shape
        and metadata.get("slice") == "xz"
        and metadata.get("orientation") == "plot"
        and metadata.get("physical_units") == "vdf"
    )


def save_plot_xz_slice_cache_metadata(metadata_path, metadata):
    """
    Save plot xz-slice cache metadata.

    Parameters
    ----------
    metadata_path : str or pathlib.Path
        Cache metadata path.
    metadata : dict
        Cache metadata.
    """

    metadata_path = Path(metadata_path)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        metadata_path,
        enabled=np.asarray(metadata["enabled"]),
        cache_path=np.asarray(metadata["cache_path"]),
        metadata_path=np.asarray(metadata["metadata_path"]),
        raw_vdf_shape=np.asarray(metadata["raw_vdf_shape"], dtype=int),
        cache_shape=np.asarray(metadata["cache_shape"], dtype=int),
        slice=np.asarray(metadata["slice"]),
        orientation=np.asarray(metadata["orientation"]),
        physical_units=np.asarray(metadata["physical_units"]),
        elapsed_seconds=np.asarray(metadata["elapsed_seconds"]),
    )


def load_plot_xz_slice_cache_metadata(metadata_path):
    """
    Load plot xz-slice cache metadata.

    Parameters
    ----------
    metadata_path : str or pathlib.Path
        Cache metadata path.

    Returns
    -------
    dict
        Cache metadata.
    """

    with np.load(metadata_path, allow_pickle=False) as metadata:
        return {
            "enabled": bool(metadata["enabled"].item()),
            "cache_path": str(metadata["cache_path"].item()),
            "metadata_path": str(metadata["metadata_path"].item()),
            "raw_vdf_shape": tuple(
                int(value) for value in metadata["raw_vdf_shape"]
            ),
            "cache_shape": tuple(int(value) for value in metadata["cache_shape"]),
            "slice": str(metadata["slice"].item()),
            "orientation": str(metadata["orientation"].item()),
            "physical_units": str(metadata["physical_units"].item()),
            "elapsed_seconds": float(metadata["elapsed_seconds"].item()),
        }


def plot_vdf_sample_from_dataset(
        X_path,
        X_plot_path,
        sample_index,
        y_label,
        metadata_row,
        extent,
        output_path,
        dv,
        threshold,
        vdflim=2e6,
):
    """
    Plot and save one VDF sample loaded from a memory-mapped dataset.

    Parameters
    ----------
    X_path : str
        Path to the saved VDF ``.npy`` array.
    X_plot_path : str or None
        Path to cached plot-oriented xz slices. If ``None``, the full VDF is
        read from ``X_path``.
    sample_index : int
        Sample index to plot.
    y_label : int
        Integer label.
    metadata_row : dict
        Metadata row for the sample.
    extent : array-like of float
        Velocity mesh extent ``[vxmin, vymin, vzmin, vxmax, vymax, vzmax]``.
    output_path : str
        Output PNG path.
    dv : float
        Velocity grid cell size in m/s.
    threshold : float
        VDF threshold before multiplying by ``dv``.
    vdflim : float, optional
        Visible velocity limit in m/s.
    """

    if X_plot_path is not None:
        X_plot = load_memmap_array(X_plot_path)
        plot_vdf_xz_slice_from_physical_xz(
            vdf_xz_slice=X_plot[int(sample_index)],
            y_label=y_label,
            metadata_row=metadata_row,
            extent=extent,
            output_path=output_path,
            dv=dv,
            threshold=threshold,
            vdflim=vdflim,
        )
        return

    X = load_memmap_array(X_path)
    plot_vdf_xz_slice(
        vdf=X[int(sample_index)],
        y_label=y_label,
        metadata_row=metadata_row,
        extent=extent,
        output_path=output_path,
        dv=dv,
        threshold=threshold,
        vdflim=vdflim,
    )


def plot_vdf_xz_slice_from_physical_xz(
        vdf_xz_slice,
        y_label,
        metadata_row,
        extent,
        output_path,
        dv,
        threshold,
        vdflim=2e6,
):
    """
    Plot and save one cached physical xz VDF slice.

    Parameters
    ----------
    vdf_xz_slice : numpy.ndarray
        Plot-oriented physical xz slice with shape ``(vz, vx)``.
    y_label : int
        Integer label.
    metadata_row : dict
        Metadata row for the sample.
    extent : array-like of float
        Velocity mesh extent ``[vxmin, vymin, vzmin, vxmax, vymax, vzmax]``.
    output_path : str
        Output PNG path.
    dv : float
        Velocity grid cell size in m/s.
    threshold : float
        VDF threshold before multiplying by ``dv``.
    vdflim : float, optional
        Visible velocity limit in m/s.
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax1 = plt.subplots(figsize=(7, 6))
    im = _plot_physical_xz_slice_on_axis(
        ax=ax1,
        vdf_xz_slice=vdf_xz_slice,
        y_label=y_label,
        metadata_row=metadata_row,
        extent=extent,
        dv=dv,
        threshold=threshold,
        vdflim=vdflim,
    )

    if im is None:
        plt.close(fig)
        return

    fig.colorbar(im, ax=ax1, label="f(v)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_vdf_xz_slice(
        vdf,
        y_label,
        metadata_row,
        extent,
        output_path,
        dv,
        threshold,
        vdflim=2e6,
        decision_score=None,
        predicted_class_name=None,
):
    """
    Plot and save an xz VDF slice from one saved VDF sample.

    Parameters
    ----------
    vdf : numpy.ndarray
        3D VDF array with axis order ``[vx, vy, vz]``.
    y_label : int
        Integer label.
    metadata_row : pandas.Series
        Metadata row for the sample.
    extent : array-like of float
        Velocity mesh extent ``[vxmin, vymin, vzmin, vxmax, vymax, vzmax]``.
    output_path : str
        Output PNG path.
    dv : float
        Velocity grid cell size in m/s.
    threshold : float
        VDF threshold before multiplying by ``dv``.
    vdflim : float, optional
        Visible velocity limit in m/s.
    decision_score : float, optional
        Optional model score or probability to show in the plot title.
    predicted_class_name : str, optional
        Optional predicted class name to show in the plot title.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax1 = plt.subplots(figsize=(7, 6))
    im = _plot_vdf_xz_slice_on_axis(
        ax=ax1,
        vdf=vdf,
        y_label=y_label,
        metadata_row=metadata_row,
        extent=extent,
        dv=dv,
        threshold=threshold,
        vdflim=vdflim,
        decision_score=decision_score,
        predicted_class_name=predicted_class_name,
    )

    if im is None:
        plt.close(fig)
        return

    fig.colorbar(im, ax=ax1, label="f(v)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_random_class_samples(
        X,
        y,
        metadata,
        output_path,
        vdflim=2e6,
        random_state=None,
        n_columns=2,
):
    """
    Plot one random xz VDF sample from each class in one figure.

    Parameters
    ----------
    X : numpy.ndarray
        Memory-mapped VDF sample array.
    y : numpy.ndarray
        Integer labels for VDF samples.
    metadata : pandas.DataFrame
        Dataset metadata with one row per sample and a ``class_name`` column.
    output_path : str or pathlib.Path
        Output PNG path.
    vdflim : float, optional
        Visible velocity limit in m/s.
    random_state : int, optional
        Random seed for reproducible class sampling.
    n_columns : int, optional
        Number of subplot columns.

    Returns
    -------
    dict
        Mapping from class name to selected sample index.
    """

    if n_columns <= 0:
        raise ValueError("n_columns must be positive")

    selected_indices = _select_random_class_sample_indices(
        metadata=metadata,
        random_state=random_state,
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_samples = len(selected_indices)
    n_rows = (n_samples + n_columns - 1) // n_columns

    plot_axes_cache = {}
    plot_threshold_cache = {}
    reader_cache = {}
    plot_items = []
    vdf_shape = tuple(X.shape[1:])

    for class_name, sample_index in selected_indices.items():
        metadata_row = metadata.iloc[sample_index].to_dict()
        file_location = metadata_row["file_location"]
        cid = int(metadata_row["cid"])
        file_key = str(file_location)
        extent, dv, threshold = _get_cached_vdf_plot_parameters(
            reader_cache=reader_cache,
            plot_axes_cache=plot_axes_cache,
            plot_threshold_cache=plot_threshold_cache,
            file_location=file_key,
            cid=cid,
            vdf_shape=vdf_shape,
        )
        vdf_plot = _prepare_vdf_xz_plot(
            vdf=X[sample_index],
            metadata_row=metadata_row,
            dv=dv,
            threshold=threshold,
        )

        plot_items.append(
            {
                "class_name": class_name,
                "sample_index": sample_index,
                "y_label": y[sample_index],
                "metadata_row": metadata_row,
                "extent": extent,
                "vdf_plot": vdf_plot,
            }
        )

    fig, axes = plt.subplots(
        n_rows,
        n_columns,
        figsize=(5 * n_columns, 5 * n_rows),
        squeeze=False,
    )

    for plot_index, plot_item in enumerate(plot_items):
        ax = axes.flat[plot_index]
        vdf_plot = plot_item["vdf_plot"]

        if vdf_plot is None:
            ax.set_axis_off()
            ax.set_title(str(plot_item["class_name"]), fontsize=11)
            continue

        im = _plot_prepared_vdf_xz_slice_on_axis(
            ax=ax,
            vdf_plot=vdf_plot,
            y_label=plot_item["y_label"],
            metadata_row=plot_item["metadata_row"],
            extent=plot_item["extent"],
            vdflim=vdflim,
            title=str(plot_item["class_name"]),
        )
        _add_matching_colorbar(fig=fig, ax=ax, im=im)

    for ax in axes.flat[n_samples:]:
        ax.set_axis_off()

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return selected_indices


def _get_cached_vdf_plot_parameters(
        reader_cache,
        plot_axes_cache,
        plot_threshold_cache,
        file_location,
        cid,
        vdf_shape,
):
    """
    Return VDF plot parameters using per-file and per-cell caches.

    Parameters
    ----------
    reader_cache : dict
        Mapping from VLSV file path to open reader.
    plot_axes_cache : dict
        Mapping from ``(file_location, vdf_shape)`` to ``(extent, dv)``.
    plot_threshold_cache : dict
        Mapping from ``(file_location, cid)`` to VDF sparsity threshold.
    file_location : str or pathlib.Path
        Path to the VLSV file.
    cid : int
        Spatial cell ID.
    vdf_shape : tuple of int
        Shape of one saved VDF sample.

    Returns
    -------
    extent : numpy.ndarray
        Velocity mesh extent.
    dv : float
        Velocity cell size.
    threshold : float
        VDF sparsity threshold.
    """

    file_location = str(file_location)
    cid = int(cid)
    axes_key = (file_location, tuple(vdf_shape))
    threshold_key = (file_location, cid)

    reader = _get_cached_vlsv_reader(
        reader_cache=reader_cache,
        file_location=file_location,
    )

    if axes_key not in plot_axes_cache:
        plot_axes_cache[axes_key] = get_vdf_plot_axes_parameters(
            reader=reader,
            vdf_shape=vdf_shape,
        )
    if threshold_key not in plot_threshold_cache:
        plot_threshold_cache[threshold_key] = get_vdf_plot_threshold(
            reader=reader,
            cid=cid,
        )

    extent, dv = plot_axes_cache[axes_key]
    threshold = plot_threshold_cache[threshold_key]

    return extent, dv, threshold


def _get_cached_vlsv_reader(reader_cache, file_location):
    """
    Return an open VLSV reader from a per-call cache.

    Parameters
    ----------
    reader_cache : dict
        Mapping from VLSV file path to open reader.
    file_location : str or pathlib.Path
        Path to the VLSV file.

    Returns
    -------
    analysator.vlsvfile.VlsvReader
        Open reader for ``file_location``.
    """

    file_location = str(file_location)
    reader = reader_cache.get(file_location)
    if reader is None:
        reader = pt.vlsvfile.VlsvReader(file_location)
        reader_cache[file_location] = reader

    return reader


def _select_random_class_sample_indices(metadata, random_state=None):
    """
    Select one random sample index from each class.

    Parameters
    ----------
    metadata : pandas.DataFrame
        Dataset metadata with a ``class_name`` column. If available, the
        ``sample_index`` column is used as the index into ``X`` and ``y``.
    random_state : int, optional
        Random seed for reproducible class sampling.

    Returns
    -------
    dict
        Mapping from class name to selected sample index.
    """

    if "class_name" not in metadata.columns:
        raise ValueError("metadata must contain a class_name column")

    if metadata.empty:
        raise ValueError("metadata must contain at least one sample")

    rng = np.random.default_rng(random_state)
    selected_indices = {}

    for class_name, class_rows in metadata.groupby("class_name", sort=True):
        if "sample_index" in class_rows.columns:
            candidate_indices = class_rows["sample_index"].to_numpy(dtype=int)
        else:
            candidate_indices = class_rows.index.to_numpy(dtype=int)

        selected_indices[str(class_name)] = int(rng.choice(candidate_indices))

    return selected_indices


def _add_matching_colorbar(fig, ax, im):
    """
    Add a colorbar with the same height as its VDF subplot.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure containing the subplot.
    ax : matplotlib.axes.Axes
        Axis whose height the colorbar should match.
    im : matplotlib.image.AxesImage
        Image object used for colorbar scaling.
    """

    divider = make_axes_locatable(ax)
    colorbar_ax = divider.append_axes("right", size="4%", pad=0.05)
    colorbar = fig.colorbar(im, cax=colorbar_ax, label="f(v)")
    colorbar.ax.tick_params(labelsize=8)


def _plot_vdf_xz_slice_on_axis(
        ax,
        vdf,
        y_label,
        metadata_row,
        extent,
        dv,
        threshold,
        vdflim=2e6,
        decision_score=None,
        predicted_class_name=None,
        sample_index=None,
):
    """
    Plot one xz VDF slice on an existing matplotlib axis.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis where the VDF slice is drawn.
    vdf : numpy.ndarray
        3D VDF array with axis order ``[vx, vy, vz]``.
    y_label : int
        Integer label.
    metadata_row : dict or pandas.Series
        Metadata row for the sample.
    extent : array-like of float
        Velocity mesh extent ``[vxmin, vymin, vzmin, vxmax, vymax, vzmax]``.
    dv : float
        Velocity grid cell size in m/s.
    threshold : float
        VDF threshold before multiplying by ``dv``.
    vdflim : float, optional
        Visible velocity limit in m/s.
    decision_score : float, optional
        Optional model score or probability to show in the plot title.
    predicted_class_name : str, optional
        Optional predicted class name to show in the plot title.
    sample_index : int, optional
        Dataset sample index to show in the plot title.

    Returns
    -------
    matplotlib.image.AxesImage or None
        Image object for colorbar creation, or ``None`` for empty samples.
    """

    vdf_plot = _prepare_vdf_xz_plot(
        vdf=vdf,
        metadata_row=metadata_row,
        dv=dv,
        threshold=threshold,
    )

    if vdf_plot is None:
        return None

    return _plot_prepared_vdf_xz_slice_on_axis(
        ax=ax,
        vdf_plot=vdf_plot,
        y_label=y_label,
        metadata_row=metadata_row,
        extent=extent,
        vdflim=vdflim,
        decision_score=decision_score,
        predicted_class_name=predicted_class_name,
        sample_index=sample_index,
    )


def _plot_physical_xz_slice_on_axis(
        ax,
        vdf_xz_slice,
        y_label,
        metadata_row,
        extent,
        dv,
        threshold,
        vdflim=2e6,
):
    """
    Plot one cached physical xz VDF slice on an existing matplotlib axis.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis where the VDF slice is drawn.
    vdf_xz_slice : numpy.ndarray
        Plot-oriented physical xz slice with shape ``(vz, vx)``.
    y_label : int
        Integer label.
    metadata_row : dict or pandas.Series
        Metadata row for the sample.
    extent : array-like of float
        Velocity mesh extent ``[vxmin, vymin, vzmin, vxmax, vymax, vzmax]``.
    dv : float
        Velocity grid cell size in m/s.
    threshold : float
        VDF threshold before multiplying by ``dv``.
    vdflim : float, optional
        Visible velocity limit in m/s.

    Returns
    -------
    matplotlib.image.AxesImage or None
        Image object for colorbar creation, or ``None`` for empty samples.
    """

    vdf_plot = _prepare_physical_xz_plot(
        vdf_xz_slice=vdf_xz_slice,
        metadata_row=metadata_row,
        dv=dv,
        threshold=threshold,
    )

    if vdf_plot is None:
        return None

    return _plot_prepared_vdf_xz_slice_on_axis(
        ax=ax,
        vdf_plot=vdf_plot,
        y_label=y_label,
        metadata_row=metadata_row,
        extent=extent,
        vdflim=vdflim,
    )


def _prepare_vdf_xz_plot(vdf, metadata_row, dv, threshold):
    """
    Create a thresholded xz VDF slice for plotting.

    Parameters
    ----------
    vdf : numpy.ndarray
        3D VDF array with axis order ``[vx, vy, vz]``.
    metadata_row : dict or pandas.Series
        Metadata row for the sample.
    dv : float
        Velocity grid cell size in m/s.
    threshold : float
        VDF threshold before multiplying by ``dv``.

    Returns
    -------
    numpy.ma.MaskedArray or None
        Thresholded xz VDF slice multiplied by ``dv``, or ``None`` when the
        sample has no positive values.
    """

    return _prepare_physical_xz_plot(
        vdf_xz_slice=extract_plot_xz_slice(vdf),
        metadata_row=metadata_row,
        dv=dv,
        threshold=threshold,
    )


def extract_plot_xz_slice(vdf):
    """
    Extract the plot-oriented middle xz slice from one dense VDF.

    Parameters
    ----------
    vdf : numpy.ndarray
        Dense VDF array with shape ``(vx, vy, vz)``.

    Returns
    -------
    numpy.ndarray
        Plot-oriented xz slice with shape ``(vz, vx)``.
    """

    vdf_swapped = np.swapaxes(vdf, 2, 0)
    mid = vdf_swapped.shape[1] // 2

    return np.asarray(vdf_swapped[:, mid, :], dtype=np.float32)


def _prepare_physical_xz_plot(vdf_xz_slice, metadata_row, dv, threshold):
    """
    Create a thresholded xz VDF plot slice from a physical xz slice.

    Parameters
    ----------
    vdf_xz_slice : numpy.ndarray
        Plot-oriented physical xz slice with shape ``(vz, vx)``.
    metadata_row : dict or pandas.Series
        Metadata row for the sample.
    dv : float
        Velocity grid cell size in m/s.
    threshold : float
        VDF threshold before multiplying by ``dv``.

    Returns
    -------
    numpy.ma.MaskedArray or None
        Thresholded xz VDF slice multiplied by ``dv``, or ``None`` when the
        sample has no positive values.
    """

    vdf_plot_raw = np.asarray(vdf_xz_slice, dtype=np.float32) * dv
    vdf_plot = np.where(vdf_plot_raw < threshold * dv, 0, vdf_plot_raw)
    vdf_plot = np.ma.masked_less_equal(vdf_plot, 0)

    if vdf_plot.count() == 0:
        print(
            "Using unthresholded VDF plot: "
            f"timestep={metadata_row.get('timestep', 'unknown')}, "
            f"cid={metadata_row.get('cid', 'unknown')}, "
            f"class={metadata_row.get('class_name', 'unknown')}"
        )
        vdf_plot = np.ma.masked_less_equal(vdf_plot_raw, 0)

    if vdf_plot.count() == 0:
        print(
            "Skipping empty VDF plot: "
            f"timestep={metadata_row.get('timestep', 'unknown')}, "
            f"cid={metadata_row.get('cid', 'unknown')}, "
            f"class={metadata_row.get('class_name', 'unknown')}"
        )
        return None

    return vdf_plot


def _plot_prepared_vdf_xz_slice_on_axis(
        ax,
        vdf_plot,
        y_label,
        metadata_row,
        extent,
        vdflim=2e6,
        decision_score=None,
        predicted_class_name=None,
        sample_index=None,
        title=None,
):
    """
    Plot a prepared xz VDF slice on an existing matplotlib axis.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis where the VDF slice is drawn.
    vdf_plot : numpy.ma.MaskedArray
        Thresholded xz VDF slice multiplied by ``dv``.
    y_label : int
        Integer label.
    metadata_row : dict or pandas.Series
        Metadata row for the sample.
    extent : array-like of float
        Velocity mesh extent ``[vxmin, vymin, vzmin, vxmax, vymax, vzmax]``.
    vdflim : float, optional
        Visible velocity limit in m/s.
    decision_score : float, optional
        Optional model score or probability to show in the plot title.
    predicted_class_name : str, optional
        Optional predicted class name to show in the plot title.
    sample_index : int, optional
        Dataset sample index to show in the plot title.
    title : str, optional
        Explicit plot title. When omitted, a metadata title is created.

    Returns
    -------
    matplotlib.image.AxesImage
        Image object for colorbar creation.
    """

    extent = np.asarray(extent)

    vxmin = extent[0]
    vzmin = extent[2]
    vxmax = extent[3]
    vzmax = extent[5]

    im = ax.imshow(
        vdf_plot,
        origin="lower",
        extent=[
        vxmin / 1000, vxmax / 1000,
        vzmin / 1000, vzmax / 1000
    ],

        norm="log",
        cmap="nipy_spectral",
    )

    ax.grid(color="gray", axis="both")
    ax.set_xlim(-vdflim / 1000, vdflim / 1000)
    ax.set_ylim(-vdflim / 1000, vdflim / 1000)
    ax.set_xlabel("v_x")
    ax.set_ylabel("v_z")

    if title is not None:
        ax.set_title(title)
        return im

    title_parts = [
        f"timestep={metadata_row.get('timestep', 'unknown')}",
        f"cid={metadata_row.get('cid', 'unknown')}",
    ]

    if sample_index is not None:
        title_parts.insert(0, f"sample={int(sample_index)}")

    class_name = metadata_row.get("class_name")
    if class_name is not None:
        title_parts.append(f"class={class_name}")

    if y_label is not None:
        title_parts.append(f"true={int(y_label)}")

    if predicted_class_name is not None:
        title_parts.append(f"pred={predicted_class_name}")

    if decision_score is not None:
        title_parts.append(f"score={decision_score:.3g}")

    ax.set_title(", ".join(title_parts))

    return im
