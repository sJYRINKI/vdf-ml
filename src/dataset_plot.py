from pathlib import Path

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
from src.vdf_helpers import get_vdf_plot_parameters_from_file


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


def iter_vdf_plot_jobs(X, y, metadata, output_dir, vdflim):
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

    Yields
    ------
    dict
        Keyword argument dictionary for ``plot_vdf_sample_from_dataset``.
    """

    X_path = get_memmap_path(X, "X")
    class_frame_counts = {}
    plot_parameter_cache = {}
    vdf_shape = tuple(X.shape[1:])

    for sample_index in range(X.shape[0]):
        metadata_row = metadata.iloc[sample_index].to_dict()
        class_name = metadata_row["class_name"]
        file_location = metadata_row["file_location"]
        cid = int(metadata_row["cid"])

        if class_name not in class_frame_counts:
            class_frame_counts[class_name] = 0

        class_frame_index = class_frame_counts[class_name]
        cache_key = (file_location, cid, vdf_shape)

        if cache_key not in plot_parameter_cache:
            plot_parameter_cache[cache_key] = get_vdf_plot_parameters_from_file(
                file_location=file_location,
                cid=cid,
                vdf_shape=vdf_shape,
            )

        extent, dv, threshold = plot_parameter_cache[cache_key]

        class_output_dir = output_dir / class_name
        output_path = class_output_dir / f"sample_{class_frame_index:04d}_xz.png"

        yield {
            "X_path": X_path,
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


def plot_vdf_sample_from_dataset(
        X_path,
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

    plot_parameter_cache = {}
    plot_items = []
    vdf_shape = tuple(X.shape[1:])

    for class_name, sample_index in selected_indices.items():
        metadata_row = metadata.iloc[sample_index].to_dict()
        file_location = metadata_row["file_location"]
        cid = int(metadata_row["cid"])
        cache_key = (file_location, cid, vdf_shape)

        if cache_key not in plot_parameter_cache:
            plot_parameter_cache[cache_key] = get_vdf_plot_parameters_from_file(
                file_location=file_location,
                cid=cid,
                vdf_shape=vdf_shape,
            )

        extent, dv, threshold = plot_parameter_cache[cache_key]
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

    vdf_swapped = np.swapaxes(vdf, 2, 0)
    mid = vdf_swapped.shape[1] // 2

    vdf_plot_raw = vdf_swapped[:, mid, :] * dv
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
