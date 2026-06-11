from pathlib import Path

from joblib import Parallel, delayed
import analysator as pt
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from src.colormap_helpers import expr_velocity, scatter_label_points
from src.vdf_helpers import get_vdf_plot_parameters_from_file

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
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    extent = np.asarray(extent)

    vxmin = extent[0]
    vzmin = extent[2]
    vxmax = extent[3]
    vzmax = extent[5]

    vdf_swapped = np.swapaxes(vdf, 2, 0)

    mid = vdf_swapped.shape[1] // 2

    vdf_plot = vdf_swapped[:, mid, :] * dv
    vdf_plot = np.where(vdf_plot < threshold * dv, 0, vdf_plot)
    vdf_plot = np.ma.masked_less_equal(vdf_plot, 0)

    fig, ax1 = plt.subplots(figsize=(7, 6))

    im = ax1.imshow(
        vdf_plot,
        origin="lower",
        extent=[
        vxmin / 1000, vxmax / 1000,
        vzmin / 1000, vzmax / 1000
    ],

        norm="log",
        cmap="nipy_spectral",
    )

    ax1.grid(color="gray", axis="both")
    ax1.set_xlim(-vdflim / 1000, vdflim / 1000)
    ax1.set_ylim(-vdflim / 1000, vdflim / 1000)
    ax1.set_xlabel("v_x")
    ax1.set_ylabel("v_z")

    title_parts = [
        f"timestep={metadata_row.get('timestep', 'unknown')}",
        f"cid={metadata_row.get('cid', 'unknown')}",
    ]

    class_name = metadata_row.get("class_name")
    if class_name is not None:
        title_parts.append(f"class={class_name}")

    if y_label is not None:
        title_parts.append(f"true={int(y_label)}")

    if predicted_class_name is not None:
        title_parts.append(f"pred={predicted_class_name}")

    if decision_score is not None:
        title_parts.append(f"score={decision_score:.3g}")

    ax1.set_title(", ".join(title_parts))

    fig.colorbar(im, ax=ax1, label="f(v)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_labeled_colormap(metadata_rows, output_path, boxre, vmin, vmax):
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

    scatter_label_points(ax1, metadata_rows)

    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

def run_plot_jobs(plot_function, plot_jobs, n_jobs):
    """
    Run plotting jobs serially or in parallel.

    Parameters
    ----------
    plot_function : callable
        Plotting function called with each job dictionary as keyword arguments.
    plot_jobs : list of dict
        Plot job dictionaries.
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
        Colormap plotting options.

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
            }
        )

    return colormap_jobs

def create_vdf_plot_jobs(X, y, metadata, output_dir, vdflim):
    """
    Create plot jobs for all saved VDF samples.

    Parameters
    ----------
    X : numpy.ndarray
        VDF sample array.
    y : numpy.ndarray
        Integer labels for VDF samples.
    metadata : pandas.DataFrame
        Dataset metadata with one row per sample.
    output_dir : pathlib.Path
        Base plot output directory for the dataset.
    vdflim : float
        Velocity axis limit in m/s for VDF plots.

    Returns
    -------
    list of dict
        Keyword argument dictionaries for ``plot_vdf_xz_slice``.
    """

    class_frame_counts = {}
    plot_parameter_cache = {}
    vdf_jobs = []

    for sample_index in range(X.shape[0]):
        metadata_row = metadata.iloc[sample_index]
        class_name = metadata_row["class_name"]
        file_location = metadata_row["file_location"]
        cid = int(metadata_row["cid"])

        if class_name not in class_frame_counts:
            class_frame_counts[class_name] = 0

        class_frame_index = class_frame_counts[class_name]
        vdf_shape = tuple(X[sample_index].shape)
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

        vdf_jobs.append(
            {
                "vdf": X[sample_index],
                "y_label": y[sample_index],
                "metadata_row": metadata_row,
                "extent": extent,
                "output_path": output_path,
                "dv": dv,
                "threshold": threshold,
                "vdflim": vdflim,
            }
        )

        class_frame_counts[class_name] += 1

    return vdf_jobs