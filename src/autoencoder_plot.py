from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np


def plot_reconstruction_pair(
    original_slice,
    reconstructed_slice,
    metadata_row,
    output_path,
    extent=None,
    dv=1.0,
    threshold=0.0,
    vdflim=2e6,
    split_name=None,
    sample_index=None,
):
    """
    Plot original and reconstructed VDF slices side by side.

    Parameters
    ----------
    original_slice : numpy.ndarray
        Original physical VDF xz slice.
    reconstructed_slice : numpy.ndarray
        Reconstructed physical VDF xz slice.
    metadata_row : dict
        Metadata for the sample.
    output_path : str or pathlib.Path
        Path where the figure is saved.
    extent : array-like of float, optional
        Velocity mesh extent ``[vxmin, vymin, vzmin, vxmax, vymax, vzmax]``.
    dv : float, optional
        Velocity grid cell size in m/s.
    threshold : float, optional
        VDF threshold before multiplying by ``dv``.
    vdflim : float, optional
        Visible velocity limit in m/s.
    split_name : str, optional
        Dataset split name.
    sample_index : int, optional
        Dataset sample index.

    Returns
    -------
    bool
        Whether a plot was saved.
    """

    original_plot = prepare_vdf_slice_for_plot(
        vdf_slice=original_slice,
        metadata_row=metadata_row,
        dv=dv,
        threshold=threshold,
    )
    reconstructed_plot = prepare_vdf_slice_for_plot(
        vdf_slice=reconstructed_slice,
        metadata_row=metadata_row,
        dv=dv,
        threshold=threshold,
    )

    if original_plot is None or reconstructed_plot is None:
        return False

    original_values = np.asarray(original_plot.compressed(), dtype=float)
    if len(original_values) == 0:
        return False

    vmin = float(np.nanmin(original_values))
    vmax = float(np.nanmax(original_values))
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        return False
    if vmin == vmax:
        vmax = vmin * 10.0

    plot_extent = create_vdf_image_extent(extent=extent, vdflim=vdflim)
    norm = LogNorm(vmin=vmin, vmax=vmax)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(10, 4.8),
        constrained_layout=True,
    )
    images = []
    for ax, image, title in zip(
        axes,
        [original_plot, reconstructed_plot],
        ["Original", "Reconstructed"],
    ):
        im = ax.imshow(
            image,
            origin="lower",
            extent=plot_extent,
            norm=norm,
            cmap="nipy_spectral",
        )
        ax.grid(color="gray", axis="both")
        ax.set_xlim(-vdflim / 1000.0, vdflim / 1000.0)
        ax.set_ylim(-vdflim / 1000.0, vdflim / 1000.0)
        ax.set_xlabel("v_x")
        ax.set_ylabel("v_z")
        ax.set_title(title)
        images.append(im)

    fig.colorbar(images[-1], ax=axes, label="f(v)")
    fig.suptitle(create_reconstruction_title(metadata_row, split_name, sample_index))
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    return True


def prepare_vdf_slice_for_plot(vdf_slice, metadata_row, dv, threshold):
    """
    Apply the same VDF plot scaling and thresholding used by dataset plots.

    Parameters
    ----------
    vdf_slice : numpy.ndarray
        Physical VDF xz slice in plot orientation.
    metadata_row : dict
        Metadata for the sample, used in fallback messages.
    dv : float
        Velocity grid cell size in m/s.
    threshold : float
        VDF threshold before multiplying by ``dv``.

    Returns
    -------
    numpy.ma.MaskedArray or None
        Thresholded VDF slice multiplied by ``dv``.
    """

    vdf_plot_raw = np.asarray(vdf_slice, dtype=np.float32) * float(dv)
    vdf_plot = np.where(
        vdf_plot_raw < float(threshold) * float(dv),
        0.0,
        vdf_plot_raw,
    )
    vdf_plot = np.ma.masked_less_equal(vdf_plot, 0.0)

    if vdf_plot.count() == 0:
        print(
            "Using unthresholded VDF plot: "
            f"timestep={metadata_row.get('timestep', 'unknown')}, "
            f"cid={metadata_row.get('cid', 'unknown')}, "
            f"class={metadata_row.get('class_name', 'unknown')}"
        )
        vdf_plot = np.ma.masked_less_equal(vdf_plot_raw, 0.0)

    if vdf_plot.count() == 0:
        print(
            "Skipping empty VDF plot: "
            f"timestep={metadata_row.get('timestep', 'unknown')}, "
            f"cid={metadata_row.get('cid', 'unknown')}, "
            f"class={metadata_row.get('class_name', 'unknown')}"
        )
        return None

    return vdf_plot


def create_vdf_image_extent(extent=None, vdflim=2e6):
    """
    Create an imshow extent for xz velocity plots.

    Parameters
    ----------
    extent : array-like of float, optional
        Velocity mesh extent ``[vxmin, vymin, vzmin, vxmax, vymax, vzmax]``.
    vdflim : float, optional
        Fallback visible velocity limit in m/s.

    Returns
    -------
    list of float
        Image extent in km/s.
    """

    if extent is None:
        return [
            -vdflim / 1000.0,
            vdflim / 1000.0,
            -vdflim / 1000.0,
            vdflim / 1000.0,
        ]

    extent = np.asarray(extent, dtype=float)
    if len(extent) == 6:
        vxmin = extent[0]
        vzmin = extent[2]
        vxmax = extent[3]
        vzmax = extent[5]
    elif len(extent) == 4:
        vxmin, vxmax, vzmin, vzmax = extent
    else:
        raise ValueError("extent must have four or six values")

    return [
        vxmin / 1000.0,
        vxmax / 1000.0,
        vzmin / 1000.0,
        vzmax / 1000.0,
    ]


def create_reconstruction_title(metadata_row, split_name=None, sample_index=None):
    """
    Create a concise reconstruction plot title.

    Parameters
    ----------
    metadata_row : dict
        Metadata for the sample.
    split_name : str, optional
        Dataset split name.
    sample_index : int, optional
        Dataset sample index.

    Returns
    -------
    str
        Plot title.
    """

    title_parts = []
    if split_name is not None:
        title_parts.append(str(split_name))
    if sample_index is not None:
        title_parts.append(f"sample={int(sample_index)}")
    title_parts.extend(
        [
            f"timestep={metadata_row.get('timestep', 'unknown')}",
            f"cid={metadata_row.get('cid', 'unknown')}",
        ]
    )

    class_name = metadata_row.get("class_name")
    if class_name is not None:
        title_parts.append(f"class={class_name}")

    return ", ".join(title_parts)


def create_reconstruction_filename(metadata_row, split_name, sample_index):
    """
    Create a reconstruction plot filename.

    Parameters
    ----------
    metadata_row : dict
        Metadata for the sample.
    split_name : str
        Dataset split name.
    sample_index : int
        Dataset sample index.

    Returns
    -------
    str
        PNG filename.
    """

    timestep = metadata_row.get("timestep", "unknown")
    cid = metadata_row.get("cid", "unknown")
    class_name = metadata_row.get("class_name", "unknown")

    return (
        f"{split_name}_sample_{int(sample_index):06d}_"
        f"t{timestep}_cid{cid}_{class_name}.png"
    )


def resolve_vdf_plot_parameters(metadata_row, vdf_shape, vdflim=2e6, cache=None):
    """
    Read VDF plot parameters for one sample, with fallback values.

    Parameters
    ----------
    metadata_row : dict
        Sample metadata containing ``file_location`` when available.
    vdf_shape : tuple of int
        Dense VDF sample shape.
    vdflim : float, optional
        Fallback velocity limit in m/s.
    cache : dict, optional
        Cache keyed by ``(file_location, vdf_shape)``.

    Returns
    -------
    extent : numpy.ndarray
        Velocity extent ``[vxmin, vymin, vzmin, vxmax, vymax, vzmax]``.
    dv : float
        Velocity grid cell size in m/s.
    threshold : float
        VDF threshold before multiplying by ``dv``.
    """

    if cache is None:
        cache = {}

    file_location = metadata_row.get("file_location")
    cid = metadata_row.get("cid")
    if file_location:
        cache_key = (str(file_location), int(cid), tuple(vdf_shape))
        if cache_key in cache:
            return cache[cache_key]

        try:
            import analysator as pt

            from src.vdf_helpers import get_vdf_plot_parameters

            reader = pt.vlsvfile.VlsvReader(str(file_location))
            plot_parameters = get_vdf_plot_parameters(
                reader=reader,
                cid=int(cid),
                vdf_shape=vdf_shape,
            )
            cache[cache_key] = plot_parameters
            return plot_parameters
        except Exception:
            pass

    fallback_extent = np.asarray(
        [-vdflim, 0.0, -vdflim, vdflim, 0.0, vdflim],
        dtype=float,
    )

    return fallback_extent, 1.0, 0.0


def resolve_vdf_extent(metadata_row, vdf_shape, vdflim=2e6, cache=None):
    """
    Read velocity mesh extent for one sample, with a fallback extent.

    Parameters
    ----------
    metadata_row : dict
        Sample metadata containing ``file_location`` when available.
    vdf_shape : tuple of int
        Dense VDF sample shape.
    vdflim : float, optional
        Fallback velocity limit in m/s.
    cache : dict, optional
        Cache keyed by ``(file_location, cid, vdf_shape)``.

    Returns
    -------
    numpy.ndarray
        Velocity extent ``[vxmin, vymin, vzmin, vxmax, vymax, vzmax]``.
    """

    extent, _, _ = resolve_vdf_plot_parameters(
        metadata_row=metadata_row,
        vdf_shape=vdf_shape,
        vdflim=vdflim,
        cache=cache,
    )

    return extent


def plot_latent_pca(embeddings, metadata, output_path, class_column="class_name"):
    """
    Plot a two-dimensional PCA projection of latent embeddings.

    Parameters
    ----------
    embeddings : numpy.ndarray
        Latent vectors with shape ``(n_samples, latent_dim)``.
    metadata : pandas.DataFrame
        Metadata rows aligned with ``embeddings``.
    output_path : str or pathlib.Path
        Path where the figure is saved.
    class_column : str, optional
        Metadata column used for point colors.

    Returns
    -------
    bool
        Whether a plot was saved.
    """

    embeddings = np.asarray(embeddings, dtype=np.float64)
    if embeddings.ndim != 2 or embeddings.shape[0] < 2:
        return False

    centered = embeddings - embeddings.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    coords = centered @ vt[:2].T
    if coords.shape[1] == 1:
        coords = np.column_stack([coords[:, 0], np.zeros(len(coords))])

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 5))
    if class_column in metadata.columns:
        for class_name, rows in metadata.groupby(class_column, sort=True):
            row_indices = rows.index.to_numpy(dtype=int)
            ax.scatter(
                coords[row_indices, 0],
                coords[row_indices, 1],
                s=12,
                alpha=0.75,
                label=str(class_name),
            )
        ax.legend(loc="best", fontsize=8)
    else:
        ax.scatter(coords[:, 0], coords[:, 1], s=12, alpha=0.75)

    ax.set_xlabel("latent PC1")
    ax.set_ylabel("latent PC2")
    ax.set_title("Autoencoder latent space")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    return True
