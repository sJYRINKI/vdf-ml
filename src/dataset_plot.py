from pathlib import Path
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

def plot_vdf_xz_slice(
        vdf,
        y_label,
        metadata_row,
        extent,
        output_path,
        dv=30000.0,
        threshold=8.301134972025815e-16,
        vdflim=2e6
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
    dv : float, optional
        Velocity grid cell size in m/s.
    threshold : float, optional
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

    ax1.set_title(
        f"sample {metadata_row["sample_index"]}, "
        f"label={int(y_label)}, "
        f"class={metadata_row["class_name"]}, "
        f"timestep={metadata_row["timestep"]}"
        
    )

    fig.colorbar(im, ax=ax1, label="f(v)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)