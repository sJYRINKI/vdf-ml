"""Render saved physical-VDF Hermite coefficients as Stage 6 frames.

This module owns the signed-coefficient visualization used after dataset
extraction.  It reads the sample-aligned ``X_hermite.npy`` memory map once,
groups rows by spatial cell ID, and plots ``coefficients[:, 0, :]`` in the
same stable CID and timestep sequence as the raw-VDF frame workflow.  The
saved cube is dimensional and remains unchanged; plotting applies only a
symmetric signed color range for visual comparison.

Unrotated coefficients are ordered ``(n_x, n_y, n_z)`` and display the
``n_y = 0`` plane.  Coefficients produced after optional physical
velocity-space rotation are ordered
``(n_parallel, n_perp1, n_perp2)`` and display ``n_perp1 = 0``.  Both use a
white square axes, ``RdBu_r``, and a coefficient colorbar without logarithmic
scaling or per-sample coefficient normalization.
"""

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)

from matplotlib import pyplot as plt
import numpy as np


@dataclass(frozen=True)
class DatasetHermiteFrameResult:
    """Describe one saved Stage 6 Hermite coefficient frame.

    Attributes
    ----------
    output_file : pathlib.Path
        PNG path below the numeric cell-ID directory.
    sample_index : int
        Stable sample identity aligned with ``metadata.csv``.
    cid : int
        Spatial cell ID that owns the frame sequence.
    timestep : int
        Simulation timestep used to order the CID sequence.
    class_name : str
        Current physical class displayed in the frame title.
    order : int
        Number of saved Hermite modes along each coefficient axis.
    rotated : bool
        Whether extraction rotated the physical VDF before projection.
    title : str
        Sample identity, coordinate, order, and rotation annotation.
    """

    output_file: Path
    sample_index: int
    cid: int
    timestep: int
    class_name: str
    order: int
    rotated: bool
    title: str


def plot_dataset_hermite_frames(
    dataset_dir,
    metadata,
    output_dir,
    *,
    rotated,
    dpi=150,
):
    """Render time-ordered Hermite coefficient frames for each spatial cell.

    Stage 6 opens the saved ``X_hermite.npy`` memory map once, groups
    sample-aligned rows by numeric spatial cell ID, and writes one signed
    coefficient image for each timestep.  The displayed plane fixes the
    second Hermite order at zero, matching the physical-VDF transform's
    ``coefficients[:, 0, :]`` visualization.

    Unrotated coefficients use axis order ``(n_x, n_y, n_z)``.  When optional
    rotation was enabled during extraction, the corresponding order is
    ``(n_parallel, n_perp1, n_perp2)``.  Each frame uses its unscaled
    coefficient amplitudes and a symmetric color range around zero so
    positive and negative coefficients remain directly comparable.

    Parameters
    ----------
    dataset_dir : str or pathlib.Path
        Completed dataset containing the sample-aligned
        ``X_hermite.npy`` coefficient array.
    metadata : pandas.DataFrame
        Metadata loaded once by Stage 6 with CID, timestep, class,
        sample-index, and VDF coordinates in Earth radii.
    output_dir : str or pathlib.Path
        Root directory receiving one frame sequence per numeric CID.
    rotated : bool
        Whether the saved coefficients were calculated after optional
        physical velocity-space rotation.
    dpi : int, optional
        Fixed PNG resolution in dots per inch.

    Returns
    -------
    tuple of DatasetHermiteFrameResult
        Frame records in stable numeric CID, timestep, and sample-index
        order.

    Notes
    -----
    Plotting reads the saved coefficients directly.  It does not reopen VLSV
    files, recompute the transform, normalize coefficient cubes, or alter
    the dataset.  Filenames use the same CID, frame-index, and timestep
    padding as Stage 6 raw-VDF frames.
    """

    dataset_dir = Path(dataset_dir)
    output_dir = Path(output_dir)
    ordered_rows = metadata.copy()
    ordered_rows["_array_row"] = np.arange(len(metadata), dtype=np.int64)
    ordered_rows = ordered_rows.sort_values(
        ["cid", "timestep", "sample_index"],
        kind="stable",
    )
    cid_width = max(
        9,
        max(len(str(int(value))) for value in metadata["cid"]),
    )
    coefficients = np.load(
        dataset_dir / "X_hermite.npy",
        mmap_mode="r",
        allow_pickle=False,
    )
    results = []
    try:
        for cid, rows in ordered_rows.groupby("cid", sort=False):
            cid = int(cid)
            cid_dir = output_dir / f"cid_{cid:0{cid_width}d}"
            cid_dir.mkdir(parents=True, exist_ok=True)
            for frame_index, (_, row) in enumerate(rows.iterrows()):
                timestep = int(row["timestep"])
                output_file = cid_dir / (
                    f"frame_{frame_index:06d}_"
                    f"timestep_{timestep:07d}.png"
                )
                coefficient_cube = coefficients[int(row["_array_row"])]
                figure, axis = plt.subplots(figsize=(7.0, 6.0))
                try:
                    title = render_hermite_coefficient_frame(
                        figure,
                        axis,
                        coefficient_cube,
                        row,
                        rotated=bool(rotated),
                    )
                    figure.tight_layout()
                    figure.savefig(output_file, dpi=int(dpi))
                finally:
                    plt.close(figure)
                results.append(
                    DatasetHermiteFrameResult(
                        output_file=output_file,
                        sample_index=int(row["sample_index"]),
                        cid=cid,
                        timestep=timestep,
                        class_name=str(row["class_name"]),
                        order=int(coefficient_cube.shape[0]),
                        rotated=bool(rotated),
                        title=title,
                    )
                )
    finally:
        coefficients._mmap.close()
    return tuple(results)


def render_hermite_coefficient_frame(
    figure,
    axis,
    coefficients,
    metadata_row,
    *,
    rotated,
):
    """Draw one signed zero-second-order Hermite coefficient plane.

    The input cube contains dimensional physical-VDF Hermite coefficients.
    This renderer fixes coefficient axis 1 at order zero and transposes the
    resulting ``coefficients[:, 0, :]`` plane so the first coefficient axis
    is horizontal and the third is vertical.  A symmetric linear scale
    preserves coefficient sign and amplitude without modifying the saved
    representation.

    Parameters
    ----------
    figure : matplotlib.figure.Figure
        Figure that receives the signed-coefficient colorbar.
    axis : matplotlib.axes.Axes
        White square axes receiving the coefficient image.
    coefficients : numpy.ndarray
        Saved coefficient cube with shape ``(order, order, order)``.  Its
        axes are ``(n_x, n_y, n_z)`` when unrotated or
        ``(n_parallel, n_perp1, n_perp2)`` when rotated.
    metadata_row : pandas.Series
        Saved CID, timestep, class, sample index, and VDF coordinate in
        Earth radii for the same array row.
    rotated : bool
        Select rotated coefficient labels and title wording.

    Returns
    -------
    str
        Two-line title containing the sample identity, coordinate, saved
        Hermite order, and rotation setting.

    Notes
    -----
    ``RdBu_r`` is applied directly to dimensional signed coefficients with
    limits ``[-max(abs(slice)), max(abs(slice))]``.  No logarithm,
    threshold, absolute-value projection, or coefficient normalization is
    applied.
    """

    coefficient_slice = coefficients[:, 0, :]
    coefficient_limit = float(np.max(np.abs(coefficient_slice)))
    axis.set_facecolor("white")
    image = axis.imshow(
        coefficient_slice.T,
        origin="lower",
        cmap="RdBu_r",
        vmin=-coefficient_limit,
        vmax=coefficient_limit,
        interpolation="nearest",
        aspect="equal",
    )
    if rotated:
        axis.set_xlabel(r"$n_{\parallel}$")
        axis.set_ylabel(r"$n_{\perp 2}$")
        fixed_axis = r"$n_{\perp 1}=0$"
        transform_name = "rotated"
    else:
        axis.set_xlabel(r"$n_x$")
        axis.set_ylabel(r"$n_z$")
        fixed_axis = r"$n_y=0$"
        transform_name = "unrotated"
    axis.set_box_aspect(1.0)
    order = int(coefficients.shape[0])
    title = (
        f"timestep={int(metadata_row['timestep'])}, "
        f"cid={int(metadata_row['cid'])}, "
        f"class={metadata_row['class_name']}, "
        f"sample={int(metadata_row['sample_index'])}\n"
        f"coord=({float(metadata_row['vdf_x_re']):.3g}, "
        f"{float(metadata_row['vdf_y_re']):.3g}, "
        f"{float(metadata_row['vdf_z_re']):.3g}) R_E, "
        f"Hermite order={order}, {transform_name}, {fixed_axis}"
    )
    axis.set_title(title, fontsize=10)
    figure.colorbar(image, ax=axis, label="Hermite coefficient")
    return title
