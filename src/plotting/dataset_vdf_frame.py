"""Render Stage 6 cell-ID-organized VDF evolution frames.

This focused Stage 6 owner reads each saved physical distribution in
``(vx, vy, vz)`` order and draws one square ``vx-vz`` panel through that
sample's own three-dimensional peak ``vy`` index. Same-cell historical
threshold masking, logarithmic phase-space density, kilometre-per-second
axes, and a fixed canvas make displaced populations readable without the
replaced forced-black background. Plane extraction and drawing are shared
with the standalone command; Stage 6 deliberately requests only ``xz``.
"""

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

from src.data.load_velocity_grid import load_velocity_grid
from src.plotting.vdf_slices import (
    DEFAULT_VDF_VELOCITY_LIMIT_MS,
    _close_memmap,
    draw_vdf_plane,
    extract_peak_vdf_planes,
    get_vdf_plot_velocity_cell_size_ms,
    load_vdf_plotting_thresholds,
    prepare_physical_vdf_plane,
)


@dataclass(frozen=True)
class DatasetVdfFrameResult:
    """Describe one saved Stage 6 VDF evolution frame.

    Attributes
    ----------
    output_file : pathlib.Path
        Saved PNG path below the cell-ID directory.
    sample_index : int
        Stable dataset sample identity shown in the title.
    cid : int
        Spatial cell ID that owns the frame directory.
    timestep : int
        Simulation timestep used to order frames for one cell ID.
    class_name : str
        Current physical class displayed in the title.
    peak_index : tuple of int
        Raw VDF maximum in ``[vx, vy, vz]`` index order.
    title : str
        One-line timestep, cell-ID, class, and sample annotation.
    """

    output_file: Path
    sample_index: int
    cid: int
    timestep: int
    class_name: str
    peak_index: tuple
    title: str


def plot_dataset_vdf_frames(
    dataset_dir,
    output_dir,
    *,
    velocity_limit_ms=None,
    dpi=150,
    reader_factory=None,
):
    """Render one logarithmic ``vx-vz`` frame for every saved sample.

    Stage 6 opens the aligned metadata, velocity grid, and raw ``X.npy``
    memory map once. Rows are grouped by numeric cell ID and sorted by
    timestep and sample index before rendering, which preserves the existing
    deterministic directory and filename sequence consumed by Stage 7.

    Parameters
    ----------
    dataset_dir : str or pathlib.Path
        Current dataset containing ``metadata.csv``, ``X.npy``, and
        ``velocity_grid.npz``.
    output_dir : str or pathlib.Path
        Root directory receiving one ``cid_<zero-padded ID>`` directory per
        spatial cell.
    velocity_limit_ms : float, optional
        Symmetric visible velocity limit in metres per second. The historical
        Stage 6 limit of ``2.0e6`` m/s is used when omitted.
    dpi : int, optional
        Fixed PNG resolution in dots per inch.
    reader_factory : callable, optional
        VLSV reader constructor used to load same-cell source sparsity
        thresholds.
        Production uses Analysator; focused tests may supply its interface.

    Returns
    -------
    tuple of DatasetVdfFrameResult
        Frames in numeric CID, timestep, and sample-index order.

    Notes
    -----
    Frame indices restart at zero for every CID. Names follow
    ``frame_<six digits>_timestep_<seven digits>.png`` and fixed figure
    dimensions keep one cell's animation visually stable.
    """

    dataset_dir = Path(dataset_dir)
    output_dir = Path(output_dir)
    metadata = pd.read_csv(dataset_dir / "metadata.csv")
    metadata["_array_row"] = np.arange(len(metadata), dtype=np.int64)
    ordered_rows = metadata.sort_values(
        ["cid", "timestep", "sample_index"],
        kind="stable",
    ).copy()
    ordered_rows["_plotting_threshold"] = load_vdf_plotting_thresholds(
        ordered_rows,
        reader_factory=reader_factory,
    )
    velocity_grid = load_velocity_grid(dataset_dir)
    extent_mps = tuple(
        float(value) for value in velocity_grid["extent_mps"]
    )
    velocity_cell_size_ms = get_vdf_plot_velocity_cell_size_ms(
        velocity_grid
    )
    cid_width = max(
        9,
        max(len(str(int(value))) for value in metadata["cid"]),
    )
    velocity_limit_ms = (
        DEFAULT_VDF_VELOCITY_LIMIT_MS
        if velocity_limit_ms is None
        else float(velocity_limit_ms)
    )
    X = np.load(
        dataset_dir / "X.npy",
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
                plane = extract_peak_vdf_planes(
                    X[int(row["_array_row"])],
                    extent_mps,
                    ("xz",),
                )[0]
                figure, axis = plt.subplots(
                    figsize=(7.0, 6.0),
                )
                try:
                    title = render_dataset_vdf_frame(
                        figure,
                        axis,
                        plane,
                        row,
                        plotting_threshold=float(
                            row["_plotting_threshold"]
                        ),
                        velocity_cell_size_ms=velocity_cell_size_ms,
                        velocity_limit_ms=velocity_limit_ms,
                    )
                    figure.tight_layout()
                    figure.savefig(output_file, dpi=int(dpi))
                finally:
                    plt.close(figure)
                results.append(
                    DatasetVdfFrameResult(
                        output_file=output_file,
                        sample_index=int(row["sample_index"]),
                        cid=cid,
                        timestep=timestep,
                        class_name=str(row["class_name"]),
                        peak_index=plane.peak_index,
                        title=title,
                    )
                )
    finally:
        _close_memmap(X)
    return tuple(results)


def render_dataset_vdf_frame(
    figure,
    axis,
    plane,
    metadata_row,
    *,
    plotting_threshold,
    velocity_cell_size_ms,
    velocity_limit_ms,
):
    """Render one logarithmic ``vx-vz`` frame for a saved dataset sample.

    The frame is consumed by the per-cell-ID Stage 7 animation. It fixes the
    ``vy`` coordinate at the three-dimensional VDF maximum so shifted or fast
    populations remain visible; using the velocity-grid centre could produce
    an empty plane when the distribution is displaced from zero. The raw
    phase-space density remains unchanged.

    Parameters
    ----------
    figure : matplotlib.figure.Figure
        Figure that receives the vertical phase-space-density colorbar.
    axis : matplotlib.axes.Axes
        Square axes that receive the ``vx-vz`` image and Cartesian grid.
    plane : PhysicalVdfPlane
        Peak-crossing ``xz`` plane in ``(vx, vz)`` order and ``s^3 m^-6``
        units, extracted by the shared standalone/Stage 6 plane owner.
    metadata_row : pandas.Series
        Saved timestep, cell ID, sample identity, and physical class.
    plotting_threshold : float
        Same-cell source sparsity threshold used by historical display
        preparation, in ``s^3 m^-6``.
    velocity_cell_size_ms : float
        Historical regular ``vy`` cell width in metres per second.
    velocity_limit_ms : float
        Symmetric visible velocity limit in metres per second.

    Returns
    -------
    str
        One-line frame title containing timestep, cell ID, class, and sample.

    Notes
    -----
    Velocity coordinates are divided by 1000 for kilometre-per-second axes.
    Historical threshold masking is transparent over the white axes; no black
    bad color is forced. This rendering changes only visualization; saved VDFs
    and model inputs remain complete three-dimensional arrays.
    """

    prepared = prepare_physical_vdf_plane(
        plane.values,
        plotting_threshold,
        velocity_cell_size_ms,
    )
    image = draw_vdf_plane(
        axis,
        plane,
        prepared=prepared,
        velocity_limit_ms=velocity_limit_ms,
    )
    title = (
        f"timestep={int(metadata_row['timestep'])}, "
        f"cid={int(metadata_row['cid'])}, "
        f"class={metadata_row['class_name']}, "
        f"sample={int(metadata_row['sample_index'])}"
    )
    axis.set_title(title, fontsize=10)
    figure.colorbar(image, ax=axis, label="f(v)")
    return title
