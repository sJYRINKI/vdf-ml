"""Stage 6: render optional spatial, raw-VDF, and Hermite dataset frames.

This stage follows the completed Stage 5 dataset rename and precedes
animation. It receives the final dataset directory plus accepted point
records retained by Stage 3, then delegates spatial backgrounds, exact
selection overlays, role-colored markers, and shared peak-centred velocity
planes to the current plotting package.

Colormap PNGs are ordered by simulation timestep. Raw VDF PNGs use each
source cell's resolved sparsity-threshold mask. Optional Hermite PNGs read
the saved dimensional coefficient cube directly and display its signed
zero-second-order plane. Both sample-level frame families are organized by
spatial cell ID and timestep; only the raw frames are consumed by Stage 7.
"""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.plotting.dataset_colormap import plot_dataset_colormap
from src.plotting.dataset_vdf_frame import (
    plot_dataset_vdf_frames,
)
from src.plotting.hermite_coefficients import (
    plot_dataset_hermite_frames,
)


_STAGE6_SPATIAL_BOX_RE = (-30.0, 30.0, -13.0, 13.0)
_STAGE6_VELOCITY_LIMIT_MS = 1.5e6


@dataclass(frozen=True)
class DatasetFrameResult:
    """Describe the optional frame files created for one dataset.

    Attributes
    ----------
    colormap_frames : tuple
        Per-timestep colormap results in ascending timestep order.
    vdf_frames : tuple
        Per-sample VDF frame results ordered by CID, timestep, and saved
        sample index.
    hermite_frames : tuple
        Per-sample signed Hermite coefficient frames ordered by CID,
        timestep, and saved sample index.
    """

    colormap_frames: tuple
    vdf_frames: tuple
    hermite_frames: tuple


def plot_dataset_frames(
    dataset_dir,
    *,
    config,
    plot_data_by_timestep,
    plot_colormaps,
    plot_vdfs,
    plot_hermites=False,
):
    """Create configured spatial, raw-VDF, and Hermite frame sequences.

    Stage 0 calls this function only after the core dataset has been
    completed. Colormap rendering reuses one loaded metadata table and the
    accepted Stage 3 point geometry. Its background is bulk x velocity with
    fixed symmetric limits, black magnetic streamlines, and yellow/blue/red
    candidate/source/selected roles. VDF rendering opens the raw memory map
    once and draws one historical-threshold logarithmic ``vx-vz`` plane at
    each sample's own peak ``vy`` index without forcing the masked background
    to black. Hermite rendering reads ``X_hermite.npy`` once and displays
    ``coefficients[:, 0, :]`` with signed symmetric limits, coefficient-axis
    labels, and no transform recomputation or coefficient normalization.

    Parameters
    ----------
    dataset_dir : str or pathlib.Path
        Completed dataset containing current core files.
    config : dict
        Extraction configuration containing optional plotting settings.
    plot_data_by_timestep : mapping
        Stage 3 point-record handoffs keyed by simulation timestep.
    plot_colormaps : bool
        Whether to write ``plots/colormaps`` PNG frames.
    plot_vdfs : bool
        Whether to write cell-ID-organized ``plots/vdfs`` PNG frames.
    plot_hermites : bool, optional
        Whether to write cell-ID-organized ``plots/hermite`` PNG frames.
        Stage 0 passes true only when Hermite extraction and its plotting
        toggle are both enabled.

    Returns
    -------
    DatasetFrameResult
        Deterministically ordered spatial, raw-VDF, and Hermite frame
        records.

    Notes
    -----
    Frame indices are contiguous within the colormap sequence and within
    each CID directory. Timestep fields are zero padded independently so
    filenames remain numerically sortable when simulation steps are not
    contiguous. Hermite frames are derived visual output and are not
    animated by Stage 7.
    """

    dataset_dir = Path(dataset_dir)
    plotting = config.get("plotting", {})
    colormap_frames = ()
    vdf_frames = ()
    hermite_frames = ()
    metadata = None
    if plot_colormaps or plot_hermites:
        metadata = pd.read_csv(dataset_dir / "metadata.csv")
    if plot_colormaps:
        timesteps = tuple(
            sorted(
                set(metadata["timestep"].astype(int).unique())
                | {
                    int(timestep)
                    for timestep in plot_data_by_timestep
                }
            )
        )
        timestep_width = _integer_width(timesteps, minimum=7)
        colormap_dir = dataset_dir / "plots" / "colormaps"
        spatial_slice_y_re = plotting.get("spatial_slice_y_re")
        spatial_box_re = plotting.get(
            "spatial_box_re",
            _STAGE6_SPATIAL_BOX_RE,
        )
        spatial_velocity_limit_ms = float(
            plotting.get(
                "spatial_velocity_limit_ms",
                _STAGE6_VELOCITY_LIMIT_MS,
            )
        )
        results = []
        for frame_index, timestep in enumerate(timesteps):
            plot_data = plot_data_by_timestep.get(int(timestep))
            x_point_records = (
                plot_data.x_point_records
                if plot_data is not None
                else ()
            )
            o_point_records = (
                plot_data.o_point_records
                if plot_data is not None
                else ()
            )
            output_file = colormap_dir / (
                f"frame_{frame_index:06d}_"
                f"timestep_{int(timestep):0{timestep_width}d}.png"
            )
            results.append(
                plot_dataset_colormap(
                    dataset_dir,
                    output_file,
                    extraction_config=config,
                    timestep=int(timestep),
                    spatial_box_re=spatial_box_re,
                    velocity_limit_ms=spatial_velocity_limit_ms,
                    spatial_slice_y_re=spatial_slice_y_re,
                    show_all_vdf_cells=plotting.get(
                        "show_all_vdf_cells",
                        True,
                    ),
                    show_labeled_cells=plotting.get(
                        "show_labeled_cells",
                        True,
                    ),
                    show_class_legend=plotting.get(
                        "show_class_legend",
                        True,
                    ),
                    show_magnetic_streamlines=plotting.get(
                        "show_magnetic_streamlines",
                        True,
                    ),
                    dpi=plotting.get("dpi", 150),
                    metadata=metadata,
                    x_point_records=x_point_records,
                    o_point_records=o_point_records,
                )
            )
        colormap_frames = tuple(results)
    if plot_vdfs:
        vdf_frames = plot_dataset_vdf_frames(
            dataset_dir,
            dataset_dir / "plots" / "vdfs",
            velocity_limit_ms=plotting.get("velocity_limit_ms"),
            dpi=plotting.get("dpi", 150),
        )
    if plot_hermites:
        hermite_config = config.get("representations", {}).get(
            "hermite",
            {},
        )
        hermite_frames = plot_dataset_hermite_frames(
            dataset_dir,
            metadata,
            dataset_dir / "plots" / "hermite",
            rotated=bool(hermite_config.get("rotate", False)),
            dpi=plotting.get("dpi", 150),
        )
    return DatasetFrameResult(
        colormap_frames=colormap_frames,
        vdf_frames=vdf_frames,
        hermite_frames=hermite_frames,
    )


def _integer_width(values, *, minimum):
    """Return a stable zero-padding width for nonnegative integer values.

    Parameters
    ----------
    values : sequence of int
        Timesteps or spatial cell IDs included in one frame family.
    minimum : int
        Smallest filename field width.

    Returns
    -------
    int
        Maximum of the requested minimum and represented integer lengths.
    """

    return max(
        int(minimum),
        max(len(str(int(value))) for value in values),
    )
