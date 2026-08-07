"""Stage 7: animate the frame directories produced by dataset Stage 6.

This final optional dataset stage follows static frame rendering. It receives
the completed dataset directory and animation settings, then creates one
spatial colormap evolution video and one time-ordered video for each cell ID
that has more than one VDF frame. Keeping encoding separate from Stage 6
prevents ffmpeg ownership from entering scientific plotting code.
"""

from dataclasses import dataclass
from pathlib import Path

from src.plotting.frame_animation import create_frame_animation


@dataclass(frozen=True)
class DatasetAnimationResult:
    """Describe animations written by the final dataset workflow stage.

    Instances contain output identities only; they own no ffmpeg process,
    frame arrays, or open files. Stage 0 may use this record for concise
    reporting after all configured postprocessing has finished.

    Attributes
    ----------
    colormap_animation : pathlib.Path or None
        Spatial evolution video, or ``None`` when colormap animation is
        disabled.
    vdf_animations : tuple of pathlib.Path
        Cell-ID VDF videos in deterministic directory-name order.
    """

    colormap_animation: Path | None
    vdf_animations: tuple


def animate_dataset_frames(
    dataset_dir,
    *,
    animate_colormaps,
    animate_vdfs,
    animation_fps,
    animation_format,
):
    """Create configured colormap and per-cell-ID VDF animations.

    This final dataset stage consumes only the PNG directories written by
    Stage 6. Colormap frames are ordered by their contiguous frame indices.
    VDF directories and their frames are sorted independently, preserving
    each cell ID as the animation identity and its timesteps as the temporal
    sequence.

    Parameters
    ----------
    dataset_dir : str or pathlib.Path
        Completed dataset containing Stage 6 output below ``plots/``.
    animate_colormaps : bool
        Whether to encode ``plots/colormaps`` as one spatial animation.
    animate_vdfs : bool
        Whether to encode one animation for each ``plots/vdfs/cid_*``
        directory containing more than one frame.
    animation_fps : int or float
        Playback frame rate in frames per second.
    animation_format : str
        Output suffix, normally ``mp4`` for the configured H.264 workflow.

    Returns
    -------
    DatasetAnimationResult
        Deterministically ordered animation paths that were passed to
        ffmpeg.

    Notes
    -----
    Outputs are written below ``animations/`` without regrouping cell IDs by
    physical class. A one-timestep cell retains its static PNG but has no
    one-frame video.
    """

    dataset_dir = Path(dataset_dir)
    animations_dir = dataset_dir / "animations"
    colormap_animation = None
    vdf_animations = []

    if animate_colormaps:
        colormap_frames = sorted(
            (dataset_dir / "plots" / "colormaps").glob("frame_*.png")
        )
        colormap_animation = create_frame_animation(
            colormap_frames,
            animations_dir / f"colormap_evolution.{animation_format}",
            animation_fps,
        )

    if animate_vdfs:
        vdf_animation_dir = animations_dir / "vdfs"
        for cid_directory in sorted(
            (dataset_dir / "plots" / "vdfs").glob("cid_*")
        ):
            frame_paths = sorted(cid_directory.glob("frame_*.png"))
            if len(frame_paths) > 1:
                vdf_animations.append(
                    create_frame_animation(
                        frame_paths,
                        vdf_animation_dir
                        / f"{cid_directory.name}.{animation_format}",
                        animation_fps,
                    )
                )

    return DatasetAnimationResult(
        colormap_animation=colormap_animation,
        vdf_animations=tuple(vdf_animations),
    )
