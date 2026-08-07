"""Encode a sorted sequence of PNG frames as one H.264 animation.

This plotting owner is separate from dataset Stage 7 so the same direct
ffmpeg invocation can encode either spatial colormap frames or the frames
for one cell ID. It receives already rendered PNG paths, writes a temporary
concat manifest outside the dataset, and publishes one MP4-compatible video.
"""

from pathlib import Path
import subprocess
import tempfile


def create_frame_animation(frame_paths, output_path, fps):
    """Encode ordered PNG frames as one H.264 video.

    Dataset Stage 7 calls this function after Stage 6 has rendered a
    deterministic PNG sequence. The temporary concat manifest makes every
    input path explicit, so a cell-ID animation cannot consume frames from a
    neighboring directory. ffmpeg scales odd image dimensions down by one
    pixel where needed and writes broadly compatible YUV 4:2:0 video.

    Parameters
    ----------
    frame_paths : iterable of pathlib.Path
        PNG frames in their intended temporal order. Dataset frame filenames
        encode a contiguous frame index and the actual simulation timestep.
    output_path : str or pathlib.Path
        Animation file written by ffmpeg. Its parent directory is created.
    fps : int or float
        Playback frame rate in frames per second.

    Returns
    -------
    pathlib.Path
        Path passed to ffmpeg for the completed animation.

    Notes
    -----
    The function writes no timing metadata based on simulation timestep
    spacing; each rendered frame title already records its actual timestep.
    """

    frame_paths = tuple(Path(path).resolve() for path in frame_paths)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(
        prefix="vdf-ml-animation-"
    ) as temporary_directory:
        manifest_path = Path(temporary_directory) / "frames.txt"
        manifest_path.write_text(
            "".join(_concat_manifest_entry(path) for path in frame_paths),
            encoding="utf-8",
        )
        command = [
            "ffmpeg",
            "-y",
            "-r",
            str(fps),
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(manifest_path),
            "-vf",
            (
                f"fps={fps},"
                "scale=trunc(iw/2)*2:trunc(ih/2)*2"
            ),
            "-c:v",
            "libx264",
            "-preset",
            "slow",
            "-profile:v",
            "baseline",
            "-qp",
            "18",
            "-pix_fmt",
            "yuv420p",
            str(output_path),
        ]
        subprocess.run(command, check=True)

    return output_path


def _concat_manifest_entry(path):
    """Quote one absolute frame path for the ffmpeg concat demuxer."""

    escaped_path = path.as_posix().replace("'", "'\\''")
    return f"file '{escaped_path}'\n"
