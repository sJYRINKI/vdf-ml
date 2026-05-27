import subprocess
from pathlib import Path

def build_ffmpeg_command(
        frames_dir,
        frames_pattern,
        output_path,
        start_number=0,
        input_framerate=5,
        output_framerate=5,
        qp=18,
        preset="slow"
):
    """
    Builds an ffmpeg command.

    Parameters
    ----------
    frames_dir : str
        Directory containing PNG images.
    frame_pattern : str
        ffmpeg input pattern.
    output_path : str
        Output video path.
    start_number : int, optional
        First frame number.
    input_framerate : int or float, optional
        Framerate assigned to input image sequency.
    output_framerate : int or float, optional
        Framerate of output video.
    qp : int, optional
        H.263 quantization parameter.
    preset : str, optional
        H.264 encoding preset.
    """

    frames_dir = Path(frames_dir)
    output_path = Path(output_path)

    input_pattern = frames_dir / frames_pattern

    command = [
        "ffmpeg",
        "-y",
        "-f",
        "image2",
        "-start_number",
        str(start_number),
        "-framerate",
        str(input_framerate),
        "-i",
        str(input_pattern),
        "-vf",
        f"fps={output_framerate}, scale=trunc(iw/2)*2:trunc(ih/2)*2",
        "-c:v",
        "libx264",
        "-preset",
        preset,
        "-profile:v",
        "baseline",
        "-qp",
        str(qp),
        "-pix_fmt",
        "yuv420p",
        str(output_path),
    ]

    return command

def create_animation(
        frames_dir,
        frame_pattern,
        output_path,
        start_number=0,
        input_framerate=5,
        output_framerate=5,
        qp=18,
        preset="slow"
):
    """
    Creates an animation from PNG frames.

    Parameters
    ----------
    frames_dir : str
        Directory containing PNG images.
    frame_pattern : str
        ffmpeg input pattern.
    output_path : str
        Output video path.
    start_number : int, optional
        First frame number.
    input_framerate : int or float, optional
        Framerate assigned to input image sequency.
    output_framerate : int or float, optional
        Framerate of output video.
    qp : int, optional
        H.263 quantization parameter.
    preset : str, optional
        H.264 encoding preset.
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    command = build_ffmpeg_command(
        frames_dir=frames_dir,
        frames_pattern=frame_pattern,
        output_path=output_path,
        start_number=start_number,
        input_framerate=input_framerate,
        output_framerate=output_framerate,
        qp=qp,
        preset=preset,
    )

    print("Running ffmpeg command:")
    print(" ".join(command))

    subprocess.run(command, check=True)