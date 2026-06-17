#python scripts/animate_dataset_by_class.py --config configs/animate_dataset.yaml --timestep 3408_100
import argparse
import sys
from pathlib import Path

PRPJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PRPJECT_ROOT))

from src.config import load_config
from src.timesteps import create_timestep_path
from src.animation import create_animation

def main(config_path, timestep):
    config = load_config(config_path)

    frames_dir = create_timestep_path(
        path_template=config["frames_dir"],
        timestep=timestep,
    )

    output_dir = create_timestep_path(
        path_template=config["output_dir"],
        timestep=timestep
    )

    animation_config = config["animation"]

    frame_pattern = animation_config.get("frame_pattern", "sample_%04d_xz.png")
    output_file_template = animation_config.get(
        "output_file_template",
        "{class_name}_xz.mp4"
    )

    ignore_dirs = set(animation_config.get("ignore_dirs", []))

    class_dirs = []

    for path in sorted(frames_dir.iterdir()):
        if not path.is_dir():
            continue

        if path.name in ignore_dirs:
            continue

        class_dirs.append(path)

    for class_dir in class_dirs:
        class_name = class_dir.name

        output_file = output_file_template.format(class_name=class_name)
        output_path = output_dir / output_file

        create_animation(
            frames_dir=class_dir,
            frame_pattern=frame_pattern,
            output_path=output_path,
        )

        print(f"Saved animation to: {output_path}")

    colormap_config = config.get("colormap_animation", {})
    colormap_frames_dir = frames_dir / colormap_config.get(
        "frames_dir",
        "colormaps",
    )
    colormap_output_path = output_dir / colormap_config.get(
        "output_file",
        "colormaps.mp4",
    )

    create_animation(
        frames_dir=colormap_frames_dir,
        frame_pattern=colormap_config.get(
            "frame_pattern",
            "colormap_%04d.png",
        ),
        output_path=colormap_output_path,
    )

    print(f"Saved colormap animation to: {colormap_output_path}")

if __name__=="__main__":
    parser = argparse.ArgumentParser(
        description="Create an animation from saved VDF frames."
    )

    parser.add_argument(
        "--config",
        required=True,
        help="Path to animation YAML config file."
    )

    parser.add_argument(
        "--timestep",
        required=True,
        help="Dataset timestep identifier."
    )

    args = parser.parse_args()

    main(
        config_path=args.config,
        timestep=args.timestep
    )