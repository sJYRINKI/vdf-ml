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

    class_dirs = sorted(
        path for path in frames_dir.iterdir()
        if path.is_dir()
    )

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