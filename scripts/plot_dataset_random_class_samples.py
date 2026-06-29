#python scripts/plot_dataset_random_class_samples.py --config configs/plot_dataset_random_class_samples.yaml --timestep 3408_100
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.config import load_config
from src.dataset_io import load_dataset
from src.dataset_plot import plot_random_class_samples
from src.timesteps import create_timestep_path


def main(config_path, timestep):
    config = load_config(config_path)

    dataset_dir = create_timestep_path(
        path_template=config["dataset_dir"],
        timestep=timestep,
    )

    output_dir = create_timestep_path(
        path_template=config["output_dir"],
        timestep=timestep,
    )

    plot_config = config["plot"]
    output_path = output_dir / plot_config.get(
        "output_file",
        "random_class_samples_xz.png",
    )

    X, y, metadata = load_dataset(dataset_dir, mmap=True)

    selected_indices = plot_random_class_samples(
        X=X,
        y=y,
        metadata=metadata,
        output_path=output_path,
        vdflim=float(plot_config.get("vdflim", 2e6)),
        random_state=plot_config.get("random_state"),
        n_columns=int(plot_config.get("n_columns", 2)),
    )

    print(f"Dataset directory: {dataset_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Saved plot to: {output_path}")
    print(f"Selected sample indices: {selected_indices}")


if __name__=="__main__":
    parser = argparse.ArgumentParser(
        description="Plot one random xz VDF sample from each class."
    )

    parser.add_argument(
        "--config",
        required=True,
        help="Path to plotting YAML config file.",
    )

    parser.add_argument(
        "--timestep",
        required=True,
        help="Dataset timestep identifier.",
    )

    args = parser.parse_args()

    main(
        config_path=args.config,
        timestep=args.timestep,
    )
