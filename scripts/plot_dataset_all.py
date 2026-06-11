#python scripts/plot_dataset_all.py --config configs/plot_dataset_all.yaml --timestep 3408_100
import argparse
import os
import sys
from pathlib import Path

os.environ["PTNOLATEX"] = "1"

PRPJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PRPJECT_ROOT))

from src.config import load_config
from src.dataset_io import load_dataset
from src.dataset_plot import (
    create_colormap_plot_jobs,
    create_vdf_plot_jobs,
    plot_labeled_colormap,
    plot_vdf_xz_slice,
    run_plot_jobs,
)
from src.timesteps import create_timestep_path

def main(config_path, timestep):
    config = load_config(config_path)

    dataset_dir = create_timestep_path(
        path_template=config["dataset_dir"],
        timestep=timestep
    )

    output_dir = create_timestep_path(
        path_template=config["output_dir"],
        timestep=timestep
    )

    plot_config = config["plot"]

    vdflim = float(plot_config.get("vdflim", 2e6))
    n_jobs = int(plot_config.get("n_jobs", 1))
    colormap_config = plot_config.get("colormap", {})

    X, y, metadata = load_dataset(dataset_dir, mmap=True)

    colormap_jobs = create_colormap_plot_jobs(
        metadata=metadata,
        output_dir=output_dir,
        colormap_config=colormap_config,
    )
    run_plot_jobs(
        plot_function=plot_labeled_colormap,
        plot_jobs=colormap_jobs,
        n_jobs=n_jobs,
    )

    vdf_jobs = create_vdf_plot_jobs(
        X=X,
        y=y,
        metadata=metadata,
        output_dir=output_dir,
        vdflim=vdflim,
    )
    run_plot_jobs(
        plot_function=plot_vdf_xz_slice,
        plot_jobs=vdf_jobs,
        n_jobs=n_jobs,
    )

    print(f"Dataset directory: {dataset_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Saved {len(colormap_jobs)} colormap plots")
    print(f"Saved {len(vdf_jobs)} VDF plots")

if __name__=="__main__":
    parser = argparse.ArgumentParser(
        description="Plot all xz VDF samples from saved dataset."
    )

    parser.add_argument(
        "--config",
        required=True,
        help="Path to plotting YAML config file."
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
