#python scripts/plot_dataset_all.py --config configs/plot_dataset_all.yaml --timestep 3408_100
import argparse
import os
import sys
from pathlib import Path

os.environ["PTNOLATEX"] = "1"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.config import load_config
from src.dataset_io import load_dataset
from src.dataset_plot import (
    add_dataset_sampling_plot_config,
    create_colormap_plot_jobs,
    create_or_load_plot_xz_slice_cache,
    iter_vdf_plot_jobs,
    plot_labeled_colormap,
    plot_vdf_sample_from_dataset,
    run_plot_jobs,
)
from src.timesteps import create_timestep_path


def main(config_path, timestep, plot_mode="all"):
    """
    Plot spatial colormaps and optional VDF samples for one dataset.

    Parameters
    ----------
    config_path : str or pathlib.Path
        Path to the plotting configuration file.
    timestep : str
        Dataset timestep identifier.
    plot_mode : {"all", "colormaps"}, optional
        Whether to plot both colormaps and VDF samples or only colormaps.
    """

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
    cache_config = plot_config.get("cache", {})
    colormap_config = plot_config.get("colormap", {})
    colormap_config = add_dataset_sampling_plot_config(
        colormap_config=colormap_config,
        dataset_config_path=config["dataset_config"],
    )

    X, y, metadata = load_dataset(dataset_dir, mmap=True)
    X_plot = None
    if plot_mode == "all":
        X_plot = create_or_load_plot_xz_slice_cache(
            X=X,
            dataset_dir=dataset_dir,
            dataset_id=timestep,
            cache_config=cache_config,
        )

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

    if plot_mode == "all":
        n_vdf_plots = int(X.shape[0])
        vdf_jobs = iter_vdf_plot_jobs(
            X=X,
            y=y,
            metadata=metadata,
            output_dir=output_dir,
            vdflim=vdflim,
            X_plot=X_plot,
        )
        run_plot_jobs(
            plot_function=plot_vdf_sample_from_dataset,
            plot_jobs=vdf_jobs,
            n_jobs=n_jobs,
        )

    print(f"Dataset directory: {dataset_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Saved {len(colormap_jobs)} colormap plots")
    if plot_mode == "all":
        print(f"Saved {n_vdf_plots} VDF plots")
    else:
        print("Skipped VDF plots (plot mode: colormaps)")

if __name__=="__main__":
    parser = argparse.ArgumentParser(
        description="Plot spatial colormaps and VDF samples from a saved dataset."
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

    parser.add_argument(
        "--plot-mode",
        choices=("all", "colormaps"),
        default="all",
        help="Plot everything or only spatial colormaps."
    )

    args = parser.parse_args()

    main(
        config_path=args.config,
        timestep=args.timestep,
        plot_mode=args.plot_mode,
    )
