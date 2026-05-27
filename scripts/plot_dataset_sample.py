#python scripts/plot_dataset_sample.py --config configs/plot_dataset.yaml --timestep 3408_100
import argparse
import sys
from pathlib import Path

PRPJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PRPJECT_ROOT))

from src.config import load_config
from src.dataset_io import load_dataset
from src.dataset_plot import plot_vdf_xz_slice
from src.timesteps import create_timestep_path
from src.vdf_helpers import get_velocity_mesh_extent_from_file

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

    sample_index = 0

    plot_config = config["plot"]

    dv = float(plot_config.get("dv", 30000.0))
    threshold = float(plot_config.get("threshold", 8.301134972025815e-16))
    vdflim = float(plot_config.get("vdflim", 2e6))

    X, y, metadata = load_dataset(dataset_dir)

    metadata_row = metadata.iloc[sample_index]
    file_location = metadata_row["file_location"]

    extent = get_velocity_mesh_extent_from_file(
        file_location=file_location
    )

    output_path = output_dir / f"sample_{sample_index:04d}_xz.png"

    plot_vdf_xz_slice(
        vdf=X[sample_index],
        y_label=y[sample_index],
        metadata_row=metadata_row,
        extent=extent,
        output_path=output_path,
        dv=dv,
        threshold=threshold,
        vdflim=vdflim,
    )

    print(f"Dataset directory: {dataset_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Saved plot to: {output_path}")

if __name__=="__main__":
    parser = argparse.ArgumentParser(
        description="Plot xz VDF sample from saved dataset."
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