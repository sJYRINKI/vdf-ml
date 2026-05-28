#python scripts/plot_dataset_all.py --config configs/plot_dataset_all.yaml --timestep 3408_100
import argparse
import sys
from pathlib import Path

PRPJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PRPJECT_ROOT))

from src.config import load_config
from src.dataset_io import load_dataset
from src.dataset_plot import plot_vdf_xz_slice
from src.timesteps import create_timestep_path
from src.vdf_helpers import get_vdf_plot_parameters_from_file

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

    X, y, metadata = load_dataset(dataset_dir)

    class_frame_counts = {}

    for sample_index in range(X.shape[0]):
        metadata_row = metadata.iloc[sample_index]
        class_name = metadata_row["class_name"]
        file_location = metadata_row["file_location"]
        cid = int(metadata_row["cid"])

        if class_name not in class_frame_counts:
            class_frame_counts[class_name] = 0

        class_frame_index = class_frame_counts[class_name]

        extent, dv, threshold = get_vdf_plot_parameters_from_file(
            file_location=file_location,
            cid=cid,
            vdf_shape=X[sample_index].shape,
        )

        print(dv)
        print(threshold)

        class_output_dir = output_dir / class_name
        output_path = class_output_dir / f"sample_{class_frame_index:04d}_xz.png"

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

        class_frame_counts[class_name] += 1

    print(f"Dataset directory: {dataset_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Saved plots to: {output_path}")

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