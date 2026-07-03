# python scripts/create_log_slice_cache.py --config configs/create_log_slice_cache.yaml --dataset-id 2000_2300

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.autoencoder_data import create_log_slice_cache_for_dataset
from src.config import load_config


def main(config_path, dataset_id, model_id=None):
    """
    Create or reuse a log-scaled xz-slice cache for a dataset.

    Parameters
    ----------
    config_path : str
        Path to cache creation YAML config.
    dataset_id : str
        Dataset identifier.
    model_id : str, optional
        Model identifier used only for path templates.
    """

    config = load_config(config_path)
    result = create_log_slice_cache_for_dataset(
        config=config,
        dataset_id=dataset_id,
        model_id=model_id,
    )
    cache_metadata = result["cache_metadata"]
    cache_config = result["cache_config"]

    print(f"Dataset directory: {result['dataset_dir']}")
    print(f"Raw VDF shape: {result['raw_vdf_shape']}")
    print(f"Cache path: {cache_config['cache_path']}")
    print(f"Cache metadata path: {cache_config['metadata_path']}")
    print(f"Cache shape: {result['cache_shape']}")
    print(f"Cache reused or created: {cache_metadata.get('enabled', False)}")
    if "elapsed_seconds" in cache_metadata:
        print(f"Cache creation seconds: {cache_metadata['elapsed_seconds']:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create or reuse cached log-scaled VDF xz slices."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to cache creation YAML config.",
    )
    parser.add_argument(
        "--dataset-id",
        required=True,
        help="Dataset identifier.",
    )
    parser.add_argument(
        "--model-id",
        default=None,
        help="Optional model identifier used only for path templates.",
    )
    args = parser.parse_args()

    main(
        config_path=args.config,
        dataset_id=args.dataset_id,
        model_id=args.model_id,
    )
