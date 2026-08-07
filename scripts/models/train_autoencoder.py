"""Train one raw or Hermite reconstruction autoencoder.

The command loads model configuration and delegates timestep splitting,
training-only normalization, optimization, evaluation, checkpointing, and
artifact writing to ``src.autoencoder``.
"""

import argparse

from src.autoencoder import (
    run_autoencoder_training,
)
from src.configuration import load_config


def main(argv=None):
    """Train one raw or Hermite reconstruction autoencoder.

    The command selects a representation and optional PyTorch device before
    delegating chronological splitting, training-only normalization,
    normalized-space reconstruction, evaluation, and checkpoint writing
    to the ordered autoencoder package.

    Parameters
    ----------
    argv : sequence of str, optional
        Command-line arguments without the executable name. ``None`` reads
        arguments from the current process.

    Returns
    -------
    int
        Zero after training artifacts are written successfully.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Train one timestep-split raw or Hermite autoencoder."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--representation",
        required=True,
        choices=("raw", "hermite"),
    )
    parser.add_argument("--device")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    result = run_autoencoder_training(
        config,
        args.dataset_dir,
        args.output_dir,
        args.representation,
        device=args.device,
    )
    print(f"Checkpoint: {result['checkpoint_path']}")
    print(f"Metrics: {result['metrics_path']}")
    print(f"Reconstruction figure: {result['reconstruction_figure_path']}")
    print(f"Representation: {result['representation']}")
    print(f"Device: {result['device']}")
    print(f"Best epoch: {result['best_epoch']}")
    print(
        "Validation reconstruction MSE: "
        f"{result['validation_reconstruction_mse']:.8g}"
    )
    return 0


if __name__ == "__main__":
    main()
