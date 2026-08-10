"""Train one full-volume raw or Hermite multitask autoencoder.

The command loads model configuration and delegates timestep splitting,
training-only representation/topology scaling, one-process model-parallel
optimization, evaluation, checkpointing, and artifact writing to
``src.autoencoder``.
"""

import argparse

from src.autoencoder import (
    run_autoencoder_training,
)
from src.configuration import load_config


def main(argv=None):
    """Train one full-volume Conv3d raw or Hermite autoencoder.

    The command selects a representation and optional PyTorch device before
    delegating chronological splitting, training-only scaling,
    reconstruction plus auxiliary topology optimization, and artifact
    writing to the ordered autoencoder package.

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
            "Train one full-3D topology-aware raw or Hermite autoencoder."
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
    parser.add_argument(
        "--model-parallel-gpus",
        type=int,
        help=(
            "Visible GPUs used for consecutive model stages in one process; "
            "overrides model_parallel_gpus in YAML."
        ),
    )
    args = parser.parse_args(argv)
    config = load_config(args.config)
    result = run_autoencoder_training(
        config,
        args.dataset_dir,
        args.output_dir,
        args.representation,
        device=args.device,
        model_parallel_gpus=args.model_parallel_gpus,
    )
    print(f"Checkpoint: {result['checkpoint_path']}")
    print(f"Metrics: {result['metrics_path']}")
    print(f"Training history: {result['training_history_path']}")
    print(f"Reconstruction figure: {result['reconstruction_figure_path']}")
    print(f"Representation: {result['representation']}")
    print(f"Device: {result['device']}")
    print(
        "Effective model-parallel GPUs: "
        f"{result['effective_model_parallel_gpus']}"
    )
    print(f"Best epoch: {result['best_epoch']}")
    print(
        "Validation total loss: "
        f"{result['validation_total_loss']:.8g}"
    )
    return 0


if __name__ == "__main__":
    main()
