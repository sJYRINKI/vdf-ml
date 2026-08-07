"""Train one raw or Hermite multitask VDF CNN.

The command loads model configuration and delegates timestep splitting,
training-only scaling, optimization, evaluation, checkpointing, and
artifact writing to ``src.cnn``.

``--model-parallel-gpus`` keeps one Python process while the package places
consecutive encoder and output stages on several visible CUDA devices.
"""

import argparse

from src.cnn import run_cnn_training
from src.configuration import load_config


def main(argv=None):
    """
    Run the CNN training command.

    The script chooses the raw or Hermite representation, optional PyTorch
    device, and maximum model-parallel GPU count, then delegates timestep
    splitting, training-only normalization, consecutive stage placement,
    multitask optimization, evaluation, and the two saved outputs to the
    ordered CNN package.

    Parameters
    ----------
    argv : sequence of str, optional
        Command-line arguments excluding the executable name.

    Returns
    -------
    int
        Zero after a successful training run.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Train one timestep-split raw or Hermite CNN."
        )
    )
    parser.add_argument("--config", required=True, help="CNN YAML config.")
    parser.add_argument(
        "--dataset-dir",
        required=True,
        help="dataset directory.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Model output directory.",
    )
    parser.add_argument(
        "--representation",
        required=True,
        choices=("raw", "hermite"),
        help="public input representation.",
    )
    parser.add_argument(
        "--device",
        help="Override device with cpu, cuda, cuda:N, or auto.",
    )
    parser.add_argument(
        "--model-parallel-gpus",
        type=int,
        help=(
            "Override the number of consecutive GPUs used for model "
            "stages."
        ),
    )
    args = parser.parse_args(argv)

    config = load_config(args.config)
    overrides = {}
    if args.model_parallel_gpus is not None:
        overrides["model_parallel_gpus"] = args.model_parallel_gpus
    result = run_cnn_training(
        config=config,
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        representation=args.representation,
        device=args.device,
        **overrides,
    )
    print(f"Output: {result['output_dir']}")
    print(f"Representation: {result['representation']}")
    print(f"Device: {result['device']}")
    print(f"Best epoch: {result['best_epoch']}")
    print(
        "Validation macro F1: "
        f"{result['validation_macro_f1']:.6f}"
    )
    return 0


if __name__ == "__main__":
    main()
