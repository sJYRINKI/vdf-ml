"""Predict the VDF nearest one requested coordinate with a version-6 CNN.

The command loads prediction settings and delegates source-cell selection,
representation construction, inference, physical-unit topology output, and
the optional combined figure to ``src.prediction``.
"""

import argparse

from src.configuration import load_config
from src.prediction import run_coordinate_prediction


def main(argv=None):
    """
    Parse arguments and run the coordinate workflow.

    The CLI keeps the requested Earth-radius coordinate explicit, applies
    optional representation, source, plot, and runtime model-stage-device
    overrides to the YAML mapping, and delegates nearest-cell selection,
    CNN inference, one-row CSV output, and the optional combined figure to
    ``src.prediction``.

    Parameters
    ----------
    argv : sequence of str, optional
        Command-line arguments excluding the executable name.

    Returns
    -------
    int
        Zero after successful prediction output creation.
    """

    parser = argparse.ArgumentParser(
        description="Predict the nearest VDF cell with a version-6 CNN.",
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--timestep", required=True, type=int)
    parser.add_argument(
        "--coord-re",
        required=True,
        nargs=3,
        type=float,
        metavar=("X_RE", "Y_RE", "Z_RE"),
    )
    parser.add_argument(
        "--representation",
        choices=("raw", "hermite"),
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--model-parallel-gpus",
        type=int,
        help="Place consecutive CNN stages across this many visible GPUs.",
    )
    parser.add_argument("--file-source")
    parser.add_argument(
        "--plot",
        dest="plotting_enabled",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override the configured combined prediction figure.",
    )
    arguments = parser.parse_args(argv)

    config = load_config(arguments.config)
    if arguments.representation is not None:
        config = {**config, "representation": arguments.representation}
    model_parallel = {}
    if arguments.model_parallel_gpus is not None:
        model_parallel["model_parallel_gpus"] = (
            arguments.model_parallel_gpus
        )
    result = run_coordinate_prediction(
        config,
        arguments.checkpoint,
        arguments.output_dir,
        arguments.timestep,
        arguments.coord_re,
        device=arguments.device,
        **model_parallel,
        file_source=arguments.file_source,
        plotting_enabled=arguments.plotting_enabled,
    )
    prediction = result["prediction"]
    print(f"CSV: {result['csv_path']}")
    if result["plot_path"] is not None:
        print(f"Figure: {result['plot_path']}")
    print(
        "Prediction: "
        f"{prediction['predicted_class_name']} "
        f"(class_id={prediction['predicted_class_id']}, "
        f"p={prediction['predicted_probability']:.6f})"
    )
    return 0


if __name__ == "__main__":
    main()
