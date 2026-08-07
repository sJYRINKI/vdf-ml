"""Predict bounded VDF cells across configured timesteps with a CNN checkpoint.

The command constructs the requested timestep sequence and delegates stable
cell selection, bounded inference, result streaming, and optional combined
figures to ``src.prediction``.
"""

import argparse

from src.configuration import load_config
from src.data.step_01_open_vlsv_files import create_timestep_list
from src.prediction import run_region_prediction


def main(argv=None):
    """
    Parse arguments and run the region workflow.

    The CLI expands the requested consecutive timestep range, applies
    optional representation, source, plot, and runtime model-stage-device
    overrides, and delegates stable spatial selection, bounded CNN
    inference, per-timestep CSV output, and optional combined figures to
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
        description="Predict VDF cells in a region with a CNN checkpoint.",
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--start-timestep", required=True, type=int)
    parser.add_argument("--n-timesteps", required=True, type=int)
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
        help="Override configured per-timestep combined figures.",
    )
    arguments = parser.parse_args(argv)

    config = load_config(arguments.config)
    if arguments.representation is not None:
        config = {**config, "representation": arguments.representation}
    timesteps = create_timestep_list(
        arguments.start_timestep,
        arguments.n_timesteps,
    )
    model_parallel = {}
    if arguments.model_parallel_gpus is not None:
        model_parallel["model_parallel_gpus"] = (
            arguments.model_parallel_gpus
        )
    result = run_region_prediction(
        config,
        arguments.checkpoint,
        arguments.output_dir,
        timesteps,
        device=arguments.device,
        **model_parallel,
        file_source=arguments.file_source,
        plotting_enabled=arguments.plotting_enabled,
    )
    print(f"Output: {result['output_dir']}")
    print(f"Representation: {result['representation']}")
    print(f"Samples by timestep: {result['sample_counts']}")
    return 0


if __name__ == "__main__":
    main()
