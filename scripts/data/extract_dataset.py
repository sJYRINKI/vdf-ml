"""Create one staged raw or raw-plus-Hermite VDF dataset.

The command resolves CLI overrides and delegates physical sample selection,
VLSV extraction, metadata construction, and staged dataset writing to
``src.data``. The same package optionally renders dataset frames and
encodes their animations after the completed directory has been renamed.
"""

import argparse

from src.configuration import load_config
from src.data import run_extraction


def main(argv=None):
    """
    Run extraction from one YAML configuration.

    CLI values select the consecutive timesteps, configured train/test
    destination, and optional X/O selection overrides. All VLSV access,
    physical labeling, raw/Hermite transformation, and staged dataset
    writing remain in the ordered data package. Frame and animation stages
    are controlled entirely by the loaded extraction configuration.

    Parameters
    ----------
    argv : sequence of str, optional
        Command-line arguments excluding the executable name.

    Returns
    -------
    int
        Zero after dataset creation.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Extract physical VDF samples and create a raw or "
            "raw-plus-Hermite dataset."
        )
    )
    parser.add_argument(
        "--config",
        required=True,
        help="extraction YAML configuration.",
    )
    parser.add_argument(
        "--start-timestep",
        type=int,
        required=True,
        help="First timestep.",
    )
    parser.add_argument(
        "--n-timesteps",
        type=int,
        required=True,
        help="Number of consecutive timesteps.",
    )
    parser.add_argument(
        "--dataset-kind",
        choices=("train", "test"),
        required=True,
        help="Configured output-directory key.",
    )
    parser.add_argument(
        "--x-selection-method",
        choices=(
            "manual",
            "physical",
            "union",
            "consensus",
        ),
        help="Optional X-point selection override.",
    )
    parser.add_argument(
        "--o-selection-method",
        choices=(
            "manual",
            "physical",
            "union",
            "consensus",
        ),
        help="Optional O-point selection override.",
    )
    args = parser.parse_args(argv)

    result = run_extraction(
        config=load_config(args.config),
        start_timestep=args.start_timestep,
        n_timesteps=args.n_timesteps,
        dataset_kind=args.dataset_kind,
        x_selection_method=args.x_selection_method,
        o_selection_method=args.o_selection_method,
    )
    print(f"Dataset: {result}")
    return 0


if __name__ == "__main__":
    main()
