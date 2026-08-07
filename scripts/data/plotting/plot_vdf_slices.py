"""Expose shared peak-centred physical VDF planes as a thin command.

The command selects deterministic metadata rows and delegates all extraction
and drawing to ``src.plotting.vdf_slices``. It can request ``xz``,
``xy``, or ``yz`` in user order; omission keeps the stable three-plane
default, while extraction Stage 6 always requests only ``xz``.
"""

import argparse

from src.plotting.vdf_slices import (
    DEFAULT_VDF_PLANES,
    plot_dataset_vdf_slices,
)


def main(argv=None):
    """
    Plot one or more deterministic dataset VDF samples.

    The command parses one selector family and passes it to the plotting
    package, which preserves metadata/array alignment and draws
    full-resolution physical cuts through each sample's own VDF peak.

    Parameters
    ----------
    argv : sequence of str, optional
        Command-line arguments excluding the executable name.

    Returns
    -------
    int
        Zero after publishing one PNG.
    """

    parser = _create_parser()
    args = parser.parse_args(argv)
    result = plot_dataset_vdf_slices(
        dataset_dir=args.dataset_dir,
        output_file=args.output,
        sample_indices=args.sample_index or (),
        cids=args.cid or (),
        class_names=args.class_name or (),
        number_of_samples=args.number_of_samples,
        planes=tuple(args.plane or DEFAULT_VDF_PLANES),
        velocity_limit_ms=args.velocity_limit_ms,
        dpi=args.dpi,
    )
    print(f"Dataset: {args.dataset_dir}")
    print(f"Samples: {list(result.sample_indices)}")
    print(f"CIDs: {list(result.cids)}")
    print(f"Plot: {result.output_file}")
    return 0


def _create_parser():
    """Create the current dataset VDF-slice argument parser.

    Returns
    -------
    argparse.ArgumentParser
        Parser for dataset selection, deterministic sample selectors,
        physical velocity display limits, and the combined PNG destination.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Plot requested xz, xy, and yz planes through each VDF peak."
        )
    )
    parser.add_argument(
        "--dataset-dir",
        required=True,
        help="Dataset directory containing X.npy and metadata.csv.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Combined PNG output path.",
    )
    selectors = parser.add_mutually_exclusive_group()
    selectors.add_argument(
        "--sample-index",
        type=int,
        action="append",
        help="Stored sample index; repeat to select several.",
    )
    selectors.add_argument(
        "--cid",
        type=int,
        action="append",
        help="Spatial cell ID; repeat to select several.",
    )
    selectors.add_argument(
        "--class-name",
        action="append",
        help="Physical class name; repeat to select several.",
    )
    parser.add_argument(
        "--number-of-samples",
        type=int,
        help="Deterministic maximum sample count.",
    )
    parser.add_argument(
        "--plane",
        action="append",
        choices=DEFAULT_VDF_PLANES,
        help=(
            "Physical peak-crossing plane; repeat to preserve a requested "
            "order. Omission draws xz, xy, yz."
        ),
    )
    parser.add_argument(
        "--velocity-limit-ms",
        type=float,
        help="Optional symmetric velocity-axis limit in m/s.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="Positive PNG resolution, default 150.",
    )
    return parser


if __name__ == "__main__":
    main()
