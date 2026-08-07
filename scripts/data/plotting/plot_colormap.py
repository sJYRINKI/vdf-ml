"""Expose the shared Stage 6 dataset-colormap renderer as a thin command.

The command loads extraction plotting settings and delegates the fixed
x-velocity background, yellow/blue/red selection roles, exact X/O areas,
legend, and PNG output to ``src.plotting.dataset_colormap``.
"""

import argparse

from src.configuration import load_config
from src.plotting.dataset_colormap import (
    plot_dataset_colormap,
)


def main(argv=None):
    """
    Plot one dataset timestep with the shared Stage 6 visual design.

    The command combines explicit CLI choices with optional plotting values
    in the extraction YAML, then delegates all Matplotlib and Analysator
    drawing to the plotting package. Its defaults reproduce extraction
    Stage 6, while visibility and physical view overrides remain available.

    Parameters
    ----------
    argv : sequence of str, optional
        Command-line arguments excluding the executable name.

    Returns
    -------
    int
        Zero after saving one PNG.
    """

    parser = _create_parser()
    args = parser.parse_args(argv)
    config = load_config(args.config)
    plotting = config.get("plotting", {})
    result = plot_dataset_colormap(
        dataset_dir=args.dataset_dir,
        output_file=args.output,
        extraction_config=config,
        timestep=args.timestep,
        source_vlsv=args.source_vlsv,
        spatial_box_re=(
            args.spatial_box_re
            if args.spatial_box_re is not None
            else plotting.get("spatial_box_re")
        ),
        velocity_limit_ms=plotting.get("spatial_velocity_limit_ms"),
        spatial_slice_y_re=(
            args.spatial_slice_y_re
            if args.spatial_slice_y_re is not None
            else plotting.get("spatial_slice_y_re")
        ),
        show_all_vdf_cells=_boolean_option(
            args.show_all_vdf_cells,
            plotting,
            "show_all_vdf_cells",
            True,
        ),
        show_labeled_cells=_boolean_option(
            args.show_labeled_cells,
            plotting,
            "show_labeled_cells",
            True,
        ),
        show_class_legend=_boolean_option(
            args.show_class_legend,
            plotting,
            "show_class_legend",
            True,
        ),
        show_magnetic_streamlines=_boolean_option(
            args.show_magnetic_streamlines,
            plotting,
            "show_magnetic_streamlines",
            True,
        ),
        dpi=(
            args.dpi
            if args.dpi is not None
            else plotting.get("dpi", 150)
        ),
    )
    print(f"Dataset: {args.dataset_dir}")
    print(f"Source: {result.source_vlsv}")
    print(f"Timestep: {result.timestep}")
    print(f"Labeled cells: {result.labeled_cell_count}")
    print(f"Plot: {result.output_file}")
    return 0


def _create_parser():
    """Create the current dataset-colormap argument parser.

    Returns
    -------
    argparse.ArgumentParser
        Parser for source selection, saved metadata, spatial display options,
        and the PNG destination.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Plot a VLSV spatial colormap with current dataset VDF cells."
        )
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Extraction YAML configuration.",
    )
    parser.add_argument(
        "--dataset-dir",
        required=True,
        help="Dataset directory containing metadata.csv.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="PNG output path.",
    )
    parser.add_argument(
        "--timestep",
        type=int,
        help="Stored timestep to plot; defaults to the first.",
    )
    parser.add_argument(
        "--source-vlsv",
        help="Explicit VLSV source path.",
    )
    parser.add_argument(
        "--spatial-box-re",
        type=float,
        nargs=4,
        metavar=("XMIN", "XMAX", "ZMIN", "ZMAX"),
        help="X-z plot bounds in Earth radii.",
    )
    parser.add_argument(
        "--spatial-slice-y-re",
        type=float,
        help="Y slice in Earth radii for a 3-D source.",
    )
    parser.add_argument(
        "--show-all-vdf-cells",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Show every source VDF-carrying cell.",
    )
    parser.add_argument(
        "--show-labeled-cells",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Show selected dataset VDF cells as red markers.",
    )
    parser.add_argument(
        "--show-class-legend",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Show the shared role, class-marker, and search-area legend.",
    )
    parser.add_argument(
        "--show-magnetic-streamlines",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Show magnetic streamlines when available.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        help="Positive PNG resolution, default 150.",
    )
    return parser


def _boolean_option(cli_value, config, key, default):
    """Resolve one optional Boolean CLI override over a config value.

    Parameters
    ----------
    cli_value : bool or None
        Explicit ``--show-*``/``--no-show-*`` result, if supplied.
    config : mapping
        Plotting configuration section.
    key : str
        Configuration key used when the CLI leaves the choice unset.
    default : bool
        Value used when neither source supplies the option.

    Returns
    -------
    bool
        Effective plotting choice.
    """

    if cli_value is not None:
        return bool(cli_value)
    return bool(config.get(key, default))


if __name__ == "__main__":
    main()
