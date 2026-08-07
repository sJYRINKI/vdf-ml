"""Stage 2: open one VLSV source and prepare reusable VDF access.

This stage follows CNN loading and precedes per-cell representation
preparation. It receives prediction settings, a timestep, and the loaded
checkpoint and returns an open reader, resolved source producers, a dense
VDF extractor, and the raw interpolation plan when needed.
"""

from dataclasses import dataclass

from src.data.dense_vdf import (
    VdfExtractor,
    resolve_velocity_population,
)
from src.data.step_01_open_vlsv_files import (
    create_timestep_path,
    resolve_vlsv_source,
)
from src.data.velocity_grid import (
    create_trilinear_volume_plan,
    normalize_velocity_grid_geometry,
)
from src.physics.physical_context import (
    resolve_sparsity_threshold_variable,
)
from src.physics.vlsv_physical_values import resolve_bulk_velocity_source


_DEFAULT_BACKGROUND_LIMIT_MS = 1.5e6


@dataclass(frozen=True)
class PredictionPlotConfig:
    """Hold combined prediction-figure settings.

    Instances live for one coordinate or region command and carry only
    figure choices; they do not own the VLSV reader or any Matplotlib
    objects. The orchestration stage passes the record to the final plotting
    layer after inference has produced immutable prediction rows.

    Attributes
    ----------
    enabled : bool
        Whether the workflow saves a combined figure.
    spatial_box_re : tuple of float
        Fixed-order ``(xmin, xmax, zmin, zmax)`` bounds in Earth radii.
    color_limits : tuple of float
        Symmetric bulk-x-velocity limits in metres per second.
    maximum_examples : int
        Maximum representative VDF rows below a region map.
    representative_selection : str
        Deterministic representative selection strategy.
    explicit_cids : tuple of int
        Cell IDs used by the explicit selection strategy.
    show_streamlines : bool
        Whether magnetic streamlines are requested.
    dpi : int
        Output figure resolution.
    velocity_limit_ms : float
        Symmetric VDF-panel velocity limit in metres per second.
    spatial_slice_y_re : float or None
        Optional y-coordinate for a three-dimensional source slice in Earth
        radii.
    """

    enabled: bool
    spatial_box_re: tuple
    color_limits: tuple
    maximum_examples: int
    representative_selection: str
    explicit_cids: tuple
    show_streamlines: bool
    dpi: int
    velocity_limit_ms: float
    spatial_slice_y_re: float | None


@dataclass(frozen=True)
class PredictionSettings:
    """Hold direct coordinate or region configuration values.

    Stage 2 creates this immutable record once per command. It separates
    source-selection, batching, spatial-region, and plotting choices from
    checkpoint-owned model settings before individual VLSV timesteps are
    opened.

    Attributes
    ----------
    file_source : str
        Selected source-template key.
    file_templates : dict
        Timestep-formatted VLSV path templates.
    population : str or None
        Requested VDF population.
    region_re : dict or None
        Region bounds in Earth radii.
    batch_size : int
        Number of cells prepared for one CNN inference call.
    plotting : PredictionPlotConfig
        Combined-figure settings.
    """

    file_source: str
    file_templates: dict
    population: str | None
    region_re: dict | None
    batch_size: int
    plotting: PredictionPlotConfig


@dataclass(frozen=True)
class PreparedPredictionSource:
    """Hold one open source and its reusable prediction state.

    One instance is created for a single VLSV timestep and reused for every
    selected cell in that file. It owns references to the open reader,
    file-scoped producer choices, dense extractor, and optional raw-grid
    interpolation plan; the caller controls the reader lifetime.

    Attributes
    ----------
    reader : object
        Open Analysator VLSV reader.
    resolved_source : ResolvedVlsvSource
        File-scoped density, magnetic-field, population, and velocity-mesh
        information.
    sparsity_threshold_variable : str
        Historical or population-scoped raw-VDF plotting threshold selected
        once when combined figure output is enabled.
    bulk_velocity_source : tuple or None
        Total-flow producer selected once only for rotated Hermite input.
    extractor : VdfExtractor
        Sparse-to-dense VDF extractor for the selected population.
    interpolation_plan : dict or None
        Source-to-training volume interpolation plan for raw prediction.
    """

    reader: object
    resolved_source: object
    sparsity_threshold_variable: str | None
    bulk_velocity_source: tuple | None
    extractor: VdfExtractor
    interpolation_plan: dict | None


def load_prediction_settings(
    config,
    *,
    mode,
    file_source=None,
    plotting_enabled=None,
):
    """Read direct prediction settings from one current configuration.

    This configuration step runs before a checkpoint or source file is used.
    It resolves command-line source and plotting overrides once, keeps the
    fixed x-z view independent from region cell selection, and returns the
    immutable settings consumed by every later timestep. Prediction maps use
    the shared bulk-x-velocity renderer with symmetric physical limits.

    Parameters
    ----------
    config : mapping
        Current coordinate or region prediction configuration.
    mode : {"coordinate", "region"}
        Public workflow consuming the settings.
    file_source : str or None, optional
        CLI override for the source-template key.
    plotting_enabled : bool or None, optional
        CLI override for combined-figure creation.

    Returns
    -------
    PredictionSettings
        Direct values used by source loading, batching, and plotting.
    """

    plot_values = config.get("plotting", {})
    enabled = (
        bool(plot_values.get("enabled", False))
        if plotting_enabled is None
        else bool(plotting_enabled)
    )
    spatial_box_re = tuple(
        float(value) for value in plot_values["spatial_box_re"]
    )
    region_re = config.get("region_re") if mode == "region" else None
    plotting = PredictionPlotConfig(
        enabled=enabled,
        spatial_box_re=spatial_box_re,
        color_limits=(
            -_DEFAULT_BACKGROUND_LIMIT_MS,
            _DEFAULT_BACKGROUND_LIMIT_MS,
        ),
        maximum_examples=int(
            plot_values.get("maximum_examples", 5)
        ),
        representative_selection=str(
            plot_values.get(
                "representative_selection",
                "highest_confidence_per_class",
            )
        ),
        explicit_cids=tuple(
            int(value)
            for value in (plot_values.get("explicit_cids") or ())
        ),
        show_streamlines=bool(
            plot_values.get("show_streamlines", True)
        ),
        dpi=int(plot_values.get("dpi", 150)),
        velocity_limit_ms=float(
            plot_values.get("velocity_limit_ms", 2.0e6)
        ),
        spatial_slice_y_re=(
            None
            if plot_values.get("spatial_slice_y_re") is None
            else float(plot_values["spatial_slice_y_re"])
        ),
    )
    prediction_values = config.get("prediction", {})
    return PredictionSettings(
        file_source=str(
            config["file_source"]
            if file_source is None
            else file_source
        ),
        file_templates=config["file_templates"],
        population=config.get("population"),
        region_re=region_re,
        batch_size=int(prediction_values.get("batch_size", 16)),
        plotting=plotting,
    )


def source_path_for_timestep(settings, timestep):
    """Resolve the configured VLSV path for one timestep.

    The helper keeps timestep formatting in the source-loading stage so both
    coordinate and region orchestration choose files with the same template
    convention.

    Parameters
    ----------
    settings : PredictionSettings
        Source key and timestep-formatted path templates.
    timestep : int
        Simulation timestep.

    Returns
    -------
    tuple of str and pathlib.Path
        Source key and direct VLSV path.
    """

    source = settings.file_source
    return (
        source,
        create_timestep_path(
            settings.file_templates[source],
            int(timestep),
        ),
    )


def load_source_vdf(
    source_path,
    loaded,
    *,
    population=None,
    reader_factory=None,
    plotting_enabled=False,
):
    """Open one VLSV file and prepare reusable source VDF access.

    Stage 2 resolves the population and file-scoped VLSV producers once,
    including the historical or population-scoped sparsity threshold used
    by every displayed VDF cut. It then creates the sparse-to-dense extractor
    reused by every selected cell. Raw prediction also constructs one
    cell-centred trilinear plan for all three velocity axes from the source
    grid to the training grid; Hermite prediction instead retains the
    same-cell physical producers needed by Stage 3.

    Parameters
    ----------
    source_path : str or pathlib.Path
        VLSV source selected for one timestep.
    loaded : LoadedCnnCheckpoint
        Checkpoint defining raw or Hermite input preparation.
    population : str or None, optional
        Requested VDF population.
    reader_factory : callable or None, optional
        Reader constructor used by tests or Analysator by default.
    plotting_enabled : bool, optional
        Whether raw VDF panels require source threshold resolution.

    Returns
    -------
    PreparedPredictionSource
        Open reader, file-scoped producers, dense extractor, and optional raw
        interpolation plan.
    """

    if reader_factory is None:
        import analysator as pt

        reader_factory = pt.vlsvfile.VlsvReader
    reader = reader_factory(str(source_path))
    selected_population = resolve_velocity_population(
        reader,
        population,
    )
    representation = loaded.checkpoint["representation"]
    hermite_rotate = representation == "hermite" and bool(
        loaded.checkpoint["hermite_rotate"]
    )
    resolved_source = resolve_vlsv_source(
        reader,
        selected_population,
        require_magnetic_field=hermite_rotate,
    )
    sparsity_threshold_variable = (
        resolve_sparsity_threshold_variable(
            reader,
            selected_population,
        )
        if plotting_enabled
        else None
    )
    bulk_velocity_source = (
        resolve_bulk_velocity_source(reader, selected_population)
        if hermite_rotate
        else None
    )
    extractor = VdfExtractor(
        reader,
        pop=selected_population,
        velocity_mesh_shape=resolved_source.velocity_mesh_shape,
    )
    interpolation_plan = None
    if representation == "raw":
        target_grid = normalize_velocity_grid_geometry(
            loaded.checkpoint["preprocessing"][
                "training_velocity_grid"
            ]
        )
        interpolation_plan = create_trilinear_volume_plan(
            extractor.velocity_grid,
            target_grid,
        )
    return PreparedPredictionSource(
        reader=reader,
        resolved_source=resolved_source,
        sparsity_threshold_variable=sparsity_threshold_variable,
        bulk_velocity_source=bulk_velocity_source,
        extractor=extractor,
        interpolation_plan=interpolation_plan,
    )


__all__ = [
    "PredictionPlotConfig",
    "PredictionSettings",
    "PreparedPredictionSource",
    "load_prediction_settings",
    "load_source_vdf",
    "source_path_for_timestep",
]
