"""Orchestrate the ordered stages for one coordinate prediction.

The public workflow calls Stage 1 to load the CNN, Stage 2 to open the VLSV
source, Stage 3 to prepare the selected VDF, Stage 4 to run inference, and
Stage 5 to save one ``predictions_<timestep>.csv`` row. The CSV is written
before an enabled combined figure reuses the same source reader, selected
cell, and already-decoded prediction.
"""

from pathlib import Path

import numpy as np

from src.data.step_02_find_vdf_cells import (
    get_nearest_vdf_cellid,
    get_vdf_cells_with_coords_re,
)
from src.prediction.step_01_load_cnn_model import (
    load_prediction_model,
)
from src.prediction.step_02_load_source_vdf import (
    load_prediction_settings,
    load_source_vdf,
    source_path_for_timestep,
)
from src.prediction.step_03_prepare_prediction_input import (
    prepare_prediction_input,
)
from src.prediction.step_04_run_cnn_prediction import (
    run_cnn_prediction,
)
from src.prediction.step_05_save_prediction_outputs import (
    create_prediction_rows,
    write_prediction_csv,
)


def run_coordinate_prediction(
    config,
    checkpoint_path,
    output_dir,
    timestep,
    coordinate_re,
    *,
    device="auto",
    model_parallel_gpus=1,
    file_source=None,
    reader_factory=None,
    plotting_enabled=None,
    spatial_renderer=None,
    cut_loader=None,
):
    """Predict the VDF nearest one requested coordinate.

    This public orchestrator connects all five prediction stages. It keeps
    the requested position separate from the nearest VDF-bearing cell
    centre, prepares that cell with the checkpoint's raw or Hermite
    convention, decodes one CNN result, and writes the structured result
    before optionally composing the matching spatial/VDF figure. It uses the
    same row builder and CSV writer as region prediction, while requested and
    selected positions remain separate in Earth radii.

    Parameters
    ----------
    config : mapping
        Current coordinate-prediction configuration.
    checkpoint_path : str or pathlib.Path
        Current CNN checkpoint.
    output_dir : str or pathlib.Path
        Directory that receives ``predictions_<timestep>.csv`` and an
        optional combined PNG.
    timestep : int
        Simulation timestep.
    coordinate_re : sequence of float
        Requested ``[x, y, z]`` coordinate in Earth radii.
    device : str, optional
        PyTorch device string or ``"auto"``.
    model_parallel_gpus : int, optional
        Number of consecutive devices hosting contiguous model stages.
    file_source : str or None, optional
        CLI override for the configured source-template key.
    reader_factory : callable or None, optional
        VLSV reader constructor used by tests or Analysator by default.
    plotting_enabled : bool or None, optional
        CLI override for combined-figure creation.
    spatial_renderer : callable or None, optional
        Spatial-background renderer used by focused plotting tests.
    cut_loader : callable or None, optional
        Representative raw-VDF cut loader used by focused plotting tests.

    Returns
    -------
    dict
        Output directory, CSV path, decoded prediction row, optional plot
        path, and optional plotting diagnostics. The prediction row contains
        class probabilities and six topology values in checkpoint order.

    Notes
    -----
    Requested and selected coordinates remain separate. Topology distances
    and ``dx,dz`` outputs are inverse-scaled to Earth radii before saving.
    """

    coordinate_re = np.asarray(coordinate_re, dtype=np.float64)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    settings = load_prediction_settings(
        config,
        mode="coordinate",
        file_source=file_source,
        plotting_enabled=plotting_enabled,
    )
    loaded = load_prediction_model(
        checkpoint_path,
        device=device,
        model_parallel_gpus=model_parallel_gpus,
    )
    source_name, source_path = source_path_for_timestep(
        settings,
        int(timestep),
    )
    prepared_source = load_source_vdf(
        source_path,
        loaded,
        population=settings.population,
        reader_factory=reader_factory,
        plotting_enabled=settings.plotting.enabled,
    )
    cellids, coordinates_re = get_vdf_cells_with_coords_re(
        prepared_source.reader,
        pop=prepared_source.resolved_source.population,
    )
    cid = get_nearest_vdf_cellid(
        coordinate_re,
        cellids,
        coordinates_re,
    )
    selected_index = int(np.flatnonzero(cellids == cid)[0])
    selected_coordinate_re = coordinates_re[selected_index]
    sample = prepare_prediction_input(
        prepared_source,
        cid,
        loaded,
    )
    prediction = run_cnn_prediction(
        loaded,
        sample.tensor[None, ...],
    )
    row = create_prediction_rows(
        prediction,
        representation=loaded.checkpoint["representation"],
        timestep=int(timestep),
        file_source=source_name,
        cellids=[cid],
        coordinates_re=selected_coordinate_re[None, :],
        loaded=loaded,
    )[0]
    row.update(
        {
            "requested_x_re": float(coordinate_re[0]),
            "requested_y_re": float(coordinate_re[1]),
            "requested_z_re": float(coordinate_re[2]),
        }
    )
    csv_path = write_prediction_csv(
        output_dir / f"predictions_{int(timestep)}.csv",
        [row],
        loaded,
    )
    plot_path = None
    plotting = None
    if settings.plotting.enabled:
        from src.plotting.prediction_visualization import (
            save_coordinate_prediction_figure,
        )

        plot_path = output_dir / "prediction_combined.png"
        plotting = save_coordinate_prediction_figure(
            plot_path,
            reader=prepared_source.reader,
            prepared_source=prepared_source,
            row=row,
            requested_coordinate_re=coordinate_re,
            class_ids=loaded.checkpoint["class_mapping"]["class_ids"],
            class_names=loaded.checkpoint["class_mapping"]["class_names"],
            plotting_config=settings.plotting,
            spatial_renderer=spatial_renderer,
            cut_loader=cut_loader,
        )
    return {
        "output_dir": output_dir,
        "csv_path": csv_path,
        "prediction": row,
        "plot_path": plot_path,
        "plotting": plotting,
    }


__all__ = ["run_coordinate_prediction"]
