"""Orchestrate the ordered stages for bounded region prediction.

The public workflow loads the CNN once, repeats source loading and VDF
and same-cell context preparation for each timestep, runs bounded aligned
inference batches, and streams Stage 5 rows to
``predictions_<timestep>.csv``. Enabled combined figures reuse those saved
rows without rerunning the model.
"""

import csv
from pathlib import Path

import numpy as np

from src.data.step_02_find_vdf_cells import (
    create_region_mask_re,
    get_vdf_cells_with_coords_re,
)
from src.physics.plasma_context import (
    prepare_plasma_context_sources_for_cells,
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
    open_prediction_csv,
)


def run_region_prediction(
    config,
    checkpoint_path,
    output_dir,
    timesteps,
    *,
    device="auto",
    model_parallel_gpus=1,
    file_source=None,
    reader_factory=None,
    plotting_enabled=None,
    spatial_renderer=None,
    cut_loader=None,
):
    """Predict all selected VDF cells across requested timesteps.

    The region orchestrator loads the checkpoint once, opens each requested
    source timestep in order, selects VDF-bearing cells inside the configured
    Earth-radius box, and streams bounded inference batches to one CSV.
    Optional figures are constructed from the saved rows, so plotting never
    changes or repeats model inference.

    Parameters
    ----------
    config : mapping
        Current region-prediction configuration.
    checkpoint_path : str or pathlib.Path
        Current CNN checkpoint.
    output_dir : str or pathlib.Path
        Directory receiving one CSV and optional combined PNG per timestep.
    timesteps : sequence of int
        Simulation timesteps in requested output order.
    device : str, optional
        PyTorch device string or ``"auto"``.
    model_parallel_gpus : int, optional
        Number of consecutive devices hosting contiguous model stages.
    file_source : str or None, optional
        CLI override for the configured source-template key.
    reader_factory : callable or None, optional
        VLSV reader constructor used by tests or Analysator by default.
    plotting_enabled : bool or None, optional
        CLI override for per-timestep combined figures.
    spatial_renderer : callable or None, optional
        Spatial-background renderer used by focused plotting tests.
    cut_loader : callable or None, optional
        Representative raw-VDF cut loader used by focused plotting tests.

    Returns
    -------
    dict
        Output directory, checkpoint representation, and selected-cell count
        for each timestep.

    Notes
    -----
    Rows retain ascending VLSV cell-ID order. Topology outputs are saved in
    physical Earth-radius units after inverse scaling.
    """

    timesteps = tuple(int(value) for value in timesteps)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    settings = load_prediction_settings(
        config,
        mode="region",
        file_source=file_source,
        plotting_enabled=plotting_enabled,
    )
    loaded = load_prediction_model(
        checkpoint_path,
        device=device,
        model_parallel_gpus=model_parallel_gpus,
    )
    counts = {}
    for timestep in timesteps:
        source_name, source_path = source_path_for_timestep(
            settings,
            timestep,
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
        mask = create_region_mask_re(
            coordinates_re,
            settings.region_re,
        )
        selected_cellids = cellids[mask]
        selected_coordinates = coordinates_re[mask]
        prepare_plasma_context_sources_for_cells(
            prepared_source.reader,
            prepared_source.plasma_context_sources,
            selected_cellids,
        )
        csv_path = output_dir / f"predictions_{timestep}.csv"
        _stream_timestep_predictions(
            csv_path,
            prepared_source,
            selected_cellids,
            selected_coordinates,
            timestep=timestep,
            file_source=source_name,
            loaded=loaded,
            batch_size=settings.batch_size,
        )
        if settings.plotting.enabled:
            from src.plotting.prediction_visualization import (
                save_region_prediction_figure,
            )

            save_region_prediction_figure(
                output_dir
                / f"predictions_{timestep}_combined.png",
                reader=prepared_source.reader,
                prepared_source=prepared_source,
                rows=_read_prediction_plot_rows(csv_path),
                class_ids=loaded.checkpoint["class_mapping"]["class_ids"],
                class_names=loaded.checkpoint[
                    "class_mapping"
                ]["class_names"],
                plotting_config=settings.plotting,
                spatial_renderer=spatial_renderer,
                cut_loader=cut_loader,
            )
        counts[str(timestep)] = int(len(selected_cellids))
    return {
        "output_dir": output_dir,
        "representation": loaded.checkpoint["representation"],
        "sample_counts": counts,
    }


def _stream_timestep_predictions(
    output_path,
    prepared_source,
    cellids,
    coordinates_re,
    *,
    timestep,
    file_source,
    loaded,
    batch_size,
):
    """Prepare, infer, and stream one timestep in bounded batches.

    This inner region stage preserves the selected CID order while limiting
    the number of full raw or Hermite tensors resident at once. Each batch
    passes through representation preparation and CNN inference, then its
    decoded rows are appended immediately to the already-open timestep CSV.

    Parameters
    ----------
    output_path : str or pathlib.Path
        Destination ``predictions_<timestep>.csv`` path.
    prepared_source : PreparedPredictionSource
        Open VLSV reader and reusable representation state for this timestep.
    cellids : array-like of int
        Selected VDF-bearing cell IDs in output order.
    coordinates_re : numpy.ndarray
        Selected cell centres with shape ``(n_cells, 3)`` in Earth radii.
    timestep : int
        Simulation timestep shared by the rows.
    file_source : str
        Configured source-template identifier written to each row.
    loaded : LoadedCnnCheckpoint
        CNN and training-derived preprocessing used for inference.
    batch_size : int
        Maximum number of representation tensors prepared per inference call.
    """

    with open_prediction_csv(output_path, loaded) as output:
        for start in range(0, len(cellids), batch_size):
            end = min(start + batch_size, len(cellids))
            samples = [
                prepare_prediction_input(
                    prepared_source,
                    int(cid),
                    loaded,
                )
                for cid in cellids[start:end]
            ]
            tensors = np.stack(
                [sample.tensor for sample in samples],
                axis=0,
            )
            plasma_context = np.stack(
                [sample.plasma_context for sample in samples],
                axis=0,
            )
            predictions = run_cnn_prediction(
                loaded,
                tensors,
                plasma_context,
            )
            rows = create_prediction_rows(
                predictions,
                representation=loaded.checkpoint["representation"],
                timestep=timestep,
                file_source=file_source,
                cellids=cellids[start:end],
                coordinates_re=coordinates_re[start:end],
                loaded=loaded,
            )
            output.write_rows(rows)


def _read_prediction_plot_rows(path):
    """Read only the saved fields needed by a combined region figure.

    Plot construction follows CSV saving and must not rerun inference.
    Reading this bounded column subset preserves the saved row order while
    avoiding a second in-memory prediction table.

    Parameters
    ----------
    path : str or pathlib.Path
        Saved per-timestep prediction CSV.

    Returns
    -------
    list of dict
        String-valued plot records in their original CSV order.
    """

    required = (
        "cid",
        "selected_x_re",
        "selected_y_re",
        "selected_z_re",
        "predicted_class_id",
        "predicted_class_name",
        "predicted_probability",
        "predicted_distance_to_x_point_re",
        "predicted_distance_to_o_point_re",
    )
    with Path(path).open("r", newline="", encoding="utf-8") as input_file:
        reader = csv.DictReader(input_file)
        return [
            {column: row[column] for column in required}
            for row in reader
        ]


__all__ = ["run_region_prediction"]
