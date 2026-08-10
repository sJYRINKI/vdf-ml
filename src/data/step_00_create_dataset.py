"""Stage 0: orchestrate dataset creation and optional visual outputs.

This public stage is called by the extraction CLI. It invokes VLSV opening,
VDF-cell discovery, sample planning, physical VDF extraction, and staged
file writing in numeric stage order. When configured, it then calls Stage 6
for dataset frames and Stage 7 for time-evolution animations.

Inputs are the loaded extraction configuration and timestep range. The
returned path contains aligned raw, optional Hermite, metadata, and
velocity-grid files plus optional plot and animation directories.
"""

from functools import partial
import os
from pathlib import Path
import time

import numpy as np

from src.data.metadata import create_extraction_metadata
from src.data.step_01_open_vlsv_files import create_timestep_list
from src.data.step_03_plan_dataset_samples import (
    count_sample_counts,
    count_sample_specs_by_timestep,
    extract_first_sample_from_specs,
    find_first_nonempty_timestep,
    get_planned_velocity_grid,
    plan_dataset_sample_specs,
)
from src.data.step_04_extract_vdf_samples import (
    iter_timestep_sample_specs,
    write_extracted_sample_hermite,
    write_remaining_timesteps,
    write_timestep_samples,
)
from src.data.step_05_write_dataset_files import write_dataset
from src.physics.hermite_basis import DEFAULT_HERMITE_ORDER


def create_dataset(
    config,
    start_timestep,
    n_timesteps,
    dataset_kind,
    output_suffix=None,
):
    """Create one raw or raw-plus-Hermite dataset.

    This public orchestration stage plans samples, determines the shared
    velocity grid from the first nonempty timestep, streams raw and optional
    Hermite rows into staged arrays, and delegates final file ownership to
    stage 5. Optional frame rendering and animation begin only after that
    stage renames the completed core dataset. The function writes no
    representation preprocessing used by ML consumers.

    Parameters
    ----------
    config : dict
        Effective extraction configuration.
    start_timestep : int
        First simulation timestep.
    n_timesteps : int
        Number of consecutive timesteps to plan.
    dataset_kind : {"train", "test"}
        Configured output-directory key.
    output_suffix : str, optional
        Additional descriptive suffix for the final dataset directory.

    Returns
    -------
    pathlib.Path
        New dataset directory containing the current core files and any
        enabled plot or animation directories.

    """

    total_start = time.perf_counter()
    timesteps = create_timestep_list(
        start_timestep=start_timestep,
        n_timesteps=n_timesteps,
    )
    storage_config = resolve_extraction_storage_config(config)
    outdir = _resolve_dataset_output_dir(
        output_dir=config["output_dirs"][dataset_kind],
        start_timestep=start_timestep,
        n_timestep=n_timesteps,
        name_suffix=output_suffix,
    )

    print(f"Dataset kind: {dataset_kind}")
    print(f"Output directory: {outdir}")
    creation_config = config.get("creation", {})
    default_n_jobs = int(creation_config.get("n_jobs", 1))
    planning_n_jobs = int(
        creation_config.get("planning_n_jobs", default_n_jobs)
    )
    extraction_n_jobs = int(
        creation_config.get("extraction_n_jobs", default_n_jobs)
    )
    planning_worker_count = _worker_count(planning_n_jobs)
    extraction_worker_count = _worker_count(extraction_n_jobs)
    print(
        f"Planning jobs: {planning_n_jobs} "
        f"({planning_worker_count} workers)"
    )
    print(
        f"Extraction jobs: {extraction_n_jobs} "
        f"({extraction_worker_count} workers)"
    )

    (
        sample_specs_by_timestep,
        plot_data_by_timestep,
        planning_elapsed,
    ) = plan_dataset_sample_specs(
        config=config,
        timesteps=timesteps,
        planning_n_jobs=planning_n_jobs,
    )
    sample_counts_by_timestep = count_sample_specs_by_timestep(
        sample_specs_by_timestep
    )
    n_samples = count_sample_counts(sample_counts_by_timestep)
    print(f"Samples: {n_samples}")
    print(f"Timing planning: {planning_elapsed:.2f} s")
    velocity_grid = get_planned_velocity_grid(sample_specs_by_timestep)
    extraction_start = time.perf_counter()
    metadata = []
    sample_index = 0
    hermite_enabled = storage_config["hermite_enabled"]
    hermite_rotate = storage_config["hermite_rotate"]
    first_timestep_index, first_timestep = find_first_nonempty_timestep(
        sample_counts_by_timestep=sample_counts_by_timestep,
        timesteps=timesteps,
    )
    first_sample_specs = sample_specs_by_timestep[first_timestep]
    first_sample, first_sample_iter = extract_first_sample_from_specs(
        first_sample_specs,
        include_rotation_context=hermite_enabled and hermite_rotate,
    )
    first_sample_shape = tuple(
        int(value) for value in first_sample["vdf"].shape
    )
    raw_write_elapsed = 0.0

    def write_extraction_samples(raw_output, hermite_output=None):
        """Stream all planned rows into aligned staged representation sinks.

        Stage 5 calls this closure after allocating the final-shape memory
        maps. It writes the already-extracted first timestep, streams the
        remaining timesteps in order, and returns a metadata frame whose row
        indices match both writable arrays.
        """

        nonlocal first_sample
        nonlocal metadata
        nonlocal sample_index
        nonlocal raw_write_elapsed
        raw_write_start = time.perf_counter()

        sample_callback = (
            partial(
                write_extracted_sample_hermite,
                X_hermite=hermite_output,
                velocity_grid=velocity_grid,
                order=storage_config["hermite_order"],
                rotate=hermite_rotate,
                output_dtype=storage_config["hermite_output_dtype"],
            )
            if hermite_enabled
            else None
        )
        sample_index = write_timestep_samples(
            X=raw_output,
            metadata=metadata,
            timestep_samples=[first_sample],
            sample_index=sample_index,
            sample_callback=sample_callback,
        )
        first_sample = None
        sample_index = write_timestep_samples(
            X=raw_output,
            metadata=metadata,
            timestep_samples=first_sample_iter,
            sample_index=sample_index,
            sample_callback=sample_callback,
        )
        sample_specs_by_timestep.pop(first_timestep, None)
        sample_index = write_remaining_timesteps(
            sample_specs_by_timestep=sample_specs_by_timestep,
            X=raw_output,
            metadata=metadata,
            sample_index=sample_index,
            timesteps=timesteps[first_timestep_index + 1:],
            extraction_n_jobs=extraction_n_jobs,
            extraction_worker_count=extraction_worker_count,
            include_rotation_context=hermite_enabled and hermite_rotate,
            sample_callback=(
                sample_callback if extraction_n_jobs == 1 else None
            ),
            X_hermite=hermite_output,
            velocity_grid=velocity_grid,
            hermite_order=storage_config["hermite_order"],
            hermite_rotate=hermite_rotate,
            hermite_dtype=storage_config["hermite_output_dtype"],
        )
        raw_write_elapsed = time.perf_counter() - raw_write_start
        metadata = create_extraction_metadata(metadata)
        return metadata

    writer_start = time.perf_counter()
    writer_arguments = {
        "dataset_dir": outdir,
        "raw_shape": (n_samples, *first_sample_shape),
        "raw_dtype": np.float32,
        "velocity_grid": velocity_grid,
        "sample_writer": write_extraction_samples,
    }
    if hermite_enabled:
        order = storage_config["hermite_order"]
        writer_arguments.update(
            {
                "hermite_shape": (n_samples, order, order, order),
                "hermite_dtype": storage_config["hermite_output_dtype"],
            }
        )
    result_path = write_dataset(**writer_arguments)
    writer_elapsed = time.perf_counter() - writer_start
    extraction_elapsed = (
        writer_start - extraction_start + raw_write_elapsed
    )
    finalize_elapsed = max(0.0, writer_elapsed - raw_write_elapsed)
    total_elapsed = time.perf_counter() - total_start

    print(f"X shape: {(n_samples, *first_sample_shape)}")
    print(f"Samples written: {sample_index}")
    print(f"Timing extraction/write: {extraction_elapsed:.2f} s")
    print(f"Timing finalize/rename: {finalize_elapsed:.2f} s")
    print(f"Timing total: {total_elapsed:.2f} s")
    print(f"Saved X: {result_path / 'X.npy'}")
    print(f"Saved metadata: {result_path / 'metadata.csv'}")
    print(f"Saved velocity grid: {result_path / 'velocity_grid.npz'}")
    if hermite_enabled:
        print(f"Saved Hermite: {result_path / 'X_hermite.npy'}")
    postprocessing = config.get("postprocessing", {})
    plot_colormaps = bool(postprocessing.get("plot_colormaps", False))
    plot_vdfs = bool(postprocessing.get("plot_vdfs", False))
    plot_hermites = hermite_enabled and bool(
        postprocessing.get("plot_hermites", False)
    )
    animate_colormaps = bool(
        postprocessing.get("animate_colormaps", False)
    )
    animate_vdfs = bool(postprocessing.get("animate_vdfs", False))
    if plot_colormaps or plot_vdfs or plot_hermites:
        from src.data.step_06_plot_dataset_frames import (
            plot_dataset_frames,
        )

        plot_dataset_frames(
            result_path,
            config=config,
            plot_data_by_timestep=plot_data_by_timestep,
            plot_colormaps=plot_colormaps,
            plot_vdfs=plot_vdfs,
            plot_hermites=plot_hermites,
        )
    if animate_colormaps or animate_vdfs:
        from src.data.step_07_animate_dataset_frames import (
            animate_dataset_frames,
        )

        animate_dataset_frames(
            result_path,
            animate_colormaps=animate_colormaps,
            animate_vdfs=animate_vdfs,
            animation_fps=postprocessing.get("animation_fps", 4),
            animation_format=postprocessing.get(
                "animation_format",
                "mp4",
            ),
        )
    return result_path


def run_extraction(
    config,
    start_timestep,
    n_timesteps,
    dataset_kind,
    *,
    x_selection_method=None,
    o_selection_method=None,
):
    """Apply command overrides and run current dataset extraction.

    The extraction CLI calls this adapter after loading YAML and parsing
    optional X/O selection-method overrides. It mutates the in-memory
    configuration only, derives a descriptive output suffix, and then enters
    the stage-0 dataset workflow.

    Parameters
    ----------
    config : dict
        Loaded extraction configuration modified by explicit overrides.
    start_timestep : int
        First simulation timestep.
    n_timesteps : int
        Number of consecutive timesteps.
    dataset_kind : {"train", "test"}
        Configured output-directory key.
    x_selection_method : str, optional
        X-point selection override.
    o_selection_method : str, optional
        O-point selection override.

    Returns
    -------
    pathlib.Path
        Newly renamed current dataset directory.
    """

    output_suffix = apply_dataset_creation_overrides(
        config=config,
        x_selection_method=x_selection_method,
        o_selection_method=o_selection_method,
    )
    return create_dataset(
        config=config,
        start_timestep=start_timestep,
        n_timesteps=n_timesteps,
        dataset_kind=dataset_kind,
        output_suffix=output_suffix,
    )


def apply_dataset_creation_overrides(
    config,
    x_selection_method=None,
    o_selection_method=None,
):
    """Apply optional command-line point-selection methods in place.

    CLI selection choices must affect both the physical sampling branch and
    the resulting directory name. This function updates those two point-kind
    settings in the supplied configuration and returns a stable suffix
    describing only overrides that were actually provided.

    Parameters
    ----------
    config : dict
        Effective extraction configuration modified in place.
    x_selection_method : str, optional
        X-point selection method override.
    o_selection_method : str, optional
        O-point selection method override.

    Returns
    -------
    str or None
        Stable dataset-directory suffix for applied overrides.
    """

    points_config = config.setdefault("points", {})
    suffix_parts = []
    if x_selection_method is not None:
        method = str(x_selection_method).lower()
        points_config.setdefault("x_selection", {})["method"] = method
        suffix_parts.append(f"x_{method}")
    if o_selection_method is not None:
        method = str(o_selection_method).lower()
        points_config.setdefault("o_selection", {})["method"] = method
        suffix_parts.append(f"o_{method}")
    return "_".join(suffix_parts) if suffix_parts else None


def resolve_extraction_storage_config(config):
    """Resolve optional Hermite extraction settings.

    Stage 0 always writes the physical ``X.npy`` source. This view allocates
    optional physical-VDF Hermite storage and configures the serial callback
    or timestep worker with its order, rotation setting, and one final storage
    cast. YAML scalar values are converted to the Python and NumPy types
    consumed by later stages.

    Parameters
    ----------
    config : dict
        Effective extraction configuration.

    Returns
    -------
    dict
        Hermite enablement, order, optional rotation, and saved dtype.
    """

    representations = config.get("representations", {})
    hermite_config = representations.get("hermite", False)
    hermite_enabled = (
        hermite_config
        if isinstance(hermite_config, bool)
        else hermite_config.get("enabled", False)
    )
    hermite_options = (
        hermite_config
        if isinstance(hermite_config, dict)
        else {}
    )
    return {
        "hermite_enabled": hermite_enabled,
        "hermite_order": int(
            hermite_options.get("order", DEFAULT_HERMITE_ORDER)
        ),
        "hermite_rotate": bool(hermite_options.get("rotate", False)),
        "hermite_output_dtype": np.dtype(
            hermite_options.get("output_dtype", "float32")
        ),
    }


def _worker_count(n_jobs):
    """Return the positive worker bound used for timestep chunking.

    Joblib interprets a negative ``n_jobs`` relative to the machine CPU count.
    Dataset extraction needs the concrete count separately to bound the number
    of temporary timestep arrays that can exist concurrently.
    """

    return os.cpu_count() or 1 if n_jobs < 0 else int(n_jobs)


def _resolve_dataset_output_dir(
    output_dir,
    start_timestep,
    n_timestep,
    name_suffix=None,
):
    """Construct the final dataset path from the requested timestep range.

    The name records the first timestep, number of consecutive timesteps, and
    optional point-selection suffix. Stage 5 later creates a hidden sibling
    staging directory and renames it to this path.
    """

    dirname = f"timesteps_{int(start_timestep)}_{int(n_timestep)}"
    if name_suffix is not None and str(name_suffix).strip():
        dirname = f"{dirname}_{str(name_suffix).strip()}"
    return Path(output_dir) / dirname
