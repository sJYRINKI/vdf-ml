"""Render deterministic autoencoder reconstruction examples.

This focused output module follows final evaluation of the best
validation-selected autoencoder. It selects a compact set of train,
validation, and test examples from the in-memory normalized-space losses,
reconstructs only those memory-mapped samples, and writes one consolidated
``reconstruction_examples.png`` figure.

Raw panels restore the current middle-``vy`` log10 input to physical VDF
density, then use the same preparation and drawing operations as extraction
Stage 6. Hermite panels retain the central third-axis plane of the signed
physical-VDF coefficient cube. Original and reconstructed values share a
color scale, while a third panel shows absolute error in the same units.
"""

from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.autoencoder.step_01_load_autoencoder_data import (
    restore_autoencoder_raw_plane,
)
from src.data.load_velocity_grid import load_velocity_grid
from src.plotting.vdf_slices import (
    create_physical_vdf_plane,
    draw_vdf_plane,
    format_vdf_plane_axes,
    get_vdf_plot_velocity_cell_size_ms,
    load_vdf_plotting_thresholds,
    prepare_physical_vdf_plane,
)


def save_autoencoder_reconstruction_figure(
    output_path,
    *,
    model,
    data,
    evaluations,
    device,
    max_per_class,
    dpi,
):
    """Save original, reconstructed, and absolute-error examples.

    The restored best model reconstructs only the compact deterministic
    sample set selected from final train, validation, and test losses. For raw
    examples, source VLSV paths and CIDs are passed to the extraction Stage 6
    threshold loader, which groups rows by source and reads the supported
    physical producer once for each displayed cell. Log10 values are restored
    to physical phase-space density and rendered with that threshold,
    logarithmic scale, transparent mask, white background, and velocity axes.
    Hermite values remain denormalized signed physical-VDF coefficients on
    the central third-axis plane and require no source threshold read.

    Parameters
    ----------
    output_path : str or pathlib.Path
        PNG path receiving the consolidated reconstruction figure.
    model : VdfAutoencoder
        Best validation-selected raw or Hermite autoencoder.
    data : AutoencoderTrainingData
        Memory-mapped representation and sample-aligned physical metadata.
    evaluations : mapping
        Final split results containing sample indices and normalized-space
        reconstruction MSE values.
    device : str or torch.device
        Device hosting the restored autoencoder.
    max_per_class : int
        Historical maximum number of deterministic examples displayed for
        each physical class in each split.
    dpi : int
        Figure resolution in dots per inch.

    Returns
    -------
    pathlib.Path
        Saved ``reconstruction_examples.png`` path.

    Notes
    -----
    Physical class names affect only deterministic grouping and titles. They
    are never passed to model inference, loss calculation, or checkpoint
    selection. Source VLSV reads are limited to selected raw examples; model
    training, evaluation, and Hermite plotting do not open those sources. The
    operation reads only selected representation rows and does not persist
    reconstruction arrays.
    """

    selected = select_autoencoder_reconstruction_examples(
        evaluations,
        data.sample_identity,
        max_per_class,
    )
    if data.representation == "raw":
        selected["min_value"] = load_vdf_plotting_thresholds(selected)
    examples = _reconstruct_selected_examples(
        model,
        data,
        selected,
        device,
    )
    figure = _create_reconstruction_figure(examples, data)
    output_path = Path(output_path)
    figure.savefig(output_path, dpi=int(dpi))
    plt.close(figure)
    return output_path


def select_autoencoder_reconstruction_examples(
    evaluations,
    sample_identity,
    max_per_class,
):
    """Select compact examples in the established split and class order.

    Historical autoencoder figures traversed train, validation, and test in
    that order, ordered train rows by sample index, and ordered validation
    and test rows by decreasing reconstruction loss. Within each split,
    physical classes were traversed alphabetically and each contributed up
    to the configured historical per-class maximum. The consolidated figure
    preserves that exact ordering and count policy.

    Parameters
    ----------
    evaluations : mapping
        Final split results containing aligned ``sample_indices`` and
        ``sample_mse`` arrays.
    sample_identity : pandas.DataFrame
        Current sample index, timestep, CID, source VLSV path, class ID, and
        class name. Source identity is used only after raw examples are
        selected to read their physical sparsity thresholds.
    max_per_class : int
        Maximum number of examples retained for each physical class in each
        chronological split.

    Returns
    -------
    pandas.DataFrame
        Selected rows in deterministic display order with split, identity,
        physical class, and normalized-space reconstruction MSE columns.
    """

    selected = []
    for split_name in ("train", "validation", "test"):
        result = evaluations[split_name]
        rows = pd.DataFrame(
            {
                "split": split_name,
                "sample_index": result["sample_indices"],
                "reconstruction_mse": result["sample_mse"],
            }
        ).merge(
            sample_identity,
            on="sample_index",
            how="left",
            sort=False,
        )
        sort_column = (
            "sample_index"
            if split_name == "train"
            else "reconstruction_mse"
        )
        rows = rows.sort_values(
            sort_column,
            ascending=split_name == "train",
            kind="stable",
        )
        selected.extend(
            class_rows.head(int(max_per_class))
            for _, class_rows in rows.groupby("class_name", sort=True)
        )
    return pd.concat(selected, ignore_index=True)


def _reconstruct_selected_examples(model, data, selected, device):
    """Reconstruct only selected memory-mapped representation rows.

    Parameters
    ----------
    model : VdfAutoencoder
        Best validation-selected model holding training normalization.
    data : AutoencoderTrainingData
        Memory-mapped representation description.
    selected : pandas.DataFrame
        Deterministically selected sample identity and reporting rows.
    device : str or torch.device
        Device hosting the model and one-sample inference tensors.

    Returns
    -------
    list of dict
        Selected metadata plus two-dimensional original and reconstructed
        display planes in physical raw units or denormalized Hermite units.
    """

    device = torch.device(device)
    reader = data.create_reader()
    examples = []
    model.eval()
    with torch.inference_mode():
        for row in selected.itertuples(index=False):
            original = reader.read(int(row.sample_index))
            inputs = torch.from_numpy(
                original[np.newaxis, np.newaxis, ...]
            ).to(device)
            output = model(inputs)
            reconstructed = model.denormalize_reconstruction(
                output["reconstruction"]
            )[0, 0].detach().cpu().numpy()
            if data.representation == "raw":
                original = restore_autoencoder_raw_plane(original)
                reconstructed = restore_autoencoder_raw_plane(reconstructed)
            else:
                original = _display_hermite_plane(original)
                reconstructed = _display_hermite_plane(reconstructed)
            examples.append(
                {
                    "row": row,
                    "original": original,
                    "reconstructed": reconstructed,
                }
            )
    reader.close()
    return examples


def _create_reconstruction_figure(examples, data):
    """Create one three-column figure from denormalized examples.

    Parameters
    ----------
    examples : sequence of mapping
        Original and reconstructed two-dimensional planes plus reporting
        identity for each selected sample.
    data : AutoencoderTrainingData
        Representation and velocity-grid identity controlling axes, units,
        and color conventions.

    Returns
    -------
    matplotlib.figure.Figure
        Caller-owned figure with original, reconstructed, and absolute-error
        panels for every selected example.
    """

    figure, axes = plt.subplots(
        len(examples),
        3,
        squeeze=False,
        figsize=(12.0, 3.5 * len(examples)),
        constrained_layout=True,
    )
    velocity_grid = (
        load_velocity_grid(data.dataset_dir)
        if data.representation == "raw"
        else None
    )
    for plot_row, example in enumerate(examples):
        if data.representation == "raw":
            _draw_raw_reconstruction_row(
                figure,
                axes[plot_row],
                example,
                data,
                velocity_grid,
            )
        else:
            _draw_hermite_reconstruction_row(
                figure,
                axes[plot_row],
                example,
                data,
            )
    return figure


def _draw_raw_reconstruction_row(figure, axes, example, data, velocity_grid):
    """Render raw reconstruction examples with the extraction VDF style.

    Raw originals and reconstructions are restored from the autoencoder's
    normalized log10 representation and displayed through the same physical
    VDF preparation and drawing functions used by extraction Stage 6. The
    live autoencoder input is already a two-dimensional ``(vx, vz)`` target,
    so its fixed middle-``vy`` plane supplies physical velocity axes and the
    sample threshold while the original supplies the logarithmic color range
    shared by both panels.

    Parameters
    ----------
    figure : matplotlib.figure.Figure
        Consolidated reconstruction figure receiving row colorbars.
    axes : sequence of matplotlib.axes.Axes
        Original, reconstruction, and absolute-error axes.
    example : mapping
        Physical original and reconstruction planes with shape ``(vx, vz)``
        and density units ``s^3 m^-6``, plus reporting identity whose
        ``min_value`` field is the same-cell sparsity threshold in those
        units.
    data : AutoencoderTrainingData
        Raw model-plane convention and saved velocity-grid identity.
    velocity_grid : mapping
        Physical velocity extents in metres per second and regular grid shape.

    Returns
    -------
    tuple
        Matplotlib artists for the original, reconstruction, and error panels.

    Notes
    -----
    Masked and below-threshold values remain transparent over white axes.
    The original and reconstruction use the same fixed middle-``vy`` model
    plane, physical ``min_value`` threshold, and exact original-derived
    logarithmic normalization; a separate nonnegative panel shows absolute
    physical-space error. Historical code was inspected only as a read-only
    visual reference, while current combined-artifact and data owners remain
    authoritative. Plotting does not import legacy code or modify model
    tensors, saved VDFs, normalized-space reconstruction loss, or checkpoint
    selection.
    """

    original = example["original"]
    reconstructed = example["reconstructed"]
    peak_vx, peak_vz = np.unravel_index(
        int(np.argmax(original)),
        original.shape,
    )
    peak_index = (
        int(peak_vx),
        int(data.representation_metadata["slice_index"]),
        int(peak_vz),
    )
    original_plane = create_physical_vdf_plane(
        "xz",
        original,
        peak_index,
        velocity_grid["extent_mps"],
    )
    reconstructed_plane = create_physical_vdf_plane(
        "xz",
        reconstructed,
        peak_index,
        velocity_grid["extent_mps"],
    )
    threshold = float(example["row"].min_value)
    velocity_cell_size_ms = get_vdf_plot_velocity_cell_size_ms(velocity_grid)
    original_prepared = prepare_physical_vdf_plane(
        original_plane.values,
        threshold,
        velocity_cell_size_ms,
    )
    reconstructed_prepared = prepare_physical_vdf_plane(
        reconstructed_plane.values,
        threshold,
        velocity_cell_size_ms,
    )
    reconstructed_prepared = replace(
        reconstructed_prepared,
        norm=original_prepared.norm,
    )
    original_image = draw_vdf_plane(
        axes[0],
        original_plane,
        prepared=original_prepared,
    )
    reconstructed_image = draw_vdf_plane(
        axes[1],
        reconstructed_plane,
        prepared=reconstructed_prepared,
    )
    for axis in axes:
        axis.set_facecolor("white")
    axes[0].set_title(
        "Original\n" + _row_label(example["row"], "raw")
    )
    axes[1].set_title("Reconstructed")
    absolute_error = np.abs(reconstructed - original)
    visible_error = np.ma.masked_less_equal(absolute_error, 0.0)
    error_image = axes[2].imshow(
        visible_error.T,
        origin="lower",
        extent=original_image.get_extent(),
        aspect="equal",
        cmap="magma",
        vmin=0.0,
        vmax=float(absolute_error.max()),
    )
    format_vdf_plane_axes(axes[2], original_plane)
    axes[2].set_title("Absolute error")
    figure.colorbar(reconstructed_image, ax=axes[:2], label="f(v)")
    figure.colorbar(error_image, ax=axes[2], label="Absolute f(v)")
    return original_image, reconstructed_image, error_image


def _draw_hermite_reconstruction_row(figure, axes, example, data):
    """Draw signed Hermite coefficients and absolute reconstruction error.

    Parameters
    ----------
    figure : matplotlib.figure.Figure
        Consolidated reconstruction figure receiving row colorbars.
    axes : sequence of matplotlib.axes.Axes
        Original, reconstruction, and absolute-error axes.
    example : mapping
        Central coefficient planes plus reporting identity.
    data : AutoencoderTrainingData
        Hermite coefficient shape used for image extents.

    Returns
    -------
    tuple
        Matplotlib artists for the original, reconstruction, and error panels.
    """

    original = example["original"]
    reconstructed = example["reconstructed"]
    absolute_error = np.abs(reconstructed - original)
    value_limit = max(
        float(np.abs(original).max()),
        float(np.abs(reconstructed).max()),
    )
    extent = [
        -0.5,
        data.input_shape[0] - 0.5,
        -0.5,
        data.input_shape[1] - 0.5,
    ]
    images = []
    for column, (values, title) in enumerate(
        ((original, "Original"), (reconstructed, "Reconstructed"))
    ):
        image = axes[column].imshow(
            values,
            origin="lower",
            extent=extent,
            aspect="equal",
            cmap="RdBu_r",
            vmin=-value_limit,
            vmax=value_limit,
        )
        _format_hermite_axis(
            axes[column],
            rotated=data.representation_metadata["hermite_rotate"],
        )
        axes[column].set_title(title)
        images.append(image)
    error_image = axes[2].imshow(
        absolute_error,
        origin="lower",
        extent=extent,
        aspect="equal",
        cmap="magma",
        vmin=0.0,
        vmax=float(absolute_error.max()),
    )
    _format_hermite_axis(
        axes[2],
        rotated=data.representation_metadata["hermite_rotate"],
    )
    axes[2].set_title("Absolute error")
    figure.colorbar(images[-1], ax=axes[:2], label="Hermite coefficient")
    figure.colorbar(
        error_image,
        ax=axes[2],
        label="Absolute Hermite coefficient",
    )
    axes[0].set_title(
        "Original\n" + _row_label(example["row"], "hermite")
    )
    return images[0], images[1], error_image


def _display_hermite_plane(values):
    """Return the central signed Hermite plane in image orientation.

    Parameters
    ----------
    values : numpy.ndarray
        Complete coefficient cube. Unrotated order is
        ``(n_x, n_y, n_z)``; rotated order is
        ``(n_parallel, n_perp1, n_perp2)``.

    Returns
    -------
    numpy.ndarray
        Central third-axis coefficient plane transposed into Matplotlib
        image orientation. The selected index is derived from the actual
        saved shape.
    """

    values = np.asarray(values)
    return values[:, :, values.shape[2] // 2].T


def _format_hermite_axis(axis, *, rotated):
    """Apply signed-coefficient labels, white background, and grey grid.

    Parameters
    ----------
    axis : matplotlib.axes.Axes
        Hermite original, reconstruction, or absolute-error axes.
    rotated : bool
        Whether coefficients use ``(parallel, perp1, perp2)`` rather than
        unrotated ``(x, y, z)`` Hermite-order axes.
    """

    axis.set_facecolor("white")
    if rotated:
        axis.set_xlabel("n_parallel")
        axis.set_ylabel("n_perp1")
    else:
        axis.set_xlabel("n_x")
        axis.set_ylabel("n_y")
    axis.grid(color="gray", axis="both")


def _row_label(row, representation):
    """Describe one selected split, sample, CID, and physical class.

    Parameters
    ----------
    row : object
        Selected row exposing split, sample index, timestep, CID, and class
        name attributes.
    representation : {"raw", "hermite"}
        Representation used to identify the central Hermite plane.

    Returns
    -------
    str
        Compact title containing only post-inference reporting metadata.
    """

    label = (
        f"{row.split} | sample={int(row.sample_index)} | "
        f"t={int(row.timestep)}\n"
        f"cid={int(row.cid)} | class={row.class_name}"
    )
    if representation == "hermite":
        label += " | central n_parallel"
    return label


__all__ = [
    "save_autoencoder_reconstruction_figure",
    "select_autoencoder_reconstruction_examples",
]
