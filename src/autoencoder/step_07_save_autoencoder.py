"""Stage 7: save the autoencoder and reconstruction metrics report.

This final stage follows evaluation of the restored validation-selected
model. It writes the version-1 reconstruction checkpoint, one human-readable
``metrics.txt`` report, the established per-epoch training history, and one
consolidated reconstruction-example figure.

The report describes the run and architecture, chronological partitions,
and normalized-space reconstruction MSE by split and current physical class.
No sample-level metric array, JSON report, or class-conditioned model output
is persisted.
"""

from pathlib import Path

import pandas as pd
import torch

from src.autoencoder.step_01_load_autoencoder_data import (
    AUTOENCODER_RAW_XZ_FULL,
)
from src.autoencoder.step_06_evaluate_autoencoder import (
    create_autoencoder_reconstruction_tables,
)
from src.autoencoder.reconstruction_plots import (
    save_autoencoder_reconstruction_figure,
)
from src.data.metadata_columns import HERMITE


AUTOENCODER_CHECKPOINT_VERSION = 1


def save_autoencoder_outputs(
    output_dir,
    *,
    model,
    input_scaler,
    data,
    config,
    fit_result,
    evaluations,
    split,
    selected_device,
):
    """Save the checkpoint, metrics, history, and reconstruction figure.

    This final side-effecting stage writes the version-1 inference
    checkpoint, aggregated normalized-space MSE tables, one CSV row per
    completed optimizer epoch, and deterministic original/reconstructed/error
    panels from the restored best model. Physical labels are introduced only
    while grouping final sample losses and titling the figure.

    Parameters
    ----------
    output_dir : str or pathlib.Path
        Directory receiving the four final autoencoder files.
    model : VdfAutoencoder
        Validation-selected reconstruction model.
    input_scaler : InputFeatureScaler
        Training-only feature normalization.
    data : AutoencoderTrainingData
        Loaded representation identity.
    config : mapping
        Successful-path training configuration.
    fit_result : AutoencoderFitResult
        Selected epoch and history.
    evaluations : mapping
        Final train, validation, and test reconstruction results from the
        restored validation-selected model, including in-memory sample MSE.
    split : CnnTimestepSplit
        Chronological sample indices, timestep partitions, and excluded
        boundary gaps.
    selected_device : str or torch.device
        Effective device used for optimization and final reconstruction.

    Returns
    -------
    dict
        Checkpoint, report, and reconstruction-figure paths, representation,
        selected epoch, epoch count, and final train/validation/test
        normalized-space MSE values.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = save_autoencoder_checkpoint(
        model,
        output_dir / "autoencoder.pt",
        input_scaler=input_scaler,
        representation_metadata=data.representation_metadata,
        random_seed=config["random_state"],
    )
    split_metrics, class_metrics = (
        create_autoencoder_reconstruction_tables(
            evaluations,
            data.sample_identity,
        )
    )
    metrics_path = output_dir / "metrics.txt"
    metrics_path.write_text(
        create_autoencoder_metrics_text(
            output_dir,
            model=model,
            input_scaler=input_scaler,
            data=data,
            config=config,
            fit_result=fit_result,
            evaluations=evaluations,
            split=split,
            split_metrics=split_metrics,
            class_metrics=class_metrics,
            selected_device=selected_device,
        ),
        encoding="utf-8",
    )
    training_history_path = output_dir / "training_history.csv"
    pd.DataFrame(fit_result.history).to_csv(
        training_history_path,
        index=False,
    )
    reconstruction_figure_path = save_autoencoder_reconstruction_figure(
        output_dir / "reconstruction_examples.png",
        model=model,
        data=data,
        evaluations=evaluations,
        device=selected_device,
        max_per_class=config["plot"]["max_per_class"],
        dpi=config["plot"]["dpi"],
    )
    return {
        "output_dir": str(output_dir),
        "checkpoint_path": str(checkpoint_path),
        "metrics_path": str(metrics_path),
        "training_history_path": str(training_history_path),
        "reconstruction_figure_path": str(reconstruction_figure_path),
        "representation": data.representation,
        "best_epoch": fit_result.best_epoch,
        "epochs_completed": fit_result.epochs_completed,
        "train_reconstruction_mse": evaluations["train"]["mse"],
        "validation_reconstruction_mse": (
            evaluations["validation"]["mse"]
        ),
        "test_reconstruction_mse": evaluations["test"]["mse"],
    }


def create_autoencoder_metrics_text(
    output_dir,
    *,
    model,
    input_scaler,
    data,
    config,
    fit_result,
    evaluations,
    split,
    split_metrics,
    class_metrics,
    selected_device,
):
    """Create the autoencoder model and reconstruction metrics report.

    The report describes the best validation-selected autoencoder and
    summarizes reconstruction error from its unchanged normalized-space MSE
    objective. Split statistics show train, validation, and test behavior,
    while class statistics reveal which physical VDF populations are easier
    or harder to reconstruct.

    Physical class labels are used only after reconstruction for reporting.
    They do not enter the autoencoder input, loss, optimizer, or checkpoint
    selection.

    Parameters
    ----------
    output_dir : str or pathlib.Path
        Training-artifact directory receiving ``autoencoder.pt``,
        ``metrics.txt``, ``training_history.csv``, and
        ``reconstruction_examples.png``.
    model : VdfAutoencoder
        Best validation-selected raw or Hermite autoencoder.
    input_scaler : InputFeatureScaler
        Training-only per-feature population mean and standard deviation.
    data : AutoencoderTrainingData
        Dataset identity, source-array shape, and reporting metadata.
    config : mapping
        Resolved architecture, optimizer, loader, and split settings.
    fit_result : AutoencoderFitResult
        Current optimization history and selected epoch.
    evaluations : mapping
        Final restored-model reconstruction results for every split.
    split : CnnTimestepSplit
        Chronological partition timesteps and sample indices.
    split_metrics : pandas.DataFrame
        Count and normalized-space MSE statistics by partition.
    class_metrics : pandas.DataFrame
        Count and normalized-space MSE statistics by physical class.
    selected_device : str or torch.device
        Effective PyTorch device used by this training run.

    Returns
    -------
    str
        UTF-8 report text ending with one newline.

    Notes
    -----
    The report contains aggregated tables only. It does not persist
    sample-level reconstruction errors or use class labels to modify model
    behavior. Raw MSE describes the normalized middle-``vy`` log plane;
    Hermite MSE describes the normalized complete coefficient volume.
    """

    output_dir = Path(output_dir)
    spec = data.representation_spec
    class_names = (
        data.sample_identity.loc[:, ["class_id", "class_name"]]
        .drop_duplicates()
        .sort_values("class_id", kind="stable")["class_name"]
        .tolist()
    )
    encoder_channels = (
        model.raw_channels
        if model.representation == "raw"
        else model.hermite_channels
    )
    decoder_channels = (*reversed(encoder_channels), 1)
    selected_history = fit_result.history[fit_result.best_epoch - 1]
    final_history = fit_result.history[-1]
    separator = "=" * 70
    representation_details = []
    if data.representation == "hermite":
        representation_details = [
            "Hermite quantity: physical-VDF Hermite coefficients",
            f"Hermite volume shape: {spec.tensor_shape}",
            "Hermite order: "
            f"{data.representation_metadata['hermite_order']}",
            "Hermite rotated: "
            f"{data.representation_metadata['hermite_rotate']}",
        ]
    lines = [
        "Autoencoder training run",
        separator,
        f"Dataset directory: {data.dataset_dir}",
        f"Output directory: {output_dir}",
        f"Representation: {data.representation}",
        f"Source array: {data.source_path}",
        f"Source array shape: {spec.source_shape}",
        f"Source array dtype: {spec.source_dtype}",
        *representation_details,
        f"Samples: {len(data)}",
        f"Unique timesteps: {data.sample_identity['timestep'].nunique()}",
        f"Unique cell IDs: {data.sample_identity['cid'].nunique()}",
        f"Physical class names: {', '.join(class_names)}",
        f"Checkpoint version: {AUTOENCODER_CHECKPOINT_VERSION}",
        "Saved checkpoint filename: autoencoder.pt",
        f"Device: {selected_device}",
        f"PyTorch version: {torch.__version__}",
        f"Random seed: {config['random_state']}",
        f"Raw positive log floor: {config['raw']['log_eps']}",
        f"Data-loader batch size: {config['data_loader']['batch_size']}",
        "Normalization sample batch size: "
        f"{config['data_loader']['normalization_batch_size']}",
        f"Data-loader workers: {config['data_loader']['num_workers']}",
        f"Pinned memory: {config['data_loader']['pin_memory']}",
        "Optimizer: AdamW",
        f"Learning rate: {config['optimizer']['learning_rate']}",
        f"Weight decay: {config['optimizer']['weight_decay']}",
        f"AdamW betas: {tuple(config['optimizer']['betas'])}",
        f"AdamW epsilon: {config['optimizer']['epsilon']}",
        f"Maximum epochs: {config['training']['max_epochs']}",
        f"Early-stopping patience: {config['training']['patience']}",
        f"Minimum validation-MSE decrease: {config['training']['min_delta']}",
        "",
        "Model architecture",
        separator,
        f"Input shape: {model.input_shape}",
        f"Encoder channels: {encoder_channels}",
        f"Decoder channels: {decoder_channels}",
        f"Latent dimension: {model.latent_dim}",
        f"Pooling: {model.pooling}",
        "Trainable parameters: "
        f"{sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)}",
        "Normalization convention: training-only per-feature population "
        "mean and standard deviation in normalized representation space",
        f"Normalization epsilon: {input_scaler.epsilon}",
        "Reconstruction objective: mean squared error in normalized "
        "representation space",
        "",
        "Dataset partitions",
        separator,
        f"Configured train fraction: {config['split']['train_fraction']}",
        "Configured validation fraction: "
        f"{config['split']['validation_fraction']}",
        f"Gap timesteps per boundary: {config['split']['gap_timesteps']}",
        f"Train timestep count: {len(split.train_timesteps)}",
        f"Validation timestep count: {len(split.validation_timesteps)}",
        f"Test timestep count: {len(split.test_timesteps)}",
        "Train-validation gap timestep count: "
        f"{len(split.train_validation_gap)}",
        "Validation-test gap timestep count: "
        f"{len(split.validation_test_gap)}",
        f"Train sample count: {len(split.train_indices)}",
        f"Validation sample count: {len(split.validation_indices)}",
        f"Test sample count: {len(split.test_indices)}",
        "",
        "Reconstruction loss by split",
        separator,
        split_metrics.to_string(
            index=False,
            float_format=lambda value: f"{value:.8e}",
        ),
        "",
        "Reconstruction loss by physical class",
        separator,
        class_metrics.to_string(
            index=False,
            float_format=lambda value: f"{value:.8e}",
        ),
        "",
        "Training history summary",
        separator,
        "Selection metric: validation reconstruction MSE",
        f"Epochs completed: {fit_result.epochs_completed}",
        f"Best epoch: {fit_result.best_epoch}",
        f"Training seconds: {fit_result.total_seconds}",
        "Best-epoch train reconstruction loss: "
        f"{selected_history['train_reconstruction_mse']}",
        "Best-epoch validation reconstruction loss: "
        f"{selected_history['validation_reconstruction_mse']}",
        "Final optimization-epoch train reconstruction loss: "
        f"{final_history['train_reconstruction_mse']}",
        "Final optimization-epoch validation reconstruction loss: "
        f"{final_history['validation_reconstruction_mse']}",
        "Final train reconstruction loss: "
        f"{evaluations['train']['mse']}",
        "Final validation reconstruction loss: "
        f"{evaluations['validation']['mse']}",
        "Final test reconstruction loss: "
        f"{evaluations['test']['mse']}",
    ]
    return "\n".join(lines).rstrip() + "\n"


def save_autoencoder_checkpoint(
    model,
    checkpoint_path,
    *,
    input_scaler,
    representation_metadata,
    random_seed,
):
    """Save one optimizer-independent autoencoder checkpoint.

    The checkpoint loader reconstructs the architecture and feature
    normalization directly from this dictionary before restoring learned
    tensors. Optimizer state is intentionally absent because the file is
    intended for reconstruction, not resumed training.

    Parameters
    ----------
    model : VdfAutoencoder
        Trained raw or Hermite reconstruction model.
    checkpoint_path : str or pathlib.Path
        Destination checkpoint file.
    input_scaler : InputFeatureScaler
        Training-derived feature normalization.
    representation_metadata : mapping
        Saved representation description. Hermite checkpoints retain the
        actual coefficient volume shape, physical mode order, and optional
        rotation setting derived from ``X_hermite.npy`` and aligned dataset
        metadata.
    random_seed : int
        Training random seed.

    Returns
    -------
    pathlib.Path
        Saved checkpoint path.
    """

    checkpoint = {
        "autoencoder_checkpoint_version": (
            AUTOENCODER_CHECKPOINT_VERSION
        ),
        "model_state_dict": {
            name: value.detach().cpu()
            for name, value in model.state_dict().items()
        },
        "representation": model.representation,
        "representation_version": (
            AUTOENCODER_RAW_XZ_FULL
            if model.representation == "raw"
            else HERMITE
        ),
        "expected_input_shape": list(model.input_shape),
        "input_normalization": input_scaler.to_dict(),
        "model_architecture": model.constructor_config(),
        "random_seed": int(random_seed),
    }
    if model.representation == "hermite":
        checkpoint.update(
            {
                "volume_shape": list(model.input_shape),
                "hermite_order": int(model.input_shape[0]),
                "hermite_rotate": bool(
                    representation_metadata["hermite_rotate"]
                ),
            }
        )
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, checkpoint_path)
    return checkpoint_path


__all__ = [
    "AUTOENCODER_CHECKPOINT_VERSION",
    "create_autoencoder_metrics_text",
    "save_autoencoder_checkpoint",
    "save_autoencoder_outputs",
]
