"""Stage 7: save the full-volume topology-aware autoencoder artifacts.

The final stage writes the established ``autoencoder.pt``, ``metrics.txt``,
``training_history.csv``, and ``reconstruction_examples.png`` files. The
checkpoint contains complete three-dimensional architecture,
training-derived representation and topology scaling, selection metadata,
and CPU model tensors, but no runtime device identifiers or stage mapping.

The text report retains reconstruction summaries and adds combined objective,
physical topology error, and one-process model-parallel placement sections.
No JSON report or per-sample topology output is created.
"""

from pathlib import Path

import pandas as pd
import torch

from src.autoencoder.reconstruction_plots import (
    save_autoencoder_reconstruction_figure,
)
from src.autoencoder.step_06_evaluate_autoencoder import (
    create_autoencoder_objective_tables,
    create_autoencoder_reconstruction_tables,
)


def save_autoencoder_outputs(
    output_dir,
    *,
    model,
    input_scaler,
    topology_scaler,
    data,
    config,
    fit_result,
    evaluations,
    split,
    selected_device,
):
    """Save the checkpoint, metrics, history, and reconstruction figure.

    This is the sole publication owner for autoencoder training. It converts
    final in-memory evaluation results into aggregate text/CSV content and
    asks the focused plot owner to render selected full-volume examples.

    Parameters
    ----------
    output_dir : str or pathlib.Path
        Directory receiving the four established autoencoder artifacts.
    model : VdfAutoencoder
        Validation-total-loss-selected full-volume model.
    input_scaler : InputFeatureScaler
        Training-only per-voxel or per-coefficient normalization.
    topology_scaler : TopologyTargetScaler
        Training-only scaling for six Earth-radius auxiliary targets.
    data : AutoencoderTrainingData
        Memory-mapped complete representation and sample identity.
    config : mapping
        Successful run configuration including topology loss weight.
    fit_result : AutoencoderFitResult
        Selected epoch, best validation total loss, and history.
    evaluations : mapping
        Final train, validation, and test objective results.
    split : CnnTimestepSplit
        Unchanged chronological complete-timestep partitions.
    selected_device : str or torch.device
        Input device used by this run; output ownership is read from model.

    Returns
    -------
    dict
        All four artifact paths and final split objective values.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = save_autoencoder_checkpoint(
        model,
        output_dir / "autoencoder.pt",
        input_scaler=input_scaler,
        topology_scaler=topology_scaler,
        topology_loss_weight=config["topology"]["loss_weight"],
        representation_metadata=data.representation_metadata,
        random_seed=config["random_state"],
        best_epoch=fit_result.best_epoch,
        best_validation_total_loss=(
            fit_result.best_validation_total_loss
        ),
    )
    split_metrics, class_metrics = create_autoencoder_reconstruction_tables(
        evaluations,
        data.sample_identity,
    )
    objective_metrics, topology_metrics = create_autoencoder_objective_tables(
        evaluations
    )
    metrics_path = output_dir / "metrics.txt"
    metrics_path.write_text(
        create_autoencoder_metrics_text(
            output_dir,
            model=model,
            input_scaler=input_scaler,
            topology_scaler=topology_scaler,
            data=data,
            config=config,
            fit_result=fit_result,
            evaluations=evaluations,
            split=split,
            split_metrics=split_metrics,
            class_metrics=class_metrics,
            objective_metrics=objective_metrics,
            topology_metrics=topology_metrics,
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
        device=model.input_device,
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
        "best_validation_total_loss": (
            fit_result.best_validation_total_loss
        ),
        "epochs_completed": fit_result.epochs_completed,
        "train_reconstruction_mse": evaluations["train"][
            "reconstruction_mse"
        ],
        "validation_reconstruction_mse": evaluations["validation"][
            "reconstruction_mse"
        ],
        "test_reconstruction_mse": evaluations["test"][
            "reconstruction_mse"
        ],
        "train_topology_loss": evaluations["train"]["topology_loss"],
        "validation_topology_loss": evaluations["validation"][
            "topology_loss"
        ],
        "test_topology_loss": evaluations["test"]["topology_loss"],
        "train_total_loss": evaluations["train"]["total_loss"],
        "validation_total_loss": evaluations["validation"]["total_loss"],
        "test_total_loss": evaluations["test"]["total_loss"],
    }


def create_autoencoder_metrics_text(
    output_dir,
    *,
    model,
    input_scaler,
    topology_scaler,
    data,
    config,
    fit_result,
    evaluations,
    split,
    split_metrics,
    class_metrics,
    objective_metrics,
    topology_metrics,
    selected_device,
):
    """Create the architecture, objective, and physical-error report.

    The report makes representation shape, normalization, stage placement,
    target order and units, loss composition, selection, and final aggregate
    errors understandable without reopening the checkpoint or source data.

    Parameters
    ----------
    output_dir : str or pathlib.Path
        Directory containing the four final artifacts.
    model : VdfAutoencoder
        Selected complete-volume Conv3d autoencoder.
    input_scaler : InputFeatureScaler
        Training-only representation normalization.
    topology_scaler : TopologyTargetScaler
        Six-target mean and scale fitted from valid training values only.
    data : AutoencoderTrainingData
        Source-array shape, axis, and identity description.
    config : mapping
        Resolved loader, architecture, optimizer, topology, and split values.
    fit_result : AutoencoderFitResult
        Selected epoch and per-epoch objective history.
    evaluations : mapping
        Final aggregate objectives for train, validation, and test.
    split : CnnTimestepSplit
        Chronological timestep and sample partitions.
    split_metrics : pandas.DataFrame
        Existing reconstruction statistics by split.
    class_metrics : pandas.DataFrame
        Existing reconstruction statistics by physical class.
    objective_metrics : pandas.DataFrame
        Reconstruction, topology, and total objectives by split.
    topology_metrics : pandas.DataFrame
        Per-target valid count, MAE, and RMSE in Earth radii.
    selected_device : str or torch.device
        Input-stage runtime device.

    Returns
    -------
    str
        UTF-8 ``metrics.txt`` content ending with one newline.

    Notes
    -----
    Physical class remains reporting-only. Topology is auxiliary supervision
    from the shared latent vector and is never concatenated to raw voxels,
    Hermite coefficients, encoder inputs, or decoder inputs.
    """

    output_dir = Path(output_dir)
    spec = data.representation_spec
    class_names = (
        data.sample_identity.loc[:, ["class_id", "class_name"]]
        .drop_duplicates()
        .sort_values("class_id", kind="stable")["class_name"]
        .tolist()
    )
    decoder_channels = (*reversed(model.channels), 1)
    selected_history = fit_result.history[fit_result.best_epoch - 1]
    final_history = fit_result.history[-1]
    separator = "=" * 70
    stage_mapping = pd.DataFrame(
        {
            "stage": model.stage_names,
            "device": tuple(str(value) for value in model.stage_devices),
        }
    )
    representation_details = [
        "Full three-dimensional representation",
        separator,
        f"Representation: {data.representation}",
        f"Complete input volume shape: {model.input_shape}",
        "Model tensor shape: "
        f"(batch, 1, {', '.join(str(value) for value in model.input_shape)})",
        "Raw axis order: (vx, vy, vz)",
        "Hermite axis order: saved unrotated or rotated coefficient order",
        "Training slices or downsampling: none",
    ]
    if data.representation == "hermite":
        representation_details.extend(
            (
                "Hermite quantity: signed physical-VDF coefficients",
                "Hermite order: "
                f"{data.representation_metadata['hermite_order']}",
                "Hermite rotated: "
                f"{data.representation_metadata['hermite_rotate']}",
            )
        )
    lines = [
        "Autoencoder training run",
        separator,
        f"Dataset directory: {data.dataset_dir}",
        f"Output directory: {output_dir}",
        f"Source array: {data.source_path}",
        f"Source array shape: {spec.source_shape}",
        f"Source array dtype: {spec.source_dtype}",
        f"Samples: {len(data)}",
        f"Unique timesteps: {data.sample_identity['timestep'].nunique()}",
        f"Unique cell IDs: {data.sample_identity['cid'].nunique()}",
        f"Physical class names: {', '.join(class_names)}",
        "Saved artifacts: autoencoder.pt, metrics.txt, "
        "training_history.csv, reconstruction_examples.png",
        f"PyTorch version: {torch.__version__}",
        f"Random seed: {config['random_state']}",
        f"Raw positive log floor: {config['raw']['log_eps']}",
        f"Data-loader batch size: {config['data_loader']['batch_size']}",
        f"Data-loader workers: {config['data_loader']['num_workers']}",
        "Normalization sample batch size: "
        f"{config['data_loader']['normalization_batch_size']}",
        f"Pinned memory: {config['data_loader']['pin_memory']}",
        f"Optimizer: AdamW",
        f"Learning rate: {config['optimizer']['learning_rate']}",
        f"Weight decay: {config['optimizer']['weight_decay']}",
        f"AdamW betas: {tuple(config['optimizer']['betas'])}",
        f"AdamW epsilon: {config['optimizer']['epsilon']}",
        f"Maximum epochs: {config['training']['max_epochs']}",
        f"Early-stopping patience: {config['training']['patience']}",
        f"Minimum total-loss decrease: {config['training']['min_delta']}",
        "",
        *representation_details,
        "",
        "Model architecture",
        separator,
        f"Conv3d encoder channels: {model.channels}",
        f"Conv3d decoder channels: {decoder_channels}",
        "Configured bottleneck spatial maximum: "
        f"{model.configured_bottleneck_shape}",
        f"Effective bottleneck spatial shape: {model.bottleneck_shape}",
        f"Latent size: {model.latent_size}",
        f"Pooling: {model.pooling}",
        "Trainable parameters: "
        f"{sum(value.numel() for value in model.parameters() if value.requires_grad)}",
        "Representation normalization: training-only per-feature population "
        "mean and standard deviation",
        f"Representation normalization epsilon: {input_scaler.epsilon}",
        "Reconstruction objective: complete normalized-volume MSE",
        "",
        "Model-parallel stage placement",
        separator,
        "Parallelism: one process, one model, consecutive layer stages",
        "Requested model_parallel_gpus: "
        f"{model.requested_model_parallel_gpus}",
        "Effective model_parallel_gpus: "
        f"{model.effective_model_parallel_gpus}",
        f"Input device: {selected_device}",
        f"Output device: {model.output_device}",
        stage_mapping.to_string(index=False),
        "",
        "Auxiliary topology task",
        separator,
        "Topology role: latent-space auxiliary target, never model input",
        "Topology target order: "
        f"{tuple(topology_scaler.target_names)}",
        "Topology units after inverse scaling: Earth radii",
        f"Topology hidden size: {model.topology_hidden_size}",
        f"Topology loss weight: {config['topology']['loss_weight']}",
        f"Topology scaling centre: {topology_scaler.mean.tolist()}",
        f"Topology scaling scale: {topology_scaler.scale.tolist()}",
        "Topology scaling fit: finite training-partition values only",
        "Missing topology: independently masked per target",
        "Total loss: reconstruction_mse + topology_loss_weight * "
        "masked_topology_smooth_l1",
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
        f"Train sample count: {len(split.train_indices)}",
        f"Validation sample count: {len(split.validation_indices)}",
        f"Test sample count: {len(split.test_indices)}",
        "",
        "Combined objective by split",
        separator,
        objective_metrics.to_string(
            index=False,
            float_format=lambda value: f"{value:.8e}",
        ),
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
        "Topology error by split and target",
        separator,
        topology_metrics.to_string(
            index=False,
            float_format=lambda value: f"{value:.8e}",
        ),
        "",
        "Training history summary",
        separator,
        "Selection metric: validation total loss",
        f"Epochs completed: {fit_result.epochs_completed}",
        f"Best epoch: {fit_result.best_epoch}",
        "Best validation total loss: "
        f"{fit_result.best_validation_total_loss}",
        f"Training seconds: {fit_result.total_seconds}",
        "Best-epoch train reconstruction loss: "
        f"{selected_history['train_reconstruction_loss']}",
        "Best-epoch train topology loss: "
        f"{selected_history['train_topology_loss']}",
        "Best-epoch train total loss: "
        f"{selected_history['train_total_loss']}",
        "Best-epoch validation reconstruction loss: "
        f"{selected_history['validation_reconstruction_loss']}",
        "Best-epoch validation topology loss: "
        f"{selected_history['validation_topology_loss']}",
        "Best-epoch validation total loss: "
        f"{selected_history['validation_total_loss']}",
        "Final optimization-epoch total loss: "
        f"{final_history['train_total_loss']}",
        "Final train total loss: "
        f"{evaluations['train']['total_loss']}",
        "Final validation total loss: "
        f"{evaluations['validation']['total_loss']}",
        "Final test total loss: "
        f"{evaluations['test']['total_loss']}",
    ]
    return "\n".join(lines).rstrip() + "\n"


def save_autoencoder_checkpoint(
    model,
    checkpoint_path,
    *,
    input_scaler,
    topology_scaler,
    topology_loss_weight,
    representation_metadata,
    random_seed,
    best_epoch,
    best_validation_total_loss,
):
    """Save one device-independent autoencoder checkpoint.

    Architecture and both training-derived scalers are serialized beside a
    CPU copy of every learned tensor. Runtime stage ownership is deliberately
    reconstructed by the loader instead of becoming saved model structure.

    Parameters
    ----------
    model : VdfAutoencoder
        Selected complete-volume Conv3d model.
    checkpoint_path : str or pathlib.Path
        Destination ``autoencoder.pt`` path.
    input_scaler : InputFeatureScaler
        Training-derived full-volume normalization.
    topology_scaler : TopologyTargetScaler
        Training-derived six-target Earth-radius scaling.
    topology_loss_weight : float
        Fixed multiplier used in validation selection.
    representation_metadata : mapping
        Complete raw axis or Hermite order/rotation description.
    random_seed : int
        Training random seed.
    best_epoch : int
        One-based validation-total-loss-selected epoch.
    best_validation_total_loss : float
        Selected combined validation objective.

    Returns
    -------
    pathlib.Path
        Saved checkpoint path.

    Notes
    -----
    State tensors are copied to CPU. Runtime CUDA IDs, stage mapping,
    requested GPU count, DataLoader workers, and optimizer state are absent,
    so runtime placement is reconstructed for CPU, one GPU, or several GPUs.
    ``model_architecture.bottleneck_shape`` stores the configured per-axis
    maximum needed to rebuild the dense projections, while
    ``effective_bottleneck_shape`` records the resulting capped shape.
    """

    checkpoint = {
        "model_state_dict": {
            name: value.detach().cpu().clone()
            for name, value in model.state_dict().items()
        },
        "representation": model.representation,
        "input_volume_shape": list(model.input_shape),
        "encoder_channels": list(model.channels),
        "decoder_channels": [*reversed(model.channels), 1],
        "latent_size": model.latent_size,
        "effective_bottleneck_shape": list(model.bottleneck_shape),
        "input_normalization": {
            "mean": torch.from_numpy(input_scaler.mean.copy()),
            "scale": torch.from_numpy(input_scaler.scale.copy()),
            "epsilon": input_scaler.epsilon,
            "shape": list(input_scaler.mean.shape),
        },
        "topology_target_order": list(model.topology_target_names),
        "topology_scaler": topology_scaler.to_dict(),
        "topology_hidden_size": model.topology_hidden_size,
        "topology_loss_weight": float(topology_loss_weight),
        "model_architecture": model.constructor_config(),
        "random_seed": int(random_seed),
        "best_epoch": int(best_epoch),
        "best_validation_total_loss": float(best_validation_total_loss),
    }
    if model.representation == "hermite":
        checkpoint.update(
            {
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
    "create_autoencoder_metrics_text",
    "save_autoencoder_checkpoint",
    "save_autoencoder_outputs",
]
