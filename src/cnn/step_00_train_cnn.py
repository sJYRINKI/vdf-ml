"""Stage 0: orchestrate the ordered CNN training workflow.

The public entry point calls stages 1 through 8 in filename order: load the
saved representation, split complete timesteps, fit training-only scaling,
build the model, calculate losses during optimization, evaluate the selected
state, and save the two final training artifacts.

The stage receives a current dataset, resolved training configuration,
representation selector, output directory, and optional device override.
It returns a concise run summary after saving ``model.pt`` and the
consolidated human-readable ``metrics.txt`` report.
"""

from copy import deepcopy

import torch

from src.cnn.step_01_load_cnn_data import load_cnn_data
from src.cnn.step_02_split_cnn_timesteps import split_cnn_timesteps
from src.cnn.step_03_scale_cnn_inputs import (
    DEFAULT_SCALER_EPSILON,
    fit_input_scaler,
)
from src.cnn.step_04_build_cnn import build_cnn
from src.cnn.step_06_optimize_cnn import (
    close_loader,
    create_partition_loader,
    optimize_cnn,
    resolve_training_device,
    set_training_seed,
)
from src.cnn.step_07_evaluate_cnn import evaluate_cnn
from src.cnn.step_08_save_cnn import save_cnn_outputs
from src.learning.topology_supervision import TopologyTargetScaler
from src.representations.plasma_context import PlasmaContextScaler


def run_cnn_training(
    config,
    dataset_dir,
    output_dir,
    representation,
    *,
    device=None,
    model_parallel_gpus=None,
):
    """Train, evaluate, and save one raw or Hermite CNN.

    This is the package entry point called by the CNN training command. It
    keeps the saved representation sample-wise, prevents timestep leakage
    through chronological partitions and training-only scaling, restores
    the validation-selected model state, and delegates all artifact writes
    to stage 8.

    Parameters
    ----------
    config : mapping
        Parsed CNN YAML configuration.
    dataset_dir : str or pathlib.Path
        Current-format dataset directory.
    output_dir : str or pathlib.Path
        Directory receiving ``model.pt`` and ``metrics.txt``.
    representation : {"raw", "hermite"}
        Complete raw or Hermite VDF representation fused with the aligned
        16-value plasma context. The context stores Cartesian B, E, and fluid
        bulk-velocity components, density, and six pressure components;
        topology values remain auxiliary targets and are never inputs.
    device : str, optional
        CPU, CUDA, or automatic-device override.
    model_parallel_gpus : int, optional
        Override for the number of consecutive devices hosting contiguous
        model stages.

    Returns
    -------
    dict
        Output path, selected epoch, effective device, partition sample
        counts, and final class/topology metrics.
    """

    resolved = deepcopy(config)
    resolved["representation"] = representation
    if device is not None:
        resolved["device"] = device
    if model_parallel_gpus is not None:
        resolved["model_parallel_gpus"] = model_parallel_gpus
    effective_device = resolve_training_device(resolved["device"])
    set_training_seed(resolved["random_state"])
    data = load_cnn_data(
        dataset_dir,
        representation,
        raw_config=(
            resolved["raw"]
            if representation == "raw"
            else None
        ),
    )
    split = split_cnn_timesteps(
        data,
        train_fraction=resolved["split"]["train_fraction"],
        validation_fraction=resolved["split"]["validation_fraction"],
        gap_timesteps=resolved["split"]["gap_timesteps"],
    )
    input_scaler = fit_input_scaler(
        data,
        split.train_indices,
        batch_size=resolved["loader"]["normalization_batch_size"],
        epsilon=DEFAULT_SCALER_EPSILON,
    )
    plasma_context_scaler = PlasmaContextScaler.fit(
        data.plasma_context_path,
        split.train_indices,
    )
    topology_scaler = TopologyTargetScaler.fit(
        data.topology_targets,
        data.topology_mask,
        split.train_indices,
        epsilon=resolved["topology"]["scaler_epsilon"],
    )
    train_loader = create_partition_loader(
        data,
        split.train_indices,
        topology_scaler,
        plasma_context_scaler,
        resolved,
        shuffle=True,
        device=effective_device,
    )
    validation_loader = create_partition_loader(
        data,
        split.validation_indices,
        topology_scaler,
        plasma_context_scaler,
        resolved,
        shuffle=False,
        device=effective_device,
    )
    model = build_cnn(data, input_scaler, resolved).place_model_parallel(
        effective_device,
        resolved["model_parallel_gpus"],
    )
    input_device = model.input_device
    parameters_by_device = {}
    for parameter in model.parameters():
        if parameter.requires_grad:
            parameters_by_device.setdefault(
                parameter.device,
                [],
            ).append(parameter)
    optimizer = torch.optim.AdamW(
        [
            {"params": parameters}
            for parameters in parameters_by_device.values()
        ],
        lr=resolved["optimizer"]["learning_rate"],
        weight_decay=resolved["optimizer"]["weight_decay"],
        betas=tuple(resolved["optimizer"]["betas"]),
        eps=resolved["optimizer"]["epsilon"],
    )
    fit_result = optimize_cnn(
        model,
        train_loader,
        validation_loader,
        optimizer=optimizer,
        topology_scaler=topology_scaler,
        class_mapping=data.class_mapping,
        device=input_device,
        topology_loss_weight=resolved["topology"]["loss_weight"],
        max_epochs=resolved["training"]["max_epochs"],
        patience=resolved["training"]["patience"],
        min_delta=resolved["training"]["min_delta"],
        gradient_clip_norm=resolved["optimizer"]["gradient_clip_norm"],
    )
    close_loader(train_loader)
    close_loader(validation_loader)

    final_loaders = {
        name: create_partition_loader(
            data,
            indices,
            topology_scaler,
            plasma_context_scaler,
            resolved,
            shuffle=False,
            device=input_device,
        )
        for name, indices in (
            ("train", split.train_indices),
            ("validation", split.validation_indices),
            ("test", split.test_indices),
        )
    }
    evaluations = {
        name: evaluate_cnn(
            model=model,
            loader=loader,
            topology_scaler=topology_scaler,
            class_mapping=data.class_mapping,
            device=input_device,
            topology_loss_weight=resolved["topology"]["loss_weight"],
            split_name=name,
            include_classification_report=True,
        )
        for name, loader in final_loaders.items()
    }
    for loader in final_loaders.values():
        close_loader(loader)

    result = save_cnn_outputs(
        output_dir,
        model=model,
        input_scaler=input_scaler,
        plasma_context_scaler=plasma_context_scaler,
        topology_scaler=topology_scaler,
        data=data,
        config=resolved,
        fit_result=fit_result,
        evaluations=evaluations,
        split=split,
        selected_device=effective_device,
    )
    result.update(
        {
            "device": str(effective_device),
            "model_parallel_gpus": int(
                resolved["model_parallel_gpus"]
            ),
            "output_device": str(model.output_device),
            "sample_counts": {
                "train": len(split.train_indices),
                "validation": len(split.validation_indices),
                "test": len(split.test_indices),
            },
        }
    )
    return result


__all__ = ["run_cnn_training"]
