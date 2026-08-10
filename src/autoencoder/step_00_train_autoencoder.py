"""Stage 0: orchestrate full-volume topology-aware autoencoder training.

The ordered workflow reads complete raw or Hermite volumes on demand from
read-only memory maps, splits complete timesteps, fits representation and
topology scaling from training rows only, places one Conv3d model across
consecutive devices, optimizes one combined objective, evaluates the selected
state, and writes the four established artifacts.

One Python process owns one model and one AdamW optimizer. DataLoader worker
processes only read and preprocess samples; they do not distribute training.
"""

from copy import deepcopy

import torch

from src.autoencoder.step_01_load_autoencoder_data import (
    create_autoencoder_dataloader,
    load_autoencoder_data,
)
from src.autoencoder.step_02_split_autoencoder_timesteps import (
    split_autoencoder_timesteps,
)
from src.autoencoder.step_03_scale_autoencoder_inputs import (
    scale_autoencoder_inputs,
)
from src.autoencoder.step_04_build_autoencoder import build_autoencoder
from src.autoencoder.step_05_optimize_autoencoder import optimize_autoencoder
from src.autoencoder.step_06_evaluate_autoencoder import evaluate_autoencoder
from src.autoencoder.step_07_save_autoencoder import save_autoencoder_outputs
from src.cnn.step_06_optimize_cnn import (
    resolve_training_device,
    set_training_seed,
)
from src.learning.topology_supervision import TopologyTargetScaler


def run_autoencoder_training(
    config,
    dataset_dir,
    output_dir,
    representation,
    *,
    device=None,
    model_parallel_gpus=None,
):
    """Train, evaluate, and save one full-volume multitask autoencoder.

    Raw inputs are complete logarithmic ``X.npy`` volumes with model shape
    ``(batch, 1, vx, vy, vz)``. Hermite inputs are complete signed
    ``X_hermite.npy`` cubes with model shape ``(batch, 1, n1, n2, n3)``.
    Both use the same Conv3d architecture and share one latent vector between
    reconstruction and the six-target topology head.

    Parameters
    ----------
    config : mapping
        Parsed autoencoder YAML configuration.
    dataset_dir : str or pathlib.Path
        Current-format dataset directory opened read-only.
    output_dir : str or pathlib.Path
        Directory receiving ``autoencoder.pt``, ``metrics.txt``,
        ``training_history.csv``, and ``reconstruction_examples.png``.
    representation : {"raw", "hermite"}
        Complete three-dimensional representation to reconstruct.
    device : str, optional
        CPU, CUDA, or automatic first-device override.
    model_parallel_gpus : int, optional
        CLI override for consecutive stage placement in this one process.

    Returns
    -------
    dict
        Artifact paths, requested and effective device counts, selected
        epoch, partition counts, and final reconstruction/topology objectives.

    Notes
    -----
    Topology scaling uses valid training entries only. Missing topology does
    not remove reconstruction samples. Validation total loss selects the
    checkpoint; test rows remain evaluation-only.
    """

    resolved = deepcopy(config)
    resolved["representation"] = representation
    if device is not None:
        resolved["device"] = device
    if model_parallel_gpus is not None:
        resolved["model_parallel_gpus"] = int(model_parallel_gpus)
    effective_device = resolve_training_device(resolved["device"])
    set_training_seed(resolved["random_state"])
    data = load_autoencoder_data(
        dataset_dir,
        representation,
        raw_config=resolved["raw"] if representation == "raw" else None,
    )
    split = split_autoencoder_timesteps(data, resolved["split"])
    input_scaler = scale_autoencoder_inputs(
        data,
        split.train_indices,
        resolved["data_loader"],
    )
    topology_scaler = TopologyTargetScaler.fit(
        data.topology_targets,
        data.topology_mask,
        split.train_indices,
    )
    train_loader = _create_loader(
        data,
        split.train_indices,
        topology_scaler,
        resolved,
        shuffle=True,
        device=effective_device,
    )
    validation_loader = _create_loader(
        data,
        split.validation_indices,
        topology_scaler,
        resolved,
        shuffle=False,
        device=effective_device,
    )
    model = build_autoencoder(
        data,
        input_scaler,
        resolved["model"],
        resolved["topology"],
    ).place_model_parallel(
        effective_device,
        resolved["model_parallel_gpus"],
    )
    optimizer = _create_optimizer(model, resolved["optimizer"])
    fit_result = optimize_autoencoder(
        model,
        train_loader,
        validation_loader,
        optimizer=optimizer,
        topology_scaler=topology_scaler,
        topology_loss_weight=resolved["topology"]["loss_weight"],
        max_epochs=resolved["training"]["max_epochs"],
        patience=resolved["training"]["patience"],
        min_delta=resolved["training"]["min_delta"],
    )
    train_loader.dataset.close()
    validation_loader.dataset.close()

    final_loaders = {
        name: _create_loader(
            data,
            indices,
            topology_scaler,
            resolved,
            shuffle=False,
            device=effective_device,
        )
        for name, indices in (
            ("train", split.train_indices),
            ("validation", split.validation_indices),
            ("test", split.test_indices),
        )
    }
    evaluations = {
        name: evaluate_autoencoder(
            model,
            loader,
            topology_scaler,
            name,
            resolved["topology"]["loss_weight"],
        )
        for name, loader in final_loaders.items()
    }
    for loader in final_loaders.values():
        loader.dataset.close()

    result = save_autoencoder_outputs(
        output_dir,
        model=model,
        input_scaler=input_scaler,
        topology_scaler=topology_scaler,
        data=data,
        config=resolved,
        fit_result=fit_result,
        evaluations=evaluations,
        split=split,
        selected_device=model.input_device,
    )
    result.update(
        {
            "device": str(model.input_device),
            "output_device": str(model.output_device),
            "requested_model_parallel_gpus": (
                model.requested_model_parallel_gpus
            ),
            "effective_model_parallel_gpus": (
                model.effective_model_parallel_gpus
            ),
            "sample_counts": {
                "train": len(split.train_indices),
                "validation": len(split.validation_indices),
                "test": len(split.test_indices),
            },
        }
    )
    return result


def _create_optimizer(model, config):
    """Create one AdamW optimizer across all occupied stage devices.

    Parameters
    ----------
    model : VdfAutoencoder
        Placed model whose parameters may occupy several CUDA devices.
    config : mapping
        Established AdamW learning rate, weight decay, betas, and epsilon.

    Returns
    -------
    torch.optim.AdamW
        Single optimizer containing every trainable parameter exactly once.
    """

    parameters_by_device = {}
    for parameter in model.parameters():
        if parameter.requires_grad:
            parameters_by_device.setdefault(parameter.device, []).append(
                parameter
            )
    return torch.optim.AdamW(
        [
            {"params": parameters}
            for parameters in parameters_by_device.values()
        ],
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
        betas=tuple(config["betas"]),
        eps=config["epsilon"],
    )


def _create_loader(
    data,
    indices,
    topology_scaler,
    config,
    *,
    shuffle,
    device,
):
    """Create one on-demand chronological partition loader.

    Parameters
    ----------
    data : AutoencoderTrainingData
        Complete-volume source and aligned topology data.
    indices : sequence of int
        Stable saved rows assigned to one timestep partition.
    topology_scaler : TopologyTargetScaler
        Training-fitted six-target scaler.
    config : mapping
        Batch, worker, pinned-memory, and seed settings.
    shuffle : bool
        Shuffle training traversal when true.
    device : torch.device
        First runtime device used only to resolve automatic host pinning.

    Returns
    -------
    torch.utils.data.DataLoader
        Loader yielding full-volume input and aligned auxiliary targets.
    """

    pin_memory = config["data_loader"]["pin_memory"]
    if pin_memory == "auto":
        pin_memory = device.type == "cuda"
    return create_autoencoder_dataloader(
        data,
        indices,
        topology_scaler=topology_scaler,
        batch_size=config["data_loader"]["batch_size"],
        shuffle=shuffle,
        random_seed=config["random_state"],
        num_workers=config["data_loader"]["num_workers"],
        pin_memory=pin_memory,
    )


__all__ = ["run_autoencoder_training"]
