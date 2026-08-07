"""Stage 0: orchestrate the ordered autoencoder training workflow.

The public entry point calls stages 1 through 7 in filename order: load the
representation, split complete timesteps, fit training-only normalization,
build and optimize the model, evaluate the selected state, and save the
four final reconstruction artifacts.

The stage receives a current dataset, resolved autoencoder configuration,
representation selector, output directory, and optional device override.
It returns a concise run summary after writing ``autoencoder.pt``,
``metrics.txt``, ``training_history.csv``, and
``reconstruction_examples.png``.
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
from src.autoencoder.step_04_build_autoencoder import (
    build_autoencoder,
)
from src.autoencoder.step_05_optimize_autoencoder import (
    optimize_autoencoder,
)
from src.autoencoder.step_06_evaluate_autoencoder import (
    evaluate_autoencoder,
)
from src.autoencoder.step_07_save_autoencoder import (
    save_autoencoder_outputs,
)
from src.cnn.step_06_optimize_cnn import (
    resolve_training_device,
    set_training_seed,
)


def run_autoencoder_training(
    config,
    dataset_dir,
    output_dir,
    representation,
    *,
    device=None,
):
    """Train, evaluate, and save one raw or Hermite autoencoder.

    This package entry point coordinates sample-wise loading, timestep-aware
    splitting, training-only normalization, normalized-space
    reconstruction, validation-MSE selection, final evaluation, and stage-7
    output writing. Labels and topology values never enter the workflow.

    Parameters
    ----------
    config : mapping
        Parsed autoencoder YAML configuration.
    dataset_dir : str or pathlib.Path
        Current-format dataset directory.
    output_dir : str or pathlib.Path
        Directory receiving ``autoencoder.pt``, ``metrics.txt``,
        ``training_history.csv``, and ``reconstruction_examples.png``.
    representation : {"raw", "hermite"}
        Representation reconstructed in normalized feature space.
    device : str, optional
        CPU, CUDA, or automatic-device override.

    Returns
    -------
    dict
        Output path, effective device, selected epoch, partition sample
        counts, and final normalized-space MSE values.
    """

    resolved = deepcopy(config)
    resolved["representation"] = representation
    if device is not None:
        resolved["device"] = device
    effective_device = resolve_training_device(resolved["device"])
    set_training_seed(resolved["random_state"])
    data = load_autoencoder_data(
        dataset_dir,
        representation,
        raw_config=(
            resolved["raw"]
            if representation == "raw"
            else None
        ),
    )
    split = split_autoencoder_timesteps(data, resolved["split"])
    input_scaler = scale_autoencoder_inputs(
        data,
        split.train_indices,
        resolved["data_loader"],
    )
    train_loader = _create_loader(
        data,
        split.train_indices,
        resolved,
        shuffle=True,
        device=effective_device,
    )
    validation_loader = _create_loader(
        data,
        split.validation_indices,
        resolved,
        shuffle=False,
        device=effective_device,
    )
    model = build_autoencoder(
        data,
        input_scaler,
        resolved["model"],
    ).to(effective_device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=resolved["optimizer"]["learning_rate"],
        weight_decay=resolved["optimizer"]["weight_decay"],
        betas=tuple(resolved["optimizer"]["betas"]),
        eps=resolved["optimizer"]["epsilon"],
    )
    fit_result = optimize_autoencoder(
        model,
        train_loader,
        validation_loader,
        optimizer=optimizer,
        device=effective_device,
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
            effective_device,
            name,
        )
        for name, loader in final_loaders.items()
    }
    for loader in final_loaders.values():
        loader.dataset.close()

    result = save_autoencoder_outputs(
        output_dir,
        model=model,
        input_scaler=input_scaler,
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
            "sample_counts": {
                "train": len(split.train_indices),
                "validation": len(split.validation_indices),
                "test": len(split.test_indices),
            },
        }
    )
    return result


def _create_loader(data, indices, config, *, shuffle, device):
    """Create a sample-wise loader for one autoencoder partition.

    Parameters
    ----------
    data : AutoencoderTrainingData
        Representation path, tensor convention, and saved sample identity.
    indices : sequence of int
        Stable rows assigned to one timestep partition.
    config : mapping
        Batch size, worker count, pinned-memory, and random-seed settings.
    shuffle : bool
        Shuffle training traversal when true.
    device : torch.device
        Selected device used to resolve automatic pinned memory.

    Returns
    -------
    torch.utils.data.DataLoader
        Loader yielding one-channel inputs and saved identities.
    """

    pin_memory = config["data_loader"]["pin_memory"]
    if pin_memory == "auto":
        pin_memory = device.type == "cuda"
    return create_autoencoder_dataloader(
        data,
        indices,
        batch_size=config["data_loader"]["batch_size"],
        shuffle=shuffle,
        random_seed=config["random_state"],
        num_workers=config["data_loader"]["num_workers"],
        pin_memory=pin_memory,
    )


__all__ = ["run_autoencoder_training"]
