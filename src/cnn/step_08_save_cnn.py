"""Stage 8: save the trained CNN and one consolidated metrics report.

This final stage follows evaluation. It writes the model checkpoint and a
human-readable report describing the run, chronological split, final
physical-class performance, and auxiliary topology errors.

The stage receives the restored model, both training-derived scalers,
representation preprocessing, fit summary, split, and final evaluations.
It returns a concise run summary after directly writing ``model.pt`` and
``metrics.txt``. Prediction reads only the checkpoint.
"""

from copy import deepcopy
import json
from math import prod
from pathlib import Path

import torch

from src.cnn.class_mapping import ClassMapping
from src.data.load_velocity_grid import load_velocity_grid
from src.data.metadata_columns import TOPOLOGY_TARGET_COLUMNS
from src.data.velocity_grid import normalize_velocity_grid_geometry


def save_cnn_outputs(
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
    """Save the inference checkpoint and consolidated metrics report.

    This is the final side-effecting CNN stage. It prepares preprocessing
    needed by prediction, writes the checkpoint, serializes
    run settings and final evaluation metrics as readable text, and does
    not persist a separate epoch-history or metrics sidecar.

    Parameters
    ----------
    output_dir : str or pathlib.Path
        Directory receiving the two final CNN files.
    model : VdfCNN
        Validation-selected model.
    input_scaler : InputFeatureScaler
        Training-only representation normalization.
    topology_scaler : TopologyTargetScaler
        Training-only topology scaling.
    data : CnnTrainingData
        Loaded representation identity and dataset location used to recover
        the raw training velocity grid when required.
    config : mapping
        Resolved successful-path training configuration.
    fit_result : CnnTrainingFitResult
        Selected epoch and history.
    evaluations : mapping
        Final train, validation, and test evaluation results.
    split : CnnTimestepSplit
        Chronological partition indices, timesteps, and excluded gaps.
    selected_device : str or torch.device
        Effective device used for model optimization and evaluation.

    Returns
    -------
    dict
        Output directory, representation, selected epoch, epoch count, and
        final train/validation/test macro F1 values.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_cnn_checkpoint(
        model,
        output_dir / "model.pt",
        input_scaler=input_scaler,
        topology_scaler=topology_scaler,
        preprocessing=_prediction_preprocessing(data),
        hermite_rotate=(
            bool(data.representation_metadata["hermite_rotate"])
            if data.representation == "hermite"
            else None
        ),
        topology_loss_weight=config["topology"]["loss_weight"],
        random_seed=config["random_state"],
    )
    metrics_text = create_cnn_metrics_text(
        output_dir=output_dir,
        data=data,
        config=config,
        fit_result=fit_result,
        evaluations=evaluations,
        split=split,
        selected_device=selected_device,
    )
    (output_dir / "metrics.txt").write_text(
        metrics_text,
        encoding="utf-8",
    )
    return {
        "output_dir": str(output_dir),
        "representation": data.representation,
        "best_epoch": fit_result.best_epoch,
        "epochs_completed": fit_result.epochs_completed,
        "train_macro_f1": evaluations["train"].macro_f1,
        "validation_macro_f1": evaluations["validation"].macro_f1,
        "test_macro_f1": evaluations["test"].macro_f1,
    }


def create_cnn_metrics_text(
    output_dir,
    *,
    data,
    config,
    fit_result,
    evaluations,
    split,
    selected_device,
):
    """Create the human-readable CNN training and evaluation report.

    The report consolidates settings that identify the run, chronological
    timestep partitions, final metrics from the restored validation-selected
    model, dynamically labeled classification reports, fixed-order confusion
    matrices, and physical-unit topology errors. Physical classes come from
    the current dataset mapping; no historical class taxonomy is embedded.

    Parameters
    ----------
    output_dir : str or pathlib.Path
        Directory receiving the checkpoint and report.
    data : CnnTrainingData
        Dataset identity, representation shape, metadata, and class mapping.
    config : mapping
        Resolved training configuration used for this run.
    fit_result : CnnTrainingFitResult
        Selected epoch, completed-epoch count, timing, and loss history.
    evaluations : mapping
        Final train, validation, and test results from the restored model.
    split : CnnTimestepSplit
        Chronological partition indices, timesteps, and excluded gaps.
    selected_device : str or torch.device
        Effective PyTorch device used for the run.

    Returns
    -------
    str
        UTF-8 report text ending with one newline.

    Notes
    -----
    ``Final training loss`` is the aggregate optimizer loss from the last
    completed epoch before the selected model state is restored. The final
    partition losses are separate evaluation-mode measurements of that
    restored model. Error proxies retain the established definitions:
    error is one minus accuracy, generalization gap is test error minus
    train error, bias proxy is train error, and variance proxy is the
    nonnegative part of the generalization gap.
    """

    output_dir = Path(output_dir)
    metadata = data.metadata
    spec = data.representation_spec
    class_mapping = data.class_mapping
    class_order = ", ".join(
        f"{class_id}={class_name}"
        for class_id, class_name in zip(
            class_mapping.class_ids,
            class_mapping.class_names,
        )
    )
    train = evaluations["train"]
    validation = evaluations["validation"]
    test = evaluations["test"]
    train_error = 1.0 - train.accuracy
    validation_error = 1.0 - validation.accuracy
    test_error = 1.0 - test.accuracy
    generalization_gap = test_error - train_error
    selected_epoch_history = fit_result.history[fit_result.best_epoch - 1]
    final_epoch_history = fit_result.history[-1]
    separator = "=" * 70
    subsection = "-" * 70
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
        "CNN training run",
        separator,
        f"Dataset directory: {data.dataset_dir}",
        f"Output directory: {output_dir}",
        f"Representation: {data.representation}",
        f"Source array: {data.source_path}",
        f"Source array shape: {spec.source_shape}",
        f"Source array dtype: {spec.source_dtype}",
        f"Model input shape: {spec.tensor_shape}",
        f"Features per sample: {prod(spec.tensor_shape)}",
        *representation_details,
        f"Samples: {len(data)}",
        f"Unique timesteps: {metadata['timestep'].nunique()}",
        f"Minimum timestep: {metadata['timestep'].min()}",
        f"Maximum timestep: {metadata['timestep'].max()}",
        f"Unique cell IDs: {metadata['cid'].nunique()}",
        f"Physical classes: {len(class_mapping.class_ids)}",
        f"Class order: {class_order}",
        "Topology target order: " + ", ".join(TOPOLOGY_TARGET_COLUMNS),
        f"Device: {selected_device}",
        f"PyTorch version: {torch.__version__}",
        f"Random seed: {config['random_state']}",
        "Deterministic algorithms enabled: "
        f"{torch.are_deterministic_algorithms_enabled()}",
        "Unsupported deterministic operations warn only: "
        f"{torch.is_deterministic_algorithms_warn_only_enabled()}",
        "",
        "Physical class sample counts",
        subsection,
    ]
    lines.extend(
        f"{class_id} {class_name}: "
        f"{int((metadata['class_id'] == class_id).sum())}"
        for class_id, class_name in zip(
            class_mapping.class_ids,
            class_mapping.class_names,
        )
    )
    lines.extend(
        [
            "",
            "Resolved configuration",
            subsection,
            json.dumps(config, indent=2, sort_keys=True),
            "",
            "Chronological timestep split",
            separator,
            "Configured train fraction: "
            f"{config['split']['train_fraction']}",
            "Configured validation fraction: "
            f"{config['split']['validation_fraction']}",
            "Gap timesteps per boundary: "
            f"{config['split']['gap_timesteps']}",
            f"Train samples: {len(split.train_indices)}",
            f"Validation samples: {len(split.validation_indices)}",
            f"Test samples: {len(split.test_indices)}",
            "Train timesteps: "
            f"{_format_timestep_range(split.train_timesteps)}",
            "Train-validation gap: "
            f"{_format_timestep_range(split.train_validation_gap)}",
            "Validation timesteps: "
            f"{_format_timestep_range(split.validation_timesteps)}",
            "Validation-test gap: "
            f"{_format_timestep_range(split.validation_test_gap)}",
            "Test timesteps: "
            f"{_format_timestep_range(split.test_timesteps)}",
            "",
            "Training summary",
            separator,
            "Selection metric: validation macro F1 with total-loss "
            "tie-breaker",
            "Loss definition: classification cross entropy + "
            f"{config['topology']['loss_weight']} * scaled topology "
            "Smooth L1",
            f"Epochs completed: {fit_result.epochs_completed}",
            f"Selected best epoch: {fit_result.best_epoch}",
            f"Training seconds: {fit_result.total_seconds}",
            "Best validation macro F1: "
            f"{fit_result.best_validation_result.macro_f1}",
            "Selected-epoch training loss: "
            f"{selected_epoch_history['train_total_loss']}",
            "Final training loss: "
            f"{final_epoch_history['train_total_loss']}",
            f"Train accuracy: {train.accuracy}",
            f"Validation accuracy: {validation.accuracy}",
            f"Test accuracy: {test.accuracy}",
            f"Train macro F1: {train.macro_f1}",
            f"Validation macro F1: {validation.macro_f1}",
            f"Test macro F1: {test.macro_f1}",
            f"Train error: {train_error}",
            f"Validation error: {validation_error}",
            f"Test error: {test_error}",
            f"Generalization gap: {generalization_gap}",
            f"Bias proxy: {train_error}",
            f"Variance proxy: {max(0.0, generalization_gap)}",
            "",
            "Final restored-model partition metrics",
            separator,
        ]
    )
    for split_name, result in evaluations.items():
        lines.extend(
            [
                f"{split_name.capitalize()} samples: {result.sample_count}",
                f"{split_name.capitalize()} total loss: {result.total_loss}",
                f"{split_name.capitalize()} classification loss: "
                f"{result.classification_loss}",
                f"{split_name.capitalize()} topology loss: "
                f"{result.topology_loss}",
                f"{split_name.capitalize()} weighted F1: "
                f"{result.weighted_f1}",
                "",
            ]
        )

    lines.extend(
        [
            "Classification reports",
            separator,
        ]
    )
    for split_name, result in evaluations.items():
        lines.extend(
            [
                f"{split_name.capitalize()} classification report",
                subsection,
                result.classification_report_text.rstrip(),
                "",
            ]
        )

    lines.extend(
        [
            "Confusion matrices",
            separator,
            f"Class order: {class_order}",
            "Rows: true physical classes",
            "Columns: predicted physical classes",
            "",
        ]
    )
    for split_name, result in evaluations.items():
        lines.extend(
            [
                f"{split_name.capitalize()} confusion matrix",
                subsection,
                str(result.confusion_matrix),
                "",
            ]
        )

    lines.extend(["Topology metrics", separator])
    for split_name, result in evaluations.items():
        lines.extend(
            [
                f"{split_name.capitalize()} topology metrics",
                subsection,
                "Valid scalar targets: "
                f"{result.topology_valid_target_count}",
                "Aggregate MAE (R_E): "
                f"{_format_metric(result.topology_mae_re)}",
                "Aggregate RMSE (R_E): "
                f"{_format_metric(result.topology_rmse_re)}",
                "target | units | valid_count | mae | rmse",
            ]
        )
        lines.extend(
            "{target_name} | {units} | {valid_count} | {mae} | {rmse}".format(
                target_name=record["target_name"],
                units=record["units"],
                valid_count=record["valid_count"],
                mae=_format_metric(record["mae"]),
                rmse=_format_metric(record["rmse"]),
            )
            for record in result.topology_metrics
        )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def save_cnn_checkpoint(
    model,
    checkpoint_path,
    *,
    input_scaler,
    topology_scaler,
    preprocessing,
    hermite_rotate,
    topology_loss_weight,
    random_seed,
):
    """Save one optimizer-independent CNN checkpoint.

    Prediction reconstructs the network, input normalization, class order,
    topology scaling, and representation preprocessing from this single
    dictionary. Optimizer state is intentionally absent because the file is
    an inference checkpoint rather than a resumable training session.

    Parameters
    ----------
    model : VdfCNN
        Trained raw or Hermite multitask model.
    checkpoint_path : str or pathlib.Path
        Destination checkpoint file.
    input_scaler : InputFeatureScaler
        Training-derived feature normalization.
    topology_scaler : TopologyTargetScaler
        Training-derived topology normalization.
    preprocessing : dict
        Raw preprocessing and velocity grid, or the Hermite representation
        identity needed by prediction.
    hermite_rotate : bool or None
        Optional physical velocity-space rotation used to create Hermite
        training coefficients. Raw checkpoints pass ``None``.
    topology_loss_weight : float
        Auxiliary topology-loss multiplier.
    random_seed : int
        Training random seed.

    Returns
    -------
    pathlib.Path
        Saved checkpoint path.
    """

    class_mapping = ClassMapping(model.class_ids, model.class_names)
    checkpoint_preprocessing = deepcopy(preprocessing)
    checkpoint_preprocessing.pop("representation_version", None)
    if checkpoint_preprocessing["raw_preprocessing"] is not None:
        checkpoint_preprocessing["raw_preprocessing"].pop(
            "representation_version",
            None,
        )
    checkpoint = {
        "model_state_dict": {
            name: value.detach().cpu()
            for name, value in model.state_dict().items()
        },
        "representation": model.representation,
        "expected_input_shape": list(model.input_shape),
        "preprocessing": checkpoint_preprocessing,
        "input_normalization": input_scaler.to_dict(),
        "class_mapping": class_mapping.to_dict(),
        "topology_target_names": list(TOPOLOGY_TARGET_COLUMNS),
        "topology_scaler": topology_scaler.to_dict(),
        "model_architecture": model.constructor_config(),
        "topology_loss_weight": float(topology_loss_weight),
        "random_seed": int(random_seed),
    }
    if model.representation == "hermite":
        checkpoint.update(
            {
                "volume_shape": list(model.input_shape),
                "hermite_order": int(model.input_shape[0]),
                "hermite_rotate": bool(hermite_rotate),
            }
        )
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, checkpoint_path)
    return checkpoint_path


def _prediction_preprocessing(data):
    """Create prediction preprocessing data for the saved checkpoint.

    Raw prediction must reproduce the complete logarithmic VDF on the
    training velocity grid, while Hermite prediction must reproduce the
    coefficient convention. This function records the appropriate mutually
    exclusive information before stage 8 writes the checkpoint.

    Parameters
    ----------
    data : CnnTrainingData
        Loaded representation metadata and dataset directory.

    Returns
    -------
    dict
        Complete-volume raw preprocessing and training velocity grid, or
        the physical-VDF Hermite representation identity. Actual Hermite
        shape, order, and rotation are stored as direct checkpoint fields.
    """

    if data.representation == "raw":
        raw_preprocessing = deepcopy(data.representation_metadata)
        raw_preprocessing.pop("representation_version", None)
        return {
            "representation": "raw",
            "raw_preprocessing": raw_preprocessing,
            "training_velocity_grid": _serialize_velocity_grid(
                load_velocity_grid(data.dataset_dir)
            ),
        }
    return {
        "representation": "hermite",
        "raw_preprocessing": None,
        "training_velocity_grid": None,
    }


def _serialize_velocity_grid(velocity_grid):
    """Convert a physical velocity-grid descriptor to checkpoint values.

    Parameters
    ----------
    velocity_grid : mapping
        Grid shape, metre-per-second extent, axis order, and population.

    Returns
    -------
    dict
        JSON-like primitive values retaining ``(vx, vy, vz)`` axis order
        and the six-value ``extent_mps`` bounds.
    """

    normalized = normalize_velocity_grid_geometry(velocity_grid)
    return {
        "shape": [int(value) for value in normalized["shape"]],
        "extent_mps": [
            float(value) for value in normalized["extent_mps"]
        ],
        "axis_order": [str(value) for value in normalized["axis_order"]],
        "population": str(normalized["population"]),
    }


def _format_timestep_range(timesteps):
    """Format one chronological partition or excluded boundary gap.

    Parameters
    ----------
    timesteps : array-like
        Ordered simulation timesteps assigned to one report row.

    Returns
    -------
    str
        ``"none"``, one timestep, or the inclusive first-to-last range.
    """

    if len(timesteps) == 0:
        return "none"
    if len(timesteps) == 1:
        return str(timesteps[0])
    return f"{timesteps[0]} ... {timesteps[-1]}"


def _format_metric(value):
    """Format an available physical metric or its missing-data state.

    Parameters
    ----------
    value : float or None
        Aggregate or per-target physical topology metric.

    Returns
    -------
    str
        Numeric value or ``"none"`` when the split has no valid target.
    """

    return "none" if value is None else str(value)


__all__ = [
    "create_cnn_metrics_text",
    "save_cnn_checkpoint",
    "save_cnn_outputs",
]
