import csv

import numpy as np
import analysator as pt
from joblib import Parallel, delayed

from src.dataset_plot import plot_vdf_xz_slice
from src.features import create_features
from src.timesteps import create_path, create_timestep_list
from src.vdf_extract import VdfExtractor, extract_vdf
from src.vdf_helpers import (
    create_coordinate_name,
    create_region_mask_re,
    get_cellid_with_vdf,
    get_vdf_cells_with_coords_re,
    get_vdf_plot_parameters_from_file,
)


def predict_coordinate(
    config,
    timestep,
    model_id,
    coord_re,
    load_model,
    file_source=None,
):
    """
    Predict the class of one VDF coordinate and save an xz plot.

    Parameters
    ----------
    config : dict
        Coordinate prediction config.
    timestep : int
        Timestep to read.
    model_id : str
        Trained model identifier.
    coord_re : array-like of float
        Coordinate in Earth radii, given as ``[x, y, z]``.
    load_model : callable
        Function that loads a model and preprocessing values from ``model_dir``.
    file_source : str, optional
        Name of file template source to use.

    Returns
    -------
    dict
        Prediction result and output path.
    """

    timestep = int(timestep)
    resolved_file_source, file_template = resolve_file_source_and_template(
        config=config,
        file_source=file_source,
    )
    file_location = create_path(
        path_template=file_template,
        timestep=timestep,
    )
    model_dir = create_path(
        path_template=config["model_dir"],
        model_id=model_id,
        timestep=timestep,
        file_source=resolved_file_source,
    )
    output_dir = create_path(
        path_template=config["output_dir"],
        model_id=model_id,
        timestep=timestep,
        file_source=resolved_file_source,
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    labels = config.get("labels")
    label_to_class = {
        int(label): class_name
        for class_name, label in labels.items()
    }
    plot_config = config["plot"]
    vdflim = float(plot_config.get("vdflim", 2000000.0))

    reader = pt.vlsvfile.VlsvReader(str(file_location))
    cid = get_cellid_with_vdf(
        reader=reader,
        coord_re=coord_re,
    )
    vdf = extract_vdf(
        file_location=file_location,
        cid=cid,
    )

    model, preprocessing = load_model(model_dir)
    downsample_factor = int(preprocessing["downsample_factor"])
    log_eps = float(preprocessing["log_eps"])

    features = create_features(
        X=vdf[None, ...],
        downsample_factor=downsample_factor,
        log_eps=log_eps,
    )
    predicted_label = int(model.predict(features)[0])
    score_name, prediction_score = get_prediction_score(
        model=model,
        features=features,
        predicted_label=predicted_label,
    )
    predicted_class_name = label_to_class[predicted_label]

    extent, dv, threshold = get_vdf_plot_parameters_from_file(
        file_location=file_location,
        cid=cid,
        vdf_shape=vdf.shape,
    )
    metadata_row = {
        "timestep": timestep,
        "cid": int(cid),
        "x_re": float(coord_re[0]),
        "y_re": float(coord_re[1]),
        "z_re": float(coord_re[2]),
        "file_source": resolved_file_source,
        "file_location": str(file_location),
    }

    coord_name = create_coordinate_name(coord_re)
    output_plot_path = output_dir / f"prediction_{coord_name}_xz.png"
    plot_vdf_xz_slice(
        vdf=vdf,
        y_label=None,
        metadata_row=metadata_row,
        extent=extent,
        output_path=output_plot_path,
        dv=dv,
        threshold=threshold,
        vdflim=vdflim,
        predicted_class_name=predicted_class_name,
        decision_score=prediction_score,
    )

    return {
        "output_plot_path": output_plot_path,
        "predicted_label": predicted_label,
        "predicted_class_name": predicted_class_name,
        "score_name": score_name,
        "prediction_score": prediction_score,
        "model_classes": model.classes_,
    }


def predict_region_timesteps(
    config,
    start_timestep,
    n_timesteps,
    model_id,
    load_model,
    file_source=None,
):
    """
    Predict VDF cells inside a region for a sequence of timesteps.

    Parameters
    ----------
    config : dict
        Region prediction config.
    start_timestep : int
        First timestep to predict.
    n_timesteps : int
        Number of consecutive timesteps to predict.
    model_id : str
        Trained model identifier.
    load_model : callable
        Function that loads a model and preprocessing values from ``model_dir``.
    file_source : str, optional
        Name of file template source to use.

    Returns
    -------
    list of dict
        One prediction summary per timestep.
    """

    timesteps = create_timestep_list(
        start_timestep=int(start_timestep),
        n_timesteps=int(n_timesteps),
    )
    prediction_config = config.get("prediction", {})
    n_jobs = int(prediction_config.get("n_jobs", 1))

    if n_jobs == 0:
        raise ValueError("prediction.n_jobs must be non-zero")

    if n_jobs == 1:
        return [
            predict_region(
                config=config,
                timestep=timestep,
                model_id=model_id,
                load_model=load_model,
                file_source=file_source,
            )
            for timestep in timesteps
        ]

    return Parallel(n_jobs=n_jobs)(
        delayed(predict_region)(
            config=config,
            timestep=timestep,
            model_id=model_id,
            load_model=load_model,
            file_source=file_source,
        )
        for timestep in timesteps
    )


def predict_region(config, timestep, model_id, load_model, file_source=None):
    """
    Predict all VDF cells inside one configured spatial region.

    Parameters
    ----------
    config : dict
        Region prediction config.
    timestep : int
        VLSV timestep to read.
    model_id : str
        Trained model identifier.
    load_model : callable
        Function that loads a model and preprocessing values from ``model_dir``.
    file_source : str, optional
        Name of file template source to use.

    Returns
    -------
    dict
        Prediction summary containing output path and selected cell count.
    """

    timestep = int(timestep)
    resolved_file_source, file_template = resolve_file_source_and_template(
        config=config,
        file_source=file_source,
    )
    file_location = create_path(
        path_template=file_template,
        timestep=timestep,
    )
    model_dir = create_path(
        path_template=config["model_dir"],
        model_id=model_id,
        timestep=timestep,
        file_source=resolved_file_source,
    )
    output_dir = create_path(
        path_template=config["output_dir"],
        model_id=model_id,
        timestep=timestep,
        file_source=resolved_file_source,
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    if not file_location.exists():
        raise FileNotFoundError(file_location)

    model, preprocessing = load_model(model_dir)

    labels = config.get("labels")
    label_to_class = {
        int(label): class_name
        for class_name, label in labels.items()
    }
    prediction_config = config.get("prediction", {})
    batch_size = int(prediction_config.get("batch_size", 16))
    feature_n_jobs = int(prediction_config.get("feature_n_jobs", 1))

    if batch_size <= 0:
        raise ValueError("prediction.batch_size must be positive")

    downsample_factor = int(preprocessing["downsample_factor"])
    log_eps = float(preprocessing["log_eps"])

    reader = pt.vlsvfile.VlsvReader(str(file_location))
    cellids, coords_re = get_vdf_cells_with_coords_re(reader=reader)
    region_mask = create_region_mask_re(
        coords_re=coords_re,
        region_re=config["region_re"],
    )
    cellids = cellids[region_mask]
    coords_re = coords_re[region_mask]

    output_path = output_dir / f"predictions_timestep_{timestep}.csv"
    if hasattr(model, "predict_proba"):
        class_probability_fields = create_class_probability_fields(
            class_labels=model.classes_,
            label_to_class=label_to_class,
        )
    else:
        class_probability_fields = []

    fieldnames = [
        "timestep",
        "file_source",
        "cid",
        "x_re",
        "y_re",
        "z_re",
        "predicted_label",
        "predicted_class_name",
        "score_name",
        "prediction_score",
        "max_class_probability",
    ] + class_probability_fields

    with output_path.open("w", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()

        if len(cellids) > 0:
            extractor = VdfExtractor(reader=reader)

            for start in range(0, len(cellids), batch_size):
                end = start + batch_size
                batch_cellids = cellids[start:end]
                batch_coords_re = coords_re[start:end]
                vdfs = np.asarray(
                    [
                        extractor.extract(cid=int(cid))
                        for cid in batch_cellids
                    ],
                    dtype=np.float32,
                )
                features = create_features(
                    X=vdfs,
                    downsample_factor=downsample_factor,
                    log_eps=log_eps,
                    n_jobs=feature_n_jobs,
                )
                predicted_labels = model.predict(features)
                score_name, prediction_scores, probabilities = (
                    get_batch_prediction_scores(
                        model=model,
                        features=features,
                        predicted_labels=predicted_labels,
                    )
                )

                writer.writerows(
                    create_region_prediction_rows(
                        timestep=timestep,
                        file_source=resolved_file_source,
                        cellids=batch_cellids,
                        coords_re=batch_coords_re,
                        predicted_labels=predicted_labels,
                        score_name=score_name,
                        prediction_scores=prediction_scores,
                        probabilities=probabilities,
                        class_labels=model.classes_,
                        label_to_class=label_to_class,
                    )
                )

    return {
        "timestep": timestep,
        "file_source": resolved_file_source,
        "file_location": file_location,
        "output_path": output_path,
        "n_selected_cells": int(len(cellids)),
    }


def resolve_file_source_and_template(config, file_source=None):
    """
    Return selected file source name and VLSV path template.

    Parameters
    ----------
    config : dict
        Prediction config.
    file_source : str, optional
        File source override.

    Returns
    -------
    file_source : str
        Selected file source name.
    file_template : str
        VLSV path template for the selected source.
    """

    file_templates = config.get("file_templates")
    if file_templates is None:
        return str(file_source or config.get("file_source", "default")), config[
            "file_template"
        ]

    selected_source = file_source or config.get("file_source")
    if selected_source is None:
        raise ValueError(
            "Config with file_templates must set file_source or receive "
            "a file_source override"
        )

    selected_source = str(selected_source)
    if selected_source not in file_templates:
        raise ValueError(
            f"Unknown file_source {selected_source!r}. "
            f"Expected one of {sorted(file_templates)}."
        )

    return selected_source, file_templates[selected_source]


def create_class_probability_fields(class_labels, label_to_class):
    """
    Create CSV field names for class probabilities.

    Parameters
    ----------
    class_labels : array-like of int
        Model class labels in probability-column order.
    label_to_class : dict
        Mapping from integer label to class name.

    Returns
    -------
    list of str
        Probability column names.
    """

    return [
        f"prob_{label_to_class[int(label)]}"
        for label in class_labels
    ]


def get_batch_prediction_scores(model, features, predicted_labels):
    """
    Return predicted-class scores for a batch of feature rows.

    Parameters
    ----------
    model : sklearn estimator
        Fitted model or pipeline.
    features : numpy.ndarray
        Feature matrix.
    predicted_labels : array-like of int
        Labels predicted by the model.

    Returns
    -------
    score_name : str
        Human-readable score name.
    scores : numpy.ndarray
        Predicted-class probability or decision score for each row.
    probabilities : numpy.ndarray or None
        Class probability matrix when available.
    """

    predicted_labels = np.asarray(predicted_labels, dtype=int)
    classes = list(model.classes_)

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features)
        class_indices = np.asarray(
            [classes.index(int(label)) for label in predicted_labels],
            dtype=int,
        )
        scores = probabilities[np.arange(len(predicted_labels)), class_indices]

        return "Predicted probability", scores.astype(float), probabilities

    if hasattr(model, "decision_function"):
        scores = get_batch_decision_scores(
            model=model,
            features=features,
            predicted_labels=predicted_labels,
        )

        return "Decision score", scores.astype(float), None

    return (
        "Prediction score",
        np.full(len(predicted_labels), np.nan, dtype=float),
        None,
    )


def get_batch_decision_scores(model, features, predicted_labels):
    """
    Return decision scores with positive sign for the predicted class.

    Parameters
    ----------
    model : sklearn estimator
        Fitted model or pipeline with ``decision_function``.
    features : numpy.ndarray
        Feature matrix.
    predicted_labels : array-like of int
        Labels predicted by the model.

    Returns
    -------
    numpy.ndarray
        Decision scores for the predicted classes.
    """

    classes = list(model.classes_)
    decision_scores = np.asarray(model.decision_function(features))

    if len(classes) == 2:
        if decision_scores.ndim == 1:
            scores = decision_scores
        else:
            scores = decision_scores[:, 0]

        return np.where(
            predicted_labels == int(classes[0]),
            -scores,
            scores,
        )

    if decision_scores.ndim == 1:
        return decision_scores

    class_indices = np.asarray(
        [classes.index(int(label)) for label in predicted_labels],
        dtype=int,
    )

    return decision_scores[np.arange(len(predicted_labels)), class_indices]


def create_region_prediction_rows(
    timestep,
    file_source,
    cellids,
    coords_re,
    predicted_labels,
    score_name,
    prediction_scores,
    probabilities,
    class_labels,
    label_to_class,
):
    """
    Create CSV rows for one region-prediction batch.

    Parameters
    ----------
    timestep : int
        VLSV timestep.
    file_source : str
        Selected file source name.
    cellids : array-like of int
        Spatial cell IDs in the current batch.
    coords_re : numpy.ndarray
        Cell coordinates in Earth radii with shape ``(n_cells, 3)``.
    predicted_labels : array-like of int
        Predicted integer labels.
    score_name : str
        Human-readable prediction score name.
    prediction_scores : array-like of float
        Predicted-class scores.
    probabilities : numpy.ndarray or None
        Predicted class probabilities with shape ``(n_cells, n_classes)``.
    class_labels : array-like of int
        Model class labels in probability-column order.
    label_to_class : dict
        Mapping from integer label to class name.

    Returns
    -------
    list of dict
        CSV rows.
    """

    rows = []
    for index, cid in enumerate(cellids):
        predicted_label = int(predicted_labels[index])
        row = {
            "timestep": int(timestep),
            "file_source": file_source,
            "cid": int(cid),
            "x_re": float(coords_re[index, 0]),
            "y_re": float(coords_re[index, 1]),
            "z_re": float(coords_re[index, 2]),
            "predicted_label": predicted_label,
            "predicted_class_name": label_to_class[predicted_label],
            "score_name": score_name,
            "prediction_score": float(prediction_scores[index]),
            "max_class_probability": (
                float(probabilities[index].max())
                if probabilities is not None
                else float("nan")
            ),
        }

        if probabilities is not None:
            for class_index, class_label in enumerate(class_labels):
                class_name = label_to_class[int(class_label)]
                row[f"prob_{class_name}"] = float(
                    probabilities[index, class_index]
                )

        rows.append(row)

    return rows


def get_prediction_score(model, features, predicted_label):
    """
    Return the score for the predicted class.

    Parameters
    ----------
    model : sklearn estimator
        Fitted model or pipeline.
    features : numpy.ndarray
        Feature matrix for one VDF sample.
    predicted_label : int
        Label predicted by the model.

    Returns
    -------
    score_name : str
        Human-readable score name.
    score : float
        Predicted-class probability or decision score.
    """

    classes = list(model.classes_)
    class_index = classes.index(predicted_label)

    if hasattr(model, "predict_proba"):
        class_probabilities = model.predict_proba(features)[0]
        return "Predicted probability", float(class_probabilities[class_index])

    if hasattr(model, "decision_function"):
        decision_scores = np.asarray(model.decision_function(features))
        if len(classes) == 2 and decision_scores.size == 1:
            score = np.ravel(decision_scores)[0]
            if predicted_label == classes[0]:
                score = -score
        elif decision_scores.ndim == 0:
            score = decision_scores
        elif decision_scores.ndim == 1:
            score = decision_scores[0 if len(decision_scores) == 1 else class_index]
        else:
            score = decision_scores[0, class_index]

        return "Decision score", float(score)

    return "Prediction score", float("nan")
