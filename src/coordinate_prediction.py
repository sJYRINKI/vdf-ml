import numpy as np
import analysator as pt

from src.dataset_plot import plot_vdf_xz_slice
from src.features import create_features
from src.timesteps import create_path
from src.vdf_extract import extract_vdf
from src.vdf_helpers import (
    create_coordinate_name,
    get_cellid_with_vdf,
    get_vdf_plot_parameters_from_file,
)


def predict_coordinate(config, timestep, model_id, coord_re, load_model):
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

    Returns
    -------
    dict
        Prediction result and output path.
    """

    timestep = int(timestep)
    file_location = create_path(
        path_template=config["file_template"],
        timestep=timestep,
    )
    model_dir = create_path(
        path_template=config["model_dir"],
        model_id=model_id,
        timestep=timestep,
    )
    output_dir = create_path(
        path_template=config["output_dir"],
        model_id=model_id,
        timestep=timestep,
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
