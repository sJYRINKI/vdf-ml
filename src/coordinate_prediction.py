import csv

import numpy as np
import analysator as pt
from joblib import Parallel, delayed

from src.dataset_metadata import (
    POINT_REFERENCE_METADATA_COLUMNS,
    create_point_reference_metadata_arrays,
)
from src.dataset_io import load_velocity_grid_from_preprocessing
from src.dataset_plot import plot_vdf_xz_slice
from src.features import create_features
from src.point_topology import find_point_records
from src.timesteps import create_path, create_timestep_list
from src.vdf_extract import (
    VdfExtractor,
    create_trilinear_xz_plan,
    resolve_velocity_population,
)
from src.vdf_helpers import (
    R_EARTH,
    create_coordinate_name,
    create_region_mask_re,
    get_cellid_with_vdf,
    get_velocity_cell_size_from_extent,
    get_vdf_cells_with_coords_re,
    get_vdf_plot_threshold,
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

    model, preprocessing = load_model(model_dir)
    downsample_factor = int(preprocessing["downsample_factor"])
    log_eps = float(preprocessing["log_eps"])

    reader = pt.vlsvfile.VlsvReader(str(file_location))
    population = resolve_velocity_population(reader=reader)
    cid = get_cellid_with_vdf(
        reader=reader,
        coord_re=coord_re,
        pop=population,
    )
    extractor, target_velocity_grid, interpolation_plan = (
        create_velocity_resampling(
            reader=reader,
            population=population,
            preprocessing=preprocessing,
            config=config,
        )
    )
    xz_slice = extractor.extract_interpolated_xz(
        cid=cid,
        interpolation_plan=interpolation_plan,
    )
    velocity_resampling_path = save_velocity_resampling_metadata(
        output_dir=output_dir,
        model_id=model_id,
        source_velocity_grid=extractor.velocity_grid,
        target_velocity_grid=target_velocity_grid,
        interpolation_plan=interpolation_plan,
    )

    features = create_features(
        X=xz_slice[None, :, None, :],
        downsample_factor=downsample_factor,
        log_eps=log_eps,
    )
    vdf_coord_re = (
        np.asarray(reader.get_cell_coordinates(int(cid)), dtype=float)
        / R_EARTH
    )
    context_kwargs, point_reference_metadata = create_point_context_inputs(
        config=config,
        timestep=timestep,
        file_source=resolved_file_source,
        reader=reader,
        vdf_coords_re=vdf_coord_re.reshape(1, 3),
        model=model,
    )
    (
        predicted_labels,
        score_name,
        prediction_scores,
        _,
    ) = predict_batch_with_scores(
        model=model,
        features=features,
        prediction_kwargs=context_kwargs,
    )
    predicted_label = int(predicted_labels[0])
    prediction_score = float(prediction_scores[0])
    predicted_class_name = label_to_class[predicted_label]

    extent = target_velocity_grid["extent_mps"]
    dv = get_velocity_cell_size_from_extent(
        extent=extent,
        vdf_shape=target_velocity_grid["shape"],
    )
    threshold = get_vdf_plot_threshold(reader=reader, cid=cid)
    metadata_row = {
        "timestep": timestep,
        "cid": int(cid),
        "x_re": float(coord_re[0]),
        "y_re": float(coord_re[1]),
        "z_re": float(coord_re[2]),
        "vdf_x_re": float(vdf_coord_re[0]),
        "vdf_y_re": float(vdf_coord_re[1]),
        "vdf_z_re": float(vdf_coord_re[2]),
        "file_source": resolved_file_source,
        "file_location": str(file_location),
    }
    for column, values in point_reference_metadata.items():
        metadata_row[column] = values[0]

    coord_name = create_coordinate_name(coord_re)
    output_plot_path = output_dir / f"prediction_{coord_name}_xz.png"
    plot_vdf_xz_slice(
        vdf=xz_slice[:, None, :],
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

    result = {
        "output_plot_path": output_plot_path,
        "predicted_label": predicted_label,
        "predicted_class_name": predicted_class_name,
        "score_name": score_name,
        "prediction_score": prediction_score,
        "model_classes": model.classes_,
        "velocity_resampling_path": velocity_resampling_path,
        "velocity_population": population,
        "velocity_resampling": interpolation_plan["method_name"],
        "target_xz_center_coverage_fraction": interpolation_plan[
            "coverage_fraction"
        ],
    }
    if point_reference_metadata:
        result["point_reference_metadata"] = {
            column: values[0]
            for column, values in point_reference_metadata.items()
        }
    return result


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
    population = resolve_velocity_population(reader=reader)
    cellids, coords_re = get_vdf_cells_with_coords_re(
        reader=reader,
        pop=population,
    )
    extractor, target_velocity_grid, interpolation_plan = (
        create_velocity_resampling(
            reader=reader,
            population=population,
            preprocessing=preprocessing,
            config=config,
        )
    )
    velocity_resampling_path = save_velocity_resampling_metadata(
        output_dir=output_dir,
        model_id=model_id,
        source_velocity_grid=extractor.velocity_grid,
        target_velocity_grid=target_velocity_grid,
        interpolation_plan=interpolation_plan,
    )
    region_mask = create_region_mask_re(
        coords_re=coords_re,
        region_re=config["region_re"],
    )
    cellids = cellids[region_mask]
    coords_re = coords_re[region_mask]

    context_kwargs = {}
    point_reference_metadata = {}
    if len(cellids) > 0:
        context_kwargs, point_reference_metadata = create_point_context_inputs(
            config=config,
            timestep=timestep,
            file_source=resolved_file_source,
            reader=reader,
            vdf_coords_re=coords_re,
            model=model,
        )

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
    ]
    if model_uses_point_context(model):
        fieldnames.extend(POINT_REFERENCE_METADATA_COLUMNS)
    fieldnames.extend(class_probability_fields)

    with output_path.open("w", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()

        if len(cellids) > 0:
            for start in range(0, len(cellids), batch_size):
                end = start + batch_size
                batch_cellids = cellids[start:end]
                batch_coords_re = coords_re[start:end]
                xz_slices = np.asarray(
                    [
                        extractor.extract_interpolated_xz(
                            cid=int(cid),
                            interpolation_plan=interpolation_plan,
                        )
                        for cid in batch_cellids
                    ],
                    dtype=np.float32,
                )
                features = create_features(
                    X=xz_slices[:, :, None, :],
                    downsample_factor=downsample_factor,
                    log_eps=log_eps,
                    n_jobs=feature_n_jobs,
                )
                batch_context_kwargs = {
                    name: values[start:end]
                    for name, values in context_kwargs.items()
                }
                (
                    predicted_labels,
                    score_name,
                    prediction_scores,
                    probabilities,
                ) = predict_batch_with_scores(
                    model=model,
                    features=features,
                    prediction_kwargs=batch_context_kwargs,
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
                        point_reference_metadata={
                            column: values[start:end]
                            for column, values in point_reference_metadata.items()
                        },
                    )
                )

    return {
        "timestep": timestep,
        "file_source": resolved_file_source,
        "file_location": file_location,
        "output_path": output_path,
        "velocity_resampling_path": velocity_resampling_path,
        "n_selected_cells": int(len(cellids)),
        "velocity_population": population,
        "source_velocity_shape": tuple(extractor.vdf_shape),
        "target_velocity_shape": tuple(
            int(value) for value in target_velocity_grid["shape"]
        ),
        "velocity_resampling": interpolation_plan["method_name"],
        "target_xz_center_coverage_fraction": interpolation_plan[
            "coverage_fraction"
        ],
    }


def create_velocity_resampling(reader, population, preprocessing, config):
    """
    Create one extractor and cached target-plane interpolation plan.

    Parameters
    ----------
    reader : analysator.vlsvfile.VlsvReader
        Open source VLSV file reader.
    population : str
        Population name in the source VLSV file.
    preprocessing : numpy.lib.npyio.NpzFile or dict
        Model preprocessing values containing the training velocity grid.
    config : dict
        Prediction config containing the velocity-resampling policy.

    Returns
    -------
    extractor : src.vdf_extract.VdfExtractor
        Source VDF extractor.
    target_velocity_grid : dict
        Training velocity-grid descriptor.
    interpolation_plan : dict
        Cached target xz-plane interpolation plan.
    """

    extractor = VdfExtractor(reader=reader, pop=population)
    target_velocity_grid = load_velocity_grid_from_preprocessing(preprocessing)
    interpolation_plan = create_trilinear_xz_plan(
        source_grid=extractor.velocity_grid,
        target_grid=target_velocity_grid,
    )
    resampling_config = config.get("velocity_resampling", {})
    if not isinstance(resampling_config, dict):
        raise TypeError("velocity_resampling must be a dictionary")
    mode = str(resampling_config.get("mode", "auto")).lower()
    method = str(resampling_config.get("method", "trilinear")).lower()
    outside_source = str(
        resampling_config.get("outside_source", "zero")
    ).lower()
    minimum_coverage = float(
        resampling_config.get(
            "minimum_xz_center_coverage_fraction",
            0.0,
        )
    )
    if mode not in {"auto", "require_same_grid"}:
        raise ValueError(
            "velocity_resampling.mode must be 'auto' or 'require_same_grid'"
        )
    if method != "trilinear":
        raise ValueError("velocity_resampling.method must be 'trilinear'")
    if outside_source != "zero":
        raise ValueError("velocity_resampling.outside_source must be 'zero'")
    if minimum_coverage < 0.0 or minimum_coverage > 1.0:
        raise ValueError(
            "minimum_xz_center_coverage_fraction must be between zero and one"
        )
    if mode == "require_same_grid" and not interpolation_plan["identity"]:
        raise ValueError(
            "Source and training velocity grids differ while "
            "velocity_resampling.mode is require_same_grid"
        )
    if interpolation_plan["coverage_fraction"] < minimum_coverage:
        raise ValueError(
            "Target xz center coverage is below the configured minimum: "
            f"{interpolation_plan['coverage_fraction']:.6f} < "
            f"{minimum_coverage:.6f}"
        )

    method_name = (
        "identity"
        if interpolation_plan["identity"]
        else "trilinear_raw_zero_fill"
    )
    interpolation_plan["method_name"] = method_name
    print(
        f"Velocity population: {population}; "
        f"source shape: {tuple(extractor.vdf_shape)}; "
        f"target shape: "
        f"{tuple(int(value) for value in target_velocity_grid['shape'])}; "
        f"resampling: {method_name}; "
        f"target xz coverage: "
        f"{interpolation_plan['coverage_fraction']:.6f}"
    )

    return extractor, target_velocity_grid, interpolation_plan


def save_velocity_resampling_metadata(
    output_dir,
    model_id,
    source_velocity_grid,
    target_velocity_grid,
    interpolation_plan,
):
    """
    Save one prediction velocity-resampling provenance sidecar.

    Parameters
    ----------
    output_dir : str or pathlib.Path
        Prediction output directory.
    model_id : str
        Trained model identifier.
    source_velocity_grid : dict
        Native VLSV velocity-grid descriptor.
    target_velocity_grid : dict
        Training velocity-grid descriptor.
    interpolation_plan : dict
        Cached interpolation plan used for prediction.

    Returns
    -------
    pathlib.Path
        Saved provenance archive path.
    """

    output_path = output_dir / "velocity_resampling.npz"
    np.savez(
        output_path,
        model_id=np.asarray(str(model_id)),
        method=np.asarray(interpolation_plan["method_name"]),
        outside_source=np.asarray("zero"),
        source_population=np.asarray(source_velocity_grid["population"]),
        source_shape=np.asarray(source_velocity_grid["shape"], dtype=np.int64),
        source_extent_mps=np.asarray(
            source_velocity_grid["extent_mps"],
            dtype=np.float64,
        ),
        target_population=np.asarray(target_velocity_grid["population"]),
        target_shape=np.asarray(target_velocity_grid["shape"], dtype=np.int64),
        target_extent_mps=np.asarray(
            target_velocity_grid["extent_mps"],
            dtype=np.float64,
        ),
        target_slice_axis=np.asarray(target_velocity_grid["slice_axis"]),
        target_slice_index=np.asarray(
            target_velocity_grid["slice_index"],
            dtype=np.int64,
        ),
        target_xz_center_coverage_fraction=np.asarray(
            interpolation_plan["coverage_fraction"],
            dtype=np.float64,
        ),
    )

    return output_path


def model_uses_point_context(model):
    """
    Return whether a model requires point-distance or point-vector inputs.

    Parameters
    ----------
    model : object
        Loaded classifier.

    Returns
    -------
    bool
        Whether point-context inputs are required during prediction.
    """

    if hasattr(model, "requires_point_context"):
        return bool(model.requires_point_context)

    return bool(
        tuple(getattr(model, "distance_feature_names", ()))
        or tuple(getattr(model, "vector_feature_names", ()))
    )


def create_point_context_inputs(
    config,
    timestep,
    file_source,
    reader,
    vdf_coords_re,
    model,
):
    """
    Create all-nearest point geometry required by a trained model.

    Parameters
    ----------
    config : dict
        Prediction config containing matching flux templates and point-search
        settings.
    timestep : int
        Timestep being predicted.
    file_source : str
        Selected bulk-file source name.
    reader : analysator.vlsvfile.VlsvReader
        Open reader for the matching bulk file.
    vdf_coords_re : numpy.ndarray
        VDF-cell-center coordinates with shape ``(n_samples, 3)``.
    model : object
        Loaded classifier.

    Returns
    -------
    prediction_kwargs : dict
        Distance/vector matrices keyed by model prediction argument name.
    point_reference_metadata : dict
        All-nearest metadata arrays in sample order.
    """

    if not model_uses_point_context(model):
        return {}, {}

    distance_feature_names = tuple(
        getattr(model, "distance_feature_names", ())
    )
    vector_feature_names = tuple(
        getattr(model, "vector_feature_names", ())
    )
    if not distance_feature_names and not vector_feature_names:
        return {}, {}

    points_config = config.get("points")
    if not isinstance(points_config, dict):
        raise ValueError(
            "Topology-enabled prediction requires a points configuration"
        )
    flux_file_template = resolve_flux_file_template(
        config=config,
        file_source=file_source,
    )
    flux_file_location = create_path(
        path_template=flux_file_template,
        timestep=int(timestep),
    )
    if not flux_file_location.is_file():
        raise FileNotFoundError(flux_file_location)

    x_point_records, o_point_records = find_point_records(
        reader=reader,
        flux_file_location=flux_file_location,
        points_config=points_config,
    )
    point_reference_metadata = create_point_reference_metadata_arrays(
        vdf_coords_re=vdf_coords_re,
        x_point_coords_re=[
            point_record["coord_re"]
            for point_record in x_point_records
        ],
        o_point_coords_re=[
            point_record["coord_re"]
            for point_record in o_point_records
        ],
    )

    prediction_kwargs = {}
    for argument_name, feature_names in (
        ("distance_features", distance_feature_names),
        ("vector_features", vector_feature_names),
    ):
        if not feature_names:
            continue
        missing_columns = sorted(
            set(feature_names) - set(point_reference_metadata)
        )
        if missing_columns:
            raise ValueError(
                f"Model point-context columns are unavailable: {missing_columns}"
            )
        prediction_kwargs[argument_name] = np.column_stack(
            [
                point_reference_metadata[column]
                for column in feature_names
            ]
        ).astype(np.float32, copy=False)

    return prediction_kwargs, point_reference_metadata


def resolve_flux_file_template(config, file_source):
    """
    Return the flux-file template for one prediction source.

    Parameters
    ----------
    config : dict
        Prediction config containing one flux template or a mapping of flux
        templates by file source.
    file_source : str
        Selected bulk-file source name.

    Returns
    -------
    str
        Flux-file path template for the selected source.
    """

    flux_file_templates = config.get("flux_file_templates")
    if flux_file_templates is not None:
        if file_source not in flux_file_templates:
            raise ValueError(
                "Topology-enabled prediction has no flux template for "
                f"file_source {file_source!r}"
            )
        return flux_file_templates[file_source]

    flux_file_template = config.get("flux_file_template")
    if flux_file_template is None:
        raise ValueError(
            "Topology-enabled prediction requires flux_file_templates or "
            "flux_file_template"
        )
    return flux_file_template


def predict_batch_with_scores(model, features, prediction_kwargs=None):
    """
    Predict one batch and avoid duplicate probability forward passes.

    Parameters
    ----------
    model : object
        Classifier with ``predict`` and optional ``predict_proba`` methods.
    features : numpy.ndarray
        VDF feature matrix for one prediction batch.
    prediction_kwargs : dict, optional
        Additional model inputs such as distance and vector matrices.

    Returns
    -------
    predicted_labels : numpy.ndarray
        Predicted project labels.
    score_name : str
        Human-readable prediction score name.
    prediction_scores : numpy.ndarray
        Predicted-class scores for each sample.
    probabilities : numpy.ndarray or None
        Class probabilities when supported by the model.
    """

    prediction_kwargs = prediction_kwargs or {}
    if hasattr(model, "predict_proba"):
        probabilities = np.asarray(
            model.predict_proba(features, **prediction_kwargs)
        )
        class_indices = np.argmax(probabilities, axis=1)
        predicted_labels = np.asarray(model.classes_)[class_indices]
        prediction_scores = probabilities[
            np.arange(len(predicted_labels)),
            class_indices,
        ]
        return (
            predicted_labels,
            "Predicted probability",
            prediction_scores.astype(float),
            probabilities,
        )

    predicted_labels = model.predict(features, **prediction_kwargs)
    score_name, prediction_scores, probabilities = get_batch_prediction_scores(
        model=model,
        features=features,
        predicted_labels=predicted_labels,
    )
    return predicted_labels, score_name, prediction_scores, probabilities


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
    point_reference_metadata=None,
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
    point_reference_metadata : dict, optional
        Point-distance and vector arrays for the batch.

    Returns
    -------
    list of dict
        CSV rows.
    """

    point_reference_metadata = point_reference_metadata or {}
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

        for column, values in point_reference_metadata.items():
            row[column] = float(values[index])

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
