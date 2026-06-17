from pathlib import Path

import joblib
import numpy as np


def load_perceptron_model(model_dir):
    """
    Load a trained perceptron model and preprocessing values.

    Parameters
    ----------
    model_dir : str
        Directory containing the model and preprocessing files.

    Returns
    -------
    model : sklearn.linear_model.Perceptron
        Trained perceptron model.
    preprocessing : numpy.lib.npyio.NpzFile
        Saved preprocessing values.
    """

    return load_model(model_dir, "perceptron.joblib")


def load_logistic_regression_model(model_dir):
    """
    Load a trained logistic regression model and preprocessing values.

    Parameters
    ----------
    model_dir : str
        Directory containing the model and preprocessing files.

    Returns
    -------
    model : sklearn.pipeline.Pipeline
        Trained logistic regression model pipeline.
    preprocessing : numpy.lib.npyio.NpzFile
        Saved preprocessing values.
    """

    return load_model(model_dir, "logistic_regression.joblib")


def load_multilayer_perceptron_classifier_model(model_dir):
    """
    Load a trained multilayer perceptron classifier model and preprocessing values.

    Parameters
    ----------
    model_dir : str
        Directory containing the model and preprocessing files.

    Returns
    -------
    model : sklearn.pipeline.Pipeline
        Trained multilayer perceptron classifier model pipeline.
    preprocessing : numpy.lib.npyio.NpzFile
        Saved preprocessing values.
    """

    return load_model(model_dir, "multilayer_perceptron_classifier.joblib")


def load_model(model_dir, model_filename):
    """
    Load a trained model and preprocessing values from a model directory.

    Parameters
    ----------
    model_dir : str
        Directory containing the model and preprocessing files.
    model_filename : str
        Filename of the saved model joblib file.

    Returns
    -------
    model : object
        Trained model or model pipeline.
    preprocessing : numpy.lib.npyio.NpzFile
        Saved preprocessing values.
    """

    model_dir = Path(model_dir)
    model_path = model_dir / model_filename
    preprocessing_path = model_dir / "preprocessing.npz"

    model = joblib.load(model_path)
    preprocessing = np.load(preprocessing_path)

    return model, preprocessing
