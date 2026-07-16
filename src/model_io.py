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


def load_pytorch_convolutional_neural_network_classifier_model(
    model_dir,
    device="cpu",
):
    """
    Load a trained PyTorch CNN and preprocessing values.

    Parameters
    ----------
    model_dir : str
        Directory containing the model and preprocessing files.
    device : str or torch.device, optional
        Device used for CNN prediction.

    Returns
    -------
    model : src.pytorch_cnn.PyTorchCNNClassifier
        Trained PyTorch CNN classifier.
    preprocessing : numpy.lib.npyio.NpzFile
        Saved preprocessing values.
    """

    from src.pytorch_cnn import load_pytorch_cnn_checkpoint

    model_dir = Path(model_dir)
    model_path = (
        model_dir / "pytorch_convolutional_neural_network_classifier.pt"
    )
    preprocessing_path = model_dir / "preprocessing.npz"

    model = load_pytorch_cnn_checkpoint(
        checkpoint_path=model_path,
        device=device,
    )
    preprocessing = np.load(preprocessing_path)

    return model, preprocessing


def load_autoencoder_model(model_dir):
    """
    Load a trained VDF slice autoencoder and preprocessing values.

    Parameters
    ----------
    model_dir : str
        Directory containing the model and preprocessing files.

    Returns
    -------
    model : src.autoencoder.VdfSliceAutoencoder2D
        Trained autoencoder.
    preprocessing : numpy.lib.npyio.NpzFile
        Saved preprocessing values.
    """

    from src.autoencoder import load_autoencoder_checkpoint

    model_dir = Path(model_dir)
    model_path = model_dir / "autoencoder.pt"
    preprocessing_path = model_dir / "preprocessing.npz"

    model = load_autoencoder_checkpoint(
        checkpoint_path=model_path,
        device="cpu",
    )
    preprocessing = np.load(preprocessing_path)

    return model, preprocessing


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
