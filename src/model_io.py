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
    model_dir = Path(model_dir)

    model_path = model_dir / "perceptron.joblib"
    preprocessing_path = model_dir / "preprocessing.npz"

    model = joblib.load(model_path)
    preprocessing = np.load(preprocessing_path)

    return model, preprocessing