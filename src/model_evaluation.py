import pandas as pd

def create_predictions_dataframe(metadata, indices, y_true, y_pred, split_name):
    """
    Create a predictions table

    Parameters
    ----------
    metadata : pandas.DataFrame
        Dataset metadata.
    indices : array like of int
        Sample indices.
    y_true : array like of int
        True labels.
    y_pred : array like of int
        Predicted labels.
    split_name : str
        Name of the split.

    Returns
    -------
    pandas.DataFrame
        prediction table.
    """

    predictions = metadata.iloc[indices].copy()

    predictions["split"] = split_name
    predictions["true_label"] = y_true
    predictions["predicted_label"] = y_pred
    predictions["correct"] = predictions["true_label"] == predictions["predicted_label"]

    return predictions