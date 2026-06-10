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

def create_lobe_vs_rest_labels(metadata, positive_class_name="lobe"):
    """
    Create binary labels for lobe-vs-rest classification.

    Parameters
    ----------
    metadata : pandas.DataFrame
        Dataset metadata containing a ``class_name`` column.
    positive_class_name : str, optional
        Class name to map to 1. All other classes are mapped to 0.

    Returns
    -------
    numpy.ndarray
        Binary labels where lobe is 1 and rest is 0.
    """
    return (metadata["class_name"].to_numpy() == positive_class_name).astype("int64")
