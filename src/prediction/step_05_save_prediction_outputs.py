"""Stage 5: serialize coordinate and region predictions to shared CSV rows.

This final stage follows CNN inference. It combines decoded class and
topology predictions with requested and selected coordinates in Earth radii,
then writes the same stable CSV schema for one coordinate or a streamed
region. No plotting data or alternate JSON serialization is persisted.
"""

from pathlib import Path
import re

import numpy as np
import pandas as pd

from src.data.metadata_columns import TOPOLOGY_TARGET_COLUMNS


PREDICTED_TOPOLOGY_COLUMNS = tuple(
    f"predicted_{name}" for name in TOPOLOGY_TARGET_COLUMNS
)


def create_prediction_rows(
    batch,
    *,
    representation,
    timestep,
    file_source,
    cellids,
    coordinates_re,
    loaded,
):
    """Create stable scalar rows for coordinate or region output.

    This final-stage formatter aligns every decoded batch row with its VLSV
    cell identity and cell-centre coordinate. It expands checkpoint-ordered
    class probabilities and the fixed six topology values into named scalar
    fields suitable for either one-row coordinate or streamed region CSV.
    Requested-coordinate fields are part of the common schema and remain
    empty for region rows, which have no single requested position.

    Parameters
    ----------
    batch : PredictionBatch
        Decoded class and physical topology predictions.
    representation : {"raw", "hermite"}
        Representation used by the checkpoint.
    timestep : int
        Simulation timestep represented by all rows.
    file_source : str
        Configured source identifier.
    cellids : sequence of int
        Selected VLSV spatial cell IDs.
    coordinates_re : numpy.ndarray
        Selected cell centers with shape ``(n_samples, 3)`` in Earth radii.
    loaded : LoadedCnnCheckpoint
        Checkpoint defining class probability order.

    Returns
    -------
    list of dict
        CSV-safe rows in selected-cell order. Requested and selected
        coordinates, topology distances, and vector components are expressed
        in Earth radii.
    """

    cellids = np.asarray(cellids, dtype=np.int64)
    coordinates_re = np.asarray(coordinates_re, dtype=np.float64)
    class_names = tuple(
        str(value)
        for value in loaded.checkpoint["class_mapping"]["class_names"]
    )
    probability_names = tuple(
        _probability_column_name(name) for name in class_names
    )
    rows = []
    for sample_index in range(len(cellids)):
        row = {
            "timestep": int(timestep),
            "file_source": str(file_source),
            "cid": int(cellids[sample_index]),
            "requested_x_re": None,
            "requested_y_re": None,
            "requested_z_re": None,
            "selected_x_re": float(coordinates_re[sample_index, 0]),
            "selected_y_re": float(coordinates_re[sample_index, 1]),
            "selected_z_re": float(coordinates_re[sample_index, 2]),
            "representation": str(representation),
            "predicted_class_id": int(batch.class_ids[sample_index]),
            "predicted_class_name": batch.class_names[sample_index],
            "predicted_probability": float(
                batch.maximum_probabilities[sample_index]
            ),
        }
        for class_index, column in enumerate(probability_names):
            row[column] = float(
                batch.probabilities[sample_index, class_index]
            )
        for target_index, column in enumerate(
            PREDICTED_TOPOLOGY_COLUMNS
        ):
            row[column] = float(
                batch.topology_values[sample_index, target_index]
            )
        rows.append(row)
    return rows


def prediction_column_names(loaded):
    """Return the stable coordinate and region CSV column order.

    Probability columns follow the explicit checkpoint class order, after
    which the six topology columns follow the frozen topology target order.
    Both workflows call this once when their shared CSV writer is opened.

    Parameters
    ----------
    loaded : LoadedCnnCheckpoint
        Checkpoint defining class probability order.

    Returns
    -------
    tuple of str
        Identity, class, probability, and six physical topology columns.
    """

    class_names = tuple(
        str(value)
        for value in loaded.checkpoint["class_mapping"]["class_names"]
    )
    return (
        "timestep",
        "file_source",
        "cid",
        "requested_x_re",
        "requested_y_re",
        "requested_z_re",
        "selected_x_re",
        "selected_y_re",
        "selected_z_re",
        "representation",
        "predicted_class_id",
        "predicted_class_name",
        "predicted_probability",
        *tuple(_probability_column_name(name) for name in class_names),
        *PREDICTED_TOPOLOGY_COLUMNS,
    )


def write_prediction_csv(path, rows, loaded):
    """Write one timestep of stable coordinate or region prediction rows.

    Coordinate prediction calls this direct writer with one row before
    optional figure rendering. Large-region orchestration instead uses
    :func:`open_prediction_csv` to append bounded batches with the identical
    field order, so both commands expose one CSV contract.

    Parameters
    ----------
    path : str or pathlib.Path
        Destination CSV path.
    rows : iterable of mapping
        Prediction rows in selected-cell order. Coordinate output has one
        row; region output may contain any selected count.
    loaded : LoadedCnnCheckpoint
        Checkpoint defining the stable class-probability columns.

    Returns
    -------
    pathlib.Path
        Written CSV path.
    """

    with open_prediction_csv(path, loaded) as output:
        output.write_rows(rows)
    return Path(path)


def open_prediction_csv(path, loaded):
    """Open one prediction CSV for shared DataFrame row writing.

    The returned context manager writes the header immediately and keeps a
    single file handle open while Stage 5 receives one coordinate row or
    successive bounded region batches. Each batch uses the same ordered
    pandas DataFrame serialization with ``index=False``.

    Parameters
    ----------
    path : str or pathlib.Path
        Destination CSV path.
    loaded : LoadedCnnCheckpoint
        Checkpoint defining stable class-probability columns.

    Returns
    -------
    object
        Context manager whose ``write_rows`` method appends prediction rows.
    """

    return _PredictionCsvOutput(path, loaded)


class _PredictionCsvOutput:
    """Own one open coordinate or region prediction CSV.

    The private context manager creates the destination and DataFrame header
    on entry, accepts already-decoded rows in selected-cell order, and closes
    the sole file handle on exit. Region batches remain bounded in memory.
    """

    def __init__(self, path, loaded):
        """Store the destination and checkpoint-defined column order.

        Parameters
        ----------
        path : str or pathlib.Path
            Per-timestep CSV destination.
        loaded : LoadedCnnCheckpoint
            Checkpoint whose class order defines probability columns.
        """

        self.path = Path(path)
        self.columns = prediction_column_names(loaded)
        self.output_file = None

    def __enter__(self):
        """Open the CSV, write its stable header, and return this writer."""

        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.output_file = self.path.open(
            "w",
            newline="",
            encoding="utf-8",
        )
        pd.DataFrame(
            columns=self.columns,
        ).to_csv(
            self.output_file,
            index=False,
        )
        return self

    def __exit__(self, exception_type, exception, traceback):
        """Close the CSV handle when the streaming timestep block ends."""

        self.output_file.close()
        return False

    def write_rows(self, rows):
        """Append decoded prediction rows through the shared DataFrame path.

        Parameters
        ----------
        rows : iterable of mapping
            Scalar Stage 5 rows whose keys match the open CSV header.
        """

        pd.DataFrame(
            rows,
            columns=self.columns,
        ).to_csv(
            self.output_file,
            index=False,
            header=False,
        )


def _probability_column_name(class_name):
    """Convert one physical class name to a stable probability column.

    Parameters
    ----------
    class_name : str
        Checkpoint class name in physical model-output order.

    Returns
    -------
    str
        Lowercase ``probability_<class>`` CSV field name.
    """

    safe = re.sub(
        r"[^a-z0-9]+",
        "_",
        class_name.strip().lower(),
    ).strip("_")
    return f"probability_{safe}"


__all__ = [
    "PREDICTED_TOPOLOGY_COLUMNS",
    "create_prediction_rows",
    "open_prediction_csv",
    "prediction_column_names",
    "write_prediction_csv",
]
