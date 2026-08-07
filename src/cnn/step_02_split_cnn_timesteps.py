"""Stage 2: split CNN samples into chronological timestep partitions.

This stage follows saved-dataset loading and precedes normalization. It
assigns complete timesteps to training, validation, and test partitions,
leaving configured gaps between them, and returns stable sample indices for
all later training stages.

The stage receives the metadata held by :class:`CnnTrainingData` and split
fractions. It returns indices and timestep arrays for training, validation,
test, and both excluded boundary gaps.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CnnTimestepSplit:
    """Hold chronological sample indices, timesteps, and excluded gaps.

    Stage 2 creates this immutable record and later training stages consume
    its row-index arrays without reordering them. Keeping complete
    timesteps together prevents samples from one simulation state appearing
    in more than one model partition.

    Parameters
    ----------
    train_indices : numpy.ndarray
        Saved row positions assigned to model fitting.
    validation_indices : numpy.ndarray
        Saved row positions used for epoch selection.
    test_indices : numpy.ndarray
        Saved row positions used for final evaluation.
    train_timesteps : numpy.ndarray
        Timesteps represented by ``train_indices``.
    validation_timesteps : numpy.ndarray
        Timesteps represented by ``validation_indices``.
    test_timesteps : numpy.ndarray
        Timesteps represented by ``test_indices``.
    train_validation_gap : numpy.ndarray
        Timesteps excluded between training and validation.
    validation_test_gap : numpy.ndarray
        Timesteps excluded between validation and test.
    """

    train_indices: np.ndarray
    validation_indices: np.ndarray
    test_indices: np.ndarray
    train_timesteps: np.ndarray
    validation_timesteps: np.ndarray
    test_timesteps: np.ndarray
    train_validation_gap: np.ndarray
    validation_test_gap: np.ndarray


def split_cnn_timesteps(
    data,
    *,
    train_fraction=0.6,
    validation_fraction=0.2,
    gap_timesteps=10,
):
    """Split loaded CNN data by complete chronological timesteps.

    This stage-level adapter extracts metadata from the loaded training
    record and delegates the numerical boundary calculation to
    :func:`split_by_timestep`. Stage 3 consumes the returned training
    indices to fit all normalization values.

    Parameters
    ----------
    data : CnnTrainingData
        Loaded saved metadata and model targets.
    train_fraction : float, optional
        Fraction of usable timesteps assigned to training.
    validation_fraction : float, optional
        Fraction of usable timesteps assigned to validation.
    gap_timesteps : int, optional
        Number of unique timesteps excluded at each partition boundary.

    Returns
    -------
    CnnTimestepSplit
        Stable partition indices, timesteps, and excluded gaps.
    """

    return CnnTimestepSplit(
        *split_by_timestep(
            data.metadata,
            train_fraction=train_fraction,
            validation_fraction=validation_fraction,
            gap_timesteps=gap_timesteps,
        )
    )


def split_by_timestep(
    metadata,
    train_fraction=0.6,
    validation_fraction=0.2,
    gap_timesteps=10,
):
    """Calculate chronological sample partitions from saved timesteps.

    Unique timesteps are sorted, two configured boundary gaps are removed,
    and the remaining leading fractions define training and validation.
    Every metadata row from a selected timestep is returned in its original
    saved order.

    Parameters
    ----------
    metadata : pandas.DataFrame
        Dataset rows containing a ``timestep`` column.
    train_fraction : float, optional
        Fraction of usable timesteps assigned to training.
    validation_fraction : float, optional
        Fraction of usable timesteps assigned to validation.
    gap_timesteps : int, optional
        Number of timesteps excluded between adjacent partitions.

    Returns
    -------
    tuple
        Training, validation, and test sample-index arrays; their three
        timestep arrays; and the train-validation and validation-test gap
        timestep arrays, in that order.
    """

    train_fraction = float(train_fraction)
    validation_fraction = float(validation_fraction)
    gap_timesteps = int(gap_timesteps)
    timesteps = np.array(sorted(metadata["timestep"].unique()))
    usable_count = len(timesteps) - 2 * gap_timesteps
    train_count = int(np.floor(usable_count * train_fraction))
    validation_count = int(np.floor(usable_count * validation_fraction))

    train_end = train_count
    train_gap_end = train_end + gap_timesteps
    validation_end = train_gap_end + validation_count
    validation_gap_end = validation_end + gap_timesteps

    train_timesteps = timesteps[:train_end]
    train_validation_gap = timesteps[train_end:train_gap_end]
    validation_timesteps = timesteps[train_gap_end:validation_end]
    validation_test_gap = timesteps[validation_end:validation_gap_end]
    test_timesteps = timesteps[validation_gap_end:]

    return (
        _indices_for_timesteps(metadata, train_timesteps),
        _indices_for_timesteps(metadata, validation_timesteps),
        _indices_for_timesteps(metadata, test_timesteps),
        train_timesteps,
        validation_timesteps,
        test_timesteps,
        train_validation_gap,
        validation_test_gap,
    )


def _indices_for_timesteps(metadata, timesteps):
    """Return saved row positions belonging to selected complete timesteps.

    Parameters
    ----------
    metadata : pandas.DataFrame
        Dataset rows in saved sample order.
    timesteps : array-like
        Simulation timesteps assigned to one partition.

    Returns
    -------
    numpy.ndarray
        Zero-based metadata row positions in unchanged saved order.
    """

    return np.flatnonzero(metadata["timestep"].isin(timesteps).to_numpy())


__all__ = [
    "CnnTimestepSplit",
    "split_by_timestep",
    "split_cnn_timesteps",
]
