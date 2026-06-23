import numpy as np


def split_by_timestep(
    metadata,
    train_fraction=0.6,
    validation_fraction=0.2,
    gap_timesteps=10,
):
    """
    Split samples chronologically with gaps between dataset partitions.

    Parameters
    ----------
    metadata : pandas.DataFrame
        Dataset of metadata.
    train_fraction : float, optional
        Fraction of usable timesteps assigned to training.
    validation_fraction : float, optional
        Fraction of usable timesteps assigned to validation.
    gap_timesteps : int, optional
        Number of dataset timesteps excluded between consecutive partitions.

    Returns
    -------
    train_indices : numpy.ndarray
        Training sample indices.
    validation_indices : numpy.ndarray
        Validation sample indices.
    test_indices : numpy.ndarray
        Test sample indices.
    train_timesteps : numpy.ndarray
        Training timesteps.
    validation_timesteps : numpy.ndarray
        Validation timesteps.
    test_timesteps : numpy.ndarray
        Test timesteps.
    train_validation_gap : numpy.ndarray
        Timesteps excluded between training and validation.
    validation_test_gap : numpy.ndarray
        Timesteps excluded between validation and test.
    """

    train_fraction = float(train_fraction)
    validation_fraction = float(validation_fraction)
    gap_timesteps = int(gap_timesteps)

    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be between zero and one")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between zero and one")
    if train_fraction + validation_fraction >= 1.0:
        raise ValueError(
            "train_fraction + validation_fraction must be less than one"
        )
    if gap_timesteps < 0:
        raise ValueError("gap_timesteps must be non-negative")

    timesteps = np.array(sorted(metadata["timestep"].unique()))
    n_usable_timesteps = len(timesteps) - 2 * gap_timesteps
    if n_usable_timesteps < 3:
        raise ValueError(
            "Not enough timesteps for non-empty train, validation, and test "
            f"partitions after excluding two {gap_timesteps}-timestep gaps"
        )

    n_train = int(np.floor(n_usable_timesteps * train_fraction))
    n_validation = int(
        np.floor(n_usable_timesteps * validation_fraction)
    )
    n_test = n_usable_timesteps - n_train - n_validation
    if min(n_train, n_validation, n_test) < 1:
        raise ValueError(
            "Split fractions must leave at least one timestep in each "
            "partition"
        )

    train_end = n_train
    train_gap_end = train_end + gap_timesteps
    validation_end = train_gap_end + n_validation
    validation_gap_end = validation_end + gap_timesteps

    train_timesteps = timesteps[:train_end]
    train_validation_gap = timesteps[train_end:train_gap_end]
    validation_timesteps = timesteps[train_gap_end:validation_end]
    validation_test_gap = timesteps[validation_end:validation_gap_end]
    test_timesteps = timesteps[validation_gap_end:]

    train_indices = _indices_for_timesteps(metadata, train_timesteps)
    validation_indices = _indices_for_timesteps(
        metadata,
        validation_timesteps,
    )
    test_indices = _indices_for_timesteps(metadata, test_timesteps)

    return (
        train_indices,
        validation_indices,
        test_indices,
        train_timesteps,
        validation_timesteps,
        test_timesteps,
        train_validation_gap,
        validation_test_gap,
    )


def _indices_for_timesteps(metadata, timesteps):
    timestep_mask = metadata["timestep"].isin(timesteps)
    return np.flatnonzero(timestep_mask.to_numpy())
