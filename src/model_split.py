import numpy as np


def split_by_timestep(metadata, train_fraction=0.7):
    """
    Split samples into train and test by timestep.

    Parameters
    ----------
    metadata : pandas.DataFrame
        Dataset of metadata.
    train_fraction : float, optional
        Fraction of timesteps used for training.

    Returns
    -------
    train_indices : numpy.ndarray
        sample training indices.
    test_indices : numpy.ndarray
        sample test indices.
    train_timesteps : numpy.ndarray
        sample training timesteps.
    test_timesteps : numpy.ndarray
        sample test timesteps.
    """
    timesteps = np.array(sorted(metadata["timestep"].unique()))
    if len(timesteps) < 2:
        raise ValueError("Need at least two timesteps for a train/test split")

    n_train = int(np.floor(len(timesteps) * train_fraction))
    n_train = max(1, min(n_train, len(timesteps) - 1))

    train_timesteps = timesteps[:n_train]
    test_timesteps = timesteps[n_train:]

    train_mask = metadata["timestep"].isin(train_timesteps)
    test_mask = metadata["timestep"].isin(test_timesteps)

    train_indices = metadata.index[train_mask].to_numpy()
    test_indices = metadata.index[test_mask].to_numpy()

    return train_indices, test_indices, train_timesteps, test_timesteps
