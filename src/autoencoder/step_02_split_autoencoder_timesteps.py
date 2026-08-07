"""Stage 2: split autoencoder samples by complete timesteps.

This stage follows representation loading and precedes input normalization.
It applies the shared chronological split to sample identity only and
returns stable train, validation, and test row positions.

The stage receives :class:`AutoencoderTrainingData` plus split fractions
and gap length from the autoencoder configuration. It returns the shared
``CnnTimestepSplit`` record consumed by normalization and loader creation.
"""

from src.cnn.step_02_split_cnn_timesteps import (
    CnnTimestepSplit,
    split_by_timestep,
)


def split_autoencoder_timesteps(data, config):
    """Create chronological autoencoder partitions with boundary gaps.

    The autoencoder reuses the CNN's complete-timestep calculation so both
    workflows prevent temporal leakage identically. Only saved identity is
    passed to the split; reconstruction values remain on disk.

    Parameters
    ----------
    data : AutoencoderTrainingData
        Loaded representation and ``timestep`` sample identity.
    config : mapping
        Split fractions and gap size.

    Returns
    -------
    CnnTimestepSplit
        Stable sample indices, partition timesteps, and excluded gaps.
    """

    return CnnTimestepSplit(
        *split_by_timestep(
            data.sample_identity,
            train_fraction=config["train_fraction"],
            validation_fraction=config["validation_fraction"],
            gap_timesteps=config["gap_timesteps"],
        )
    )


__all__ = ["split_autoencoder_timesteps"]
