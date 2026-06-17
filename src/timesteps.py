from pathlib import Path


def create_timestep_list(start_timestep, n_timesteps):
    """
    Create a list of timesteps.

    Parameters
    ----------
    start_timestep : int
        First timestep.
    n_timesteps : int
        Number of timesteps to include.

    Returns
    -------
    List of int
        Timesteps starting from ``start_timestep``
    """
    return list(range(start_timestep, start_timestep + n_timesteps))


def create_path(path_template, **values):
    """
    Create a file path from a file template and named values.

    Parameter
    ---------
    path_template : str
        Path template.
    **values : dict
        Named values used to fill the template.

    Return
    ------
    pathlib.Path
        Path for the given timestep.
    """

    return Path(path_template.format(**values))


def create_timestep_path(path_template, timestep):
    """
    Create a file path from a file template and timestep.

    Parameter
    ---------
    path_template : str
        Path template.
    timestep : int
        Timestep used to fill the template.

    Return
    ------
    pathlib.Path
        Path for the given timestep.
    """

    return create_path(
        path_template,
        timestep=timestep
    )
