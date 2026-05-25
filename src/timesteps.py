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

def create_file_location(file_template, timestep):
    """
    create a VLSV file path from a file template and timestep.

    parameter
    ---------
    file_template : str
        File path template.
    timestep : int
        Timestep used to fill the tempalte.
    """

    return file_template.format(timestep=timestep)