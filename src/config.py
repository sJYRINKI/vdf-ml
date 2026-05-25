from pathlib import Path
import yaml

def load_config(config_path):
    """
    Read a YAML config file.

    parameters
    ----------
    config_path : str
        Path to YAML config file.

    Returns
    -------
    dict
        Config values loaded from the YAML fiel. 
    """

    config_path = Path(config_path)

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config