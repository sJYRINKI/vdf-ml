"""Load YAML mappings for the eight current project commands.

Subsystem-specific modules resolve defaults and numerical constraints after
this boundary; this module performs no scientific interpretation.
"""

from pathlib import Path

import yaml


def load_config(config_path):
    """
    Read one YAML configuration.

    Thin command modules call this boundary before invoking their package
    workflow. It performs only YAML decoding and returns the mapping intact,
    leaving each ordered workflow to access the fields it owns.

    Parameters
    ----------
    config_path : str or pathlib.Path
        Existing UTF-8 YAML file owned by one current workflow.

    Returns
    -------
    dict
        Parsed top-level configuration mapping.
    """

    with Path(config_path).open("r", encoding="utf-8") as config_file:
        return yaml.safe_load(config_file)

__all__ = ["load_config"]
