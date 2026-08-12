"""Stage 1: open VLSV paths and select file-scoped data producers.

This stage follows dataset orchestration and precedes VDF-cell discovery and
sample planning. It formats timestep paths and records the configured VDF
population and velocity-mesh shape used repeatedly while one reader remains
open. Same-cell physical producers are resolved separately by the focused
plasma-context physics module.

The returned records contain source identity, population, and the
``[vx, vy, vz]`` velocity-mesh shape. Later stages resolve and read the
selected physical variables directly.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.data.dense_vdf import resolve_velocity_population


@dataclass(frozen=True)
class ResolvedVlsvSource:
    """Hold VDF source identity and mesh values resolved once per reader.

    Stage 4 creates one instance when it opens a timestep file and reuses it
    for every VDF cell from that reader. It owns no file handle and contains
    only source identity, population, and the ``[vx, vy, vz]`` mesh shape
    needed by dense VDF extraction. Plasma producer names remain owned by
    ``src.physics.plasma_context``.

    Attributes
    ----------
    reader_identity : int
        Python identity of the open reader that owns these producers.
    file_path : str or None
        Source path exposed by the reader when available.
    population : str
        Selected VDF population.
    velocity_mesh_shape : tuple of int
        Velocity-cell shape in ``[vx, vy, vz]`` order.
    """

    reader_identity: int
    file_path: str | None
    population: str
    velocity_mesh_shape: tuple[int, int, int]


def create_timestep_list(start_timestep, n_timesteps):
    """Create consecutive timestep values.

    Stage 0 uses this ordered sequence for both sample planning and final row
    order. No source files are opened by this calculation.

    Parameters
    ----------
    start_timestep : int
        First timestep.
    n_timesteps : int
        Number of timesteps to include.

    Returns
    -------
    list of int
        Consecutive timesteps starting from ``start_timestep``.
    """

    return list(range(start_timestep, start_timestep + n_timesteps))


def create_path(path_template, **values):
    """Format one configured source path.

    Dataset and plotting workflows use the same formatting rule for bulk and
    magnetic-flux paths. Returning a ``Path`` keeps later file opening
    explicit while this function performs only placeholder substitution.

    Parameters
    ----------
    path_template : str
        Python format string for the source path.
    **values : object
        Named values substituted into ``path_template``.

    Returns
    -------
    pathlib.Path
        Formatted source path.
    """

    return Path(path_template.format(**values))


def create_timestep_path(path_template, timestep):
    """Format one configured source path for a timestep.

    This specialization keeps the timestep placeholder convention identical
    across planning, topology detection, extraction, and plotting.

    Parameters
    ----------
    path_template : str
        Python format string containing a ``timestep`` field.
    timestep : int
        Timestep substituted into the path.

    Returns
    -------
    pathlib.Path
        Formatted timestep path.
    """

    return create_path(path_template, timestep=timestep)


def resolve_vlsv_source(
    reader,
    population=None,
):
    """Select the VDF population and velocity shape for one open source.

    Stage 4 calls this once immediately after opening a timestep reader.
    Dense extraction reuses the selected population and mesh shape, while the
    plasma-context module independently resolves its field and moment sources
    once for the same reader.

    Parameters
    ----------
    reader : analysator.vlsvfile.VlsvReader
        Open VLSV file reader.
    population : str, optional
        Explicit VDF population. ``None`` uses the population resolver.
    Returns
    -------
    ResolvedVlsvSource
        File-scoped VDF population, identity, and velocity-mesh shape.
    """

    population = resolve_velocity_population(reader, population)
    velocity_mesh_shape = resolve_velocity_mesh_shape(reader, population)
    return ResolvedVlsvSource(
        reader_identity=id(reader),
        file_path=_reader_file_path(reader),
        population=population,
        velocity_mesh_shape=velocity_mesh_shape,
    )


def resolve_velocity_mesh_shape(reader, population):
    """Calculate the three-dimensional velocity-cell shape.

    Analysator exposes velocity-block counts and cells per block separately.
    Multiplying them once gives the dense ``[vx, vy, vz]`` allocation shape
    shared by sparse expansion and velocity-grid storage.

    Parameters
    ----------
    reader : analysator.vlsvfile.VlsvReader
        Open VLSV file reader.
    population : str
        Selected VDF population.

    Returns
    -------
    tuple of int
        Velocity-cell shape in ``[vx, vy, vz]`` order.
    """

    mesh_size = np.asarray(
        reader.get_velocity_mesh_size(population),
        dtype=np.int64,
    )
    block_size = np.asarray(
        reader.get_velocity_block_size(population),
        dtype=np.int64,
    )
    return tuple(int(value) for value in mesh_size * block_size)


def _reader_file_path(reader):
    """Return the first source-path attribute exposed by an open reader.

    The value is descriptive metadata only; stage 1 already owns the reader
    and does not reopen the path returned here.
    """

    values = (
        getattr(reader, attribute, None)
        for attribute in (
            "file_path",
            "file_location",
            "filename",
            "file_name",
        )
    )
    return next(
        (str(Path(value)) for value in values if value is not None),
        None,
    )
