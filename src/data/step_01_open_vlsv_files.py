"""Stage 1: open VLSV paths and select file-scoped data producers.

This stage follows dataset orchestration and precedes VDF-cell discovery and
sample planning. It formats timestep paths and records the density,
velocity-mesh shape, and optional magnetic-field producer used repeatedly
while one reader remains open.

The returned records contain variable names and the ``[vx, vy, vz]``
velocity-mesh shape. Later stages read the selected variables directly.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.data.dense_vdf import resolve_velocity_population


DENSITY_CANDIDATES = ("rho", "proton/vg_rho", "vg_rho")
MAGNETIC_FIELD_CANDIDATES = ("vg_b_vol", "B_vol")
LEGACY_MAGNETIC_FIELD_CANDIDATES = ("B",)


@dataclass(frozen=True)
class VlsvVariableProducer:
    """Describe one stored VLSV variable selected for an operation.

    Stage 1 creates this immutable record once per open source file so
    sample-wise physical reads reuse an exact variable name, namespace,
    centring description, and expected unit string. The record describes the
    selected producer; it does not read or validate source values.

    Attributes
    ----------
    role : str
        Physical quantity supplied to the active operation.
    variable : str
        Exact VLSV variable name read from the source file.
    operator : str
        Analysator read operator, normally ``"pass"``.
    expected_units : str
        Units expected from the stored producer.
    population : str
        Particle population or namespace that owns the variable.
    stored_centering : str
        Spatial centering of the stored value.
    """

    role: str
    variable: str
    operator: str
    expected_units: str
    population: str
    stored_centering: str


@dataclass(frozen=True)
class MagneticFieldProducer:
    """Describe selected direct fields or legacy face-field reconstruction.

    Rotation-enabled Hermite source resolution stores this immutable choice
    before samples are extracted. The physical-context stage later either
    reads the listed cell-centred field directly or applies the named
    component-specific face-field reconstruction.

    Attributes
    ----------
    method : str
        Direct cell-centred read or supported polar-plane reconstruction.
    variables : tuple of VlsvVariableProducer
        Candidate fields read for the selected method.
    output_centering : str
        Cell centering of the returned three-component field.
    reconstruction_version : str or None
        Geometry convention name for reconstructed fields.
    """

    method: str
    variables: tuple[VlsvVariableProducer, ...]
    output_centering: str
    reconstruction_version: str | None


@dataclass(frozen=True)
class ResolvedVlsvSource:
    """Hold operation-specific VLSV producers selected once per reader.

    Stage 4 creates one instance when it opens a timestep file and reuses it
    for every VDF cell from that reader. It owns no file handle and contains
    only names, descriptive conventions, and the ``[vx, vy, vz]`` mesh shape
    needed by dense extraction and optional Hermite rotation reads.

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
    density : VlsvVariableProducer
        Selected density source.
    magnetic_field : MagneticFieldProducer or None
        Field source used by optional Hermite rotation.
    """

    reader_identity: int
    file_path: str | None
    population: str
    velocity_mesh_shape: tuple[int, int, int]
    density: VlsvVariableProducer
    magnetic_field: MagneticFieldProducer | None


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
    *,
    require_magnetic_field=False,
):
    """Select source variables and velocity shape for one open VLSV file.

    Stage 4 calls this once immediately after opening a timestep reader.
    Dense extraction then reuses the selected population and mesh shape.
    Optional Hermite rotation additionally reuses the magnetic-field source;
    raw-plot sparsity thresholds remain separate from this record. The
    returned producers describe successful inputs and do not read sample
    values.

    Parameters
    ----------
    reader : analysator.vlsvfile.VlsvReader
        Open VLSV file reader.
    population : str, optional
        Explicit VDF population. ``None`` uses the population resolver.
    require_magnetic_field : bool, optional
        Whether the magnetic-field producer for optional rotation is selected.

    Returns
    -------
    ResolvedVlsvSource
        File-scoped producers and velocity-mesh shape.
    """

    population = resolve_velocity_population(reader, population)
    velocity_mesh_shape = resolve_velocity_mesh_shape(reader, population)
    density_variable = _first_available(reader, DENSITY_CANDIDATES)
    density = _variable_producer(
        role="density",
        variable=density_variable,
        expected_units="1/m3",
        population=population,
    )
    magnetic_field = None
    if require_magnetic_field:
        magnetic_field = _resolve_magnetic_field_producer(
            reader,
            population=population,
        )
    return ResolvedVlsvSource(
        reader_identity=id(reader),
        file_path=_reader_file_path(reader),
        population=population,
        velocity_mesh_shape=velocity_mesh_shape,
        density=density,
        magnetic_field=magnetic_field,
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


def _resolve_magnetic_field_producer(reader, *, population):
    """Select the field source used by every Hermite sample in one file.

    Direct volume-average variables take precedence. When neither is present,
    the returned description selects the established
    ``legacy_b_polar_2d`` component-face reconstruction; actual field values
    are read later for each selected spatial cell.
    """

    direct_variables = tuple(
        variable
        for variable in MAGNETIC_FIELD_CANDIDATES
        if _variable_exists(reader, variable)
    )
    if direct_variables:
        return MagneticFieldProducer(
            method="direct cell-centered volume average",
            variables=tuple(
                _variable_producer(
                    role="magnetic field",
                    variable=variable,
                    expected_units="T",
                    population="unnamespaced",
                    stored_centering="cell_centered_volume_average",
                )
                for variable in direct_variables
            ),
            output_centering="selected_spatial_cell_center",
            reconstruction_version=None,
        )
    return MagneticFieldProducer(
        method="legacy polar 2D reconstruction",
        variables=(
            _variable_producer(
                role="magnetic field",
                variable=_first_available(
                    reader,
                    LEGACY_MAGNETIC_FIELD_CANDIDATES,
                ),
                expected_units="T",
                population="unnamespaced",
                stored_centering="face_centered_components",
            ),
        ),
        output_centering="selected_spatial_cell_center",
        reconstruction_version="legacy_b_polar_2d",
    )


def _first_available(reader, candidates):
    """Return the first available variable in scientific preference order.

    Candidate tuple order encodes producer priority, so stage 1 performs this
    lookup once and later sample reads remain deterministic.
    """

    return next(
        variable
        for variable in candidates
        if _variable_exists(reader, variable)
    )


def _variable_exists(reader, variable):
    """Query whether an open reader exposes one exact VLSV variable name."""

    return bool(reader.check_variable(variable))


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


def _variable_producer(
    *,
    role,
    variable,
    expected_units,
    population,
    stored_centering="selected_spatial_cell_native",
):
    """Create an immutable description of one selected VLSV producer.

    Expected units and centring are retained for downstream physical-context
    metadata. Constructing the record performs no source read or conversion.
    """

    return VlsvVariableProducer(
        role=role,
        variable=variable,
        operator="pass",
        expected_units=expected_units,
        population=population,
        stored_centering=stored_centering,
    )
