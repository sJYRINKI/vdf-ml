"""Resolve and read same-cell vector plasma context.

Dataset extraction and VLSV prediction resolve one producer family per open
reader and reuse it for every selected cell. Magnetic-field candidates are
``vg_b_vol``, ``B_vol``, ``fg_b_vol``, ``fg_b``, and ``B``; electric-field
candidates are ``vg_e_vol``, ``E_vol``, ``fg_e_vol``, ``fg_e``, and ``E``.
Configured-population velocity prefers ``<population>/vg_v`` and
``<population>/V`` before unprefixed direct values and legacy ``rho_v/rho``.
Number density prefers ``<population>/vg_rho``, ``<population>/rho``, and
``rho``. Total pressure uses a complete modern or historical tensor family,
or reconstructs it by summing complete thermal/nonthermal or
backstream/nonbackstream tensor pairs.

SpatialGrid values are read at the VDF cell ID. Field-grid values are sampled
at that cell's centre with Analysator's fsgrid indexing and centering
operations. Those Analysator operations transiently materialize one field
grid; only selected vectors are retained until their same-cell read, then each
vector entry is discarded. No spatial field map is saved or passed to a model.
Legacy face-centred ``B`` uses the established project cell-centre
reconstruction.

Complete same-cell magnetic field in tesla, electric field in volts per metre,
and configured-population bulk velocity in metres per second are stored in
Cartesian ``(x, y, z)`` order. The saved ``plasma_context.npy`` is an aligned
float32 array with shape ``(n_samples, 16)`` and columns ``(Bx, By, Bz, Ex, Ey,
Ez, Vx, Vy, Vz, n, Pxx, Pyy, Pzz, Pxy, Pxz, Pyz)``. Density is in particles per
cubic metre and pressure is in pascals. Separate vector magnitudes are not
stored because the components retain magnitude and direction. Magnetic-field
and velocity vectors are reused directly for optional Hermite rotation.
"""

import numpy as np

from src.physics.magnetic_field import get_cell_centered_magnetic_field
from src.physics.vlsv_physical_values import (
    read_bulk_velocity,
    read_scalar,
    read_vector,
    resolve_bulk_velocity_source,
)


PLASMA_CONTEXT_FEATURE_NAMES = (
    "magnetic_field_x_t",
    "magnetic_field_y_t",
    "magnetic_field_z_t",
    "electric_field_x_vm",
    "electric_field_y_vm",
    "electric_field_z_vm",
    "bulk_velocity_x_ms",
    "bulk_velocity_y_ms",
    "bulk_velocity_z_ms",
    "number_density_m3",
    "pressure_xx_pa",
    "pressure_yy_pa",
    "pressure_zz_pa",
    "pressure_xy_pa",
    "pressure_xz_pa",
    "pressure_yz_pa",
)

MAGNETIC_FIELD_CANDIDATES = (
    "vg_b_vol",
    "B_vol",
    "fg_b_vol",
    "fg_b",
    "B",
)
ELECTRIC_FIELD_CANDIDATES = (
    "vg_e_vol",
    "E_vol",
    "fg_e_vol",
    "fg_e",
    "E",
)


def resolve_plasma_context_sources(reader, population):
    """Resolve the same-cell plasma producers for one VLSV source.

    The resolver selects supported historical or current producers for the
    magnetic field, electric field, configured-population bulk velocity,
    number density, and total pressure tensor. Producer names are resolved
    once per open reader and reused for every selected VDF cell in that
    timestep.

    Magnetic field, electric field, and bulk velocity are read and stored as
    complete three-component vectors. The magnetic-field and velocity vectors
    are also reused directly for optional Hermite rotation. Field-grid source
    kinds are recorded here and their selected cell-centre vectors are
    prepared in one bounded pass.

    Magnetic candidates are ``vg_b_vol``, ``B_vol``, ``fg_b_vol``, ``fg_b``,
    and legacy ``B``. Electric candidates are ``vg_e_vol``, ``E_vol``,
    ``fg_e_vol``, ``fg_e``, and ``E``. Direct bulk velocity prefers
    ``<population>/vg_v``, ``<population>/V``, ``vg_v``, ``V``, and ``fg_v``
    before historical ``rho_v/rho``. Number density uses
    ``<population>/vg_rho``, ``<population>/rho``, then ``rho``.

    Pressure selection prefers a complete modern population tensor, a
    complete population-prefixed historical tensor, or a complete unprefixed
    historical tensor. Otherwise it reconstructs the total tensor from a
    complete thermal/nonthermal pair or a complete
    backstream/nonbackstream pair. Representations are never mixed.

    Parameters
    ----------
    reader : analysator.vlsvfile.VlsvReader
        Open VLSV source whose available variables determine the selected
        producer family.
    population : str
        Configured particle population owning the VDF and fluid moments.

    Returns
    -------
    dict
        Selected producer names, source meshes, and pressure-family
        information reused for all cells in the source.

    Notes
    -----
    The resolver does not branch on run IDs, paths, datasets, or timesteps.
    Source names are printed once for the opened file rather than once per
    selected cell.
    """

    population = str(population)
    magnetic_field = _resolve_field_source(
        reader,
        MAGNETIC_FIELD_CANDIDATES,
        legacy_variable="B",
    )
    electric_field = _resolve_field_source(
        reader,
        ELECTRIC_FIELD_CANDIDATES,
    )
    velocity_source = resolve_bulk_velocity_source(reader, population)
    bulk_velocity = _create_velocity_source(velocity_source)
    number_density = next(
        variable
        for variable in (
            f"{population}/vg_rho",
            f"{population}/rho",
            "rho",
        )
        if reader.check_variable(variable)
    )
    pressure_tensor = _resolve_pressure_tensor_source(reader, population)
    sources = {
        "magnetic_field": magnetic_field,
        "electric_field": electric_field,
        "bulk_velocity": bulk_velocity,
        "number_density": number_density,
        "pressure_tensor": pressure_tensor,
    }
    print(_format_selected_sources(sources))
    return sources


def prepare_plasma_context_sources_for_cells(reader, sources, cellids):
    """Prepare only selected field-grid vectors for same-cell reads.

    SpatialGrid producers need no preparation. For each selected field-grid
    B, E, or velocity producer, the installed Analysator operation must
    materialize a complete grid before cell-centred indexing. This function
    handles one producer at a time, copies only the three-component vectors
    at the requested VDF cell centres, and releases the complete array before
    reading the next producer. Extraction calls it once per timestep; region
    prediction calls it once for the selected CID set.

    Parameters
    ----------
    reader : analysator.vlsvfile.VlsvReader
        Open source reader shared with VDF extraction or prediction.
    sources : dict
        File-scoped producer mapping returned by
        :func:`resolve_plasma_context_sources`.
    cellids : iterable of int
        Selected SpatialGrid cell IDs in dataset or prediction order.

    Returns
    -------
    dict
        The same producer mapping with small per-CID field-grid vectors that
        are removed as their samples are read.

    Notes
    -----
    The complete field-grid arrays are transient implementation details of
    Analysator. They are never saved, sent through worker IPC, broadcast over
    velocity space, or retained after the selected vectors are consumed by
    their aligned samples.
    """

    cellids = tuple(dict.fromkeys(int(cid) for cid in cellids))
    field_grid_sources = tuple(
        source
        for source in (
            sources["magnetic_field"],
            sources["electric_field"],
            sources["bulk_velocity"],
        )
        if source["mesh"] == "fsgrid"
    )
    if not field_grid_sources or not cellids:
        return sources
    indexes = {
        cid: _field_grid_index_for_cell(reader, cid)
        for cid in cellids
    }
    for source in field_grid_sources:
        values = _read_volumetric_field_grid(
            reader,
            source["variable"],
            source["centering"],
        )
        source["cell_values"] = {
            cid: np.array(values[indexes[cid]], dtype=np.float64, copy=True)
            for cid in cellids
        }
        del values
    return sources


def read_plasma_values_for_cell(reader, cid, sources):
    """Read complete plasma values at one selected VDF cell.

    SpatialGrid scalar and vector producers are read by cell ID. Field-grid
    vectors use the selected SpatialGrid cell centre in metres and
    Analysator's ``get_fsgrid_indices`` convention to index the volumetric
    field-grid array prepared by
    :func:`prepare_plasma_context_sources_for_cells`. Pressure inputs are
    mapped from Analysator's stored off-diagonal order
    ``(Pyz, Pxz, Pxy)`` into canonical ``(Pxy, Pxz, Pyz)`` order.

    Parameters
    ----------
    reader : analysator.vlsvfile.VlsvReader
        Open source reader shared with raw VDF extraction.
    cid : int
        SpatialGrid cell ID owning the selected VDF.
    sources : dict
        File-scoped mapping returned by
        :func:`resolve_plasma_context_sources`. Selected field-grid vectors
        have been prepared by
        :func:`prepare_plasma_context_sources_for_cells`.

    Returns
    -------
    dict
        Full magnetic-field, electric-field, and bulk-velocity vectors,
        number density, and six canonical pressure components. Vector units
        are tesla, volts per metre, and metres per second; density is in
        particles per cubic metre and pressure is in pascals.

    Notes
    -----
    Complete magnetic-field and velocity vectors are both saved as context
    components and passed directly to optional Hermite rotation. No separate
    magnitude values are added to the model feature row.
    """

    cid = int(cid)
    magnetic_field = _read_magnetic_field(
        reader,
        cid,
        sources["magnetic_field"],
    )
    electric_field = _read_vector_source(
        reader,
        cid,
        sources["electric_field"],
    )
    number_density, _ = read_scalar(
        reader,
        sources["number_density"],
        cid,
    )
    bulk_velocity = _read_bulk_velocity_source(
        reader,
        cid,
        sources["bulk_velocity"],
        number_density,
        sources["number_density"],
    )
    pressure_tensor = _read_pressure_tensor(
        reader,
        cid,
        sources["pressure_tensor"],
    )
    return {
        "magnetic_field": magnetic_field,
        "electric_field": electric_field,
        "bulk_velocity": bulk_velocity,
        "number_density": number_density,
        "pressure_tensor": pressure_tensor,
    }


def create_plasma_context_row(
    magnetic_field,
    electric_field,
    bulk_velocity,
    number_density,
    pressure_tensor,
):
    """Create one sample-aligned plasma-context vector.

    The returned vector stores all Cartesian components of the same-cell
    magnetic field, electric field, and fluid bulk velocity, followed by
    number density and the six independent components of the symmetric
    pressure tensor. Keeping vector components preserves both magnitude and
    direction for the CNN and autoencoder context branches.

    Parameters
    ----------
    magnetic_field : numpy.ndarray
        Three-component magnetic field ``(Bx, By, Bz)`` in tesla.
    electric_field : numpy.ndarray
        Three-component electric field ``(Ex, Ey, Ez)`` in volts per metre.
    bulk_velocity : numpy.ndarray
        Three-component fluid velocity ``(Vx, Vy, Vz)`` in metres per second.
    number_density : float
        Number density in particles per cubic metre.
    pressure_tensor : numpy.ndarray
        Six pressure-tensor components in canonical
        ``(Pxx, Pyy, Pzz, Pxy, Pxz, Pyz)`` order and pascals.

    Returns
    -------
    numpy.ndarray
        Float64 vector with shape ``(16,)`` in
        ``PLASMA_CONTEXT_FEATURE_NAMES`` order.

    Notes
    -----
    Dataset extraction writes the vector at the same sample index as the raw
    VDF, optional Hermite coefficients, and metadata row. The magnetic-field
    and velocity vectors are also reused directly when Hermite rotation is
    enabled. Separate vector magnitudes are not stored.
    """

    return np.concatenate(
        [
            np.asarray(magnetic_field, dtype=np.float64),
            np.asarray(electric_field, dtype=np.float64),
            np.asarray(bulk_velocity, dtype=np.float64),
            np.asarray([number_density], dtype=np.float64),
            np.asarray(pressure_tensor, dtype=np.float64),
        ]
    )


def _resolve_field_source(reader, candidates, legacy_variable=None):
    """Select one vector-field producer and record its owning mesh.

    Parameters
    ----------
    reader : analysator.vlsvfile.VlsvReader
        Open VLSV source queried through ``check_variable``.
    candidates : sequence of str
        Ordered vector-variable candidates from the supported magnetic- or
        electric-field naming family.
    legacy_variable : str, optional
        Historical SpatialGrid variable requiring the established
        cell-centre magnetic reconstruction.

    Returns
    -------
    dict
        Selected variable, owning mesh, read method, and field-grid centering
        when applicable. SpatialGrid values are later read by CID; field-grid
        values are sampled at the selected VDF cell centre.
    """

    variable = next(
        candidate
        for candidate in candidates
        if reader.check_variable(candidate)
    )
    if variable == legacy_variable:
        return {
            "variable": variable,
            "mesh": "SpatialGrid",
            "method": "legacy_b_polar_2d",
        }
    if variable.startswith("fg_"):
        centering = (
            "face"
            if variable == "fg_b"
            else "edge" if variable == "fg_e" else "volume"
        )
        return {
            "variable": variable,
            "mesh": "fsgrid",
            "method": "field_grid_cell_center",
            "centering": centering,
        }
    return {
        "variable": variable,
        "mesh": "SpatialGrid",
        "method": "direct_cell",
    }


def _create_velocity_source(source):
    """Describe a direct, field-grid, or density-divided velocity source.

    Parameters
    ----------
    source : tuple
        Velocity mode, primary variable name, and optional density variable
        returned by the configured-population velocity resolver.

    Returns
    -------
    dict
        Velocity read mode, selected producer names, owning mesh, and
        field-grid centering. Direct values and ``rho_v/rho`` both produce a
        complete fluid velocity vector in metres per second.
    """

    mode, primary_variable, density_variable = source
    return {
        "mode": mode,
        "variable": primary_variable,
        "density_variable": density_variable,
        "mesh": (
            "fsgrid" if primary_variable.startswith("fg_") else "SpatialGrid"
        ),
        "centering": "volume",
    }


def _resolve_pressure_tensor_source(reader, population):
    """Select one complete pressure family without mixing representations.

    Parameters
    ----------
    reader : analysator.vlsvfile.VlsvReader
        Open VLSV source queried for complete diagonal/off-diagonal pairs.
    population : str
        Configured VDF population owning population-scoped pressure moments.

    Returns
    -------
    dict
        Direct or summed representation mode and its matched diagonal and
        off-diagonal variable names. Priority is modern total, prefixed
        historical total, unprefixed historical total, thermal plus
        nonthermal, then prefixed or unprefixed backstream plus
        nonbackstream. Values are read in pascals.
    """

    direct_families = (
        (
            f"{population}/vg_ptensor_diagonal",
            f"{population}/vg_ptensor_offdiagonal",
        ),
        (
            f"{population}/PTensorDiagonal",
            f"{population}/PTensorOffDiagonal",
        ),
        ("PTensorDiagonal", "PTensorOffDiagonal"),
    )
    for diagonal, offdiagonal in direct_families:
        if reader.check_variable(diagonal) and reader.check_variable(
            offdiagonal
        ):
            return {
                "mode": "direct",
                "diagonal_variables": (diagonal,),
                "offdiagonal_variables": (offdiagonal,),
            }

    summed_families = (
        (
            (
                f"{population}/vg_ptensor_thermal_diagonal",
                f"{population}/vg_ptensor_nonthermal_diagonal",
            ),
            (
                f"{population}/vg_ptensor_thermal_offdiagonal",
                f"{population}/vg_ptensor_nonthermal_offdiagonal",
            ),
        ),
        (
            (
                f"{population}/PTensorBackstreamDiagonal",
                f"{population}/PTensorNonBackstreamDiagonal",
            ),
            (
                f"{population}/PTensorBackstreamOffDiagonal",
                f"{population}/PTensorNonBackstreamOffDiagonal",
            ),
        ),
        (
            (
                "PTensorBackstreamDiagonal",
                "PTensorNonBackstreamDiagonal",
            ),
            (
                "PTensorBackstreamOffDiagonal",
                "PTensorNonBackstreamOffDiagonal",
            ),
        ),
    )
    diagonal_variables, offdiagonal_variables = next(
        (diagonal, offdiagonal)
        for diagonal, offdiagonal in summed_families
        if all(reader.check_variable(name) for name in (*diagonal, *offdiagonal))
    )
    return {
        "mode": "sum",
        "diagonal_variables": diagonal_variables,
        "offdiagonal_variables": offdiagonal_variables,
    }


def _read_magnetic_field(reader, cid, source):
    """Read one complete same-cell magnetic-field vector.

    Parameters
    ----------
    reader : analysator.vlsvfile.VlsvReader
        Open VLSV source owning the selected magnetic producer.
    cid : int
        SpatialGrid cell ID owning the VDF sample.
    source : dict
        Resolved direct, field-grid, or legacy magnetic source.

    Returns
    -------
    numpy.ndarray
        Float64 ``(Bx, By, Bz)`` vector in tesla. All components are saved in
        the context row and the same vector may be reused for Hermite rotation.
    """

    if source["mesh"] == "fsgrid":
        return _read_field_grid_vector(cid, source)
    magnetic_field, _ = get_cell_centered_magnetic_field(
        reader,
        cid,
        magnetic_source=source,
    )
    return magnetic_field


def _read_vector_source(reader, cid, source):
    """Read one complete SpatialGrid or prepared field-grid vector.

    Parameters
    ----------
    reader : analysator.vlsvfile.VlsvReader
        Open source used for direct SpatialGrid reads.
    cid : int
        SpatialGrid cell ID owning the VDF sample.
    source : dict
        Resolved source with an owning mesh and selected variable.

    Returns
    -------
    numpy.ndarray
        Three-component same-cell vector in the selected producer's physical
        units. This path reads electric field in volts per metre and may read
        field-grid bulk velocity in metres per second.
    """

    if source["mesh"] == "fsgrid":
        return _read_field_grid_vector(cid, source)
    vector, _ = read_vector(reader, source["variable"], cid)
    return vector


def _read_bulk_velocity_source(
    reader,
    cid,
    source,
    number_density,
    number_density_variable,
):
    """Read direct or density-divided bulk velocity for one VDF cell.

    Parameters
    ----------
    reader : analysator.vlsvfile.VlsvReader
        Open VLSV source owning the configured-population fluid moments.
    cid : int
        SpatialGrid cell ID owning the VDF sample.
    source : dict
        Direct velocity or momentum-density source description.
    number_density : float
        Already-read same-cell number density in particles per cubic metre.
    number_density_variable : str
        Producer of ``number_density``; reused when it is also the denominator
        selected for ``rho_v/rho``.

    Returns
    -------
    numpy.ndarray
        Complete ``(Vx, Vy, Vz)`` fluid velocity in metres per second. All
        components are saved and the same vector may be reused for Hermite
        rotation.
    """

    if source["mesh"] == "fsgrid":
        return _read_field_grid_vector(cid, source)
    if source["mode"] == "momentum_density":
        momentum_density, _ = read_vector(reader, source["variable"], cid)
        density = (
            number_density
            if source["density_variable"] == number_density_variable
            else read_scalar(reader, source["density_variable"], cid)[0]
        )
        return momentum_density / density
    return read_bulk_velocity(
        reader,
        cid,
        (
            source["mode"],
            source["variable"],
            source["density_variable"],
        ),
    )


def _read_pressure_tensor(reader, cid, source):
    """Read and map one total symmetric pressure tensor to canonical order.

    Parameters
    ----------
    reader : analysator.vlsvfile.VlsvReader
        Open VLSV source owning the resolved pressure representation.
    cid : int
        SpatialGrid cell ID owning the VDF sample.
    source : dict
        Complete direct family or matched pair of summed tensor families.

    Returns
    -------
    numpy.ndarray
        Float64 pressure values with shape ``(6,)`` and canonical
        ``(Pxx, Pyy, Pzz, Pxy, Pxz, Pyz)`` order in pascals. Analysator's
        stored off-diagonal ``(Pyz, Pxz, Pxy)`` order is mapped explicitly.
    """

    diagonal = sum(
        (
            read_vector(reader, variable, cid)[0]
            for variable in source["diagonal_variables"]
        ),
        start=np.zeros(3, dtype=np.float64),
    )
    stored_offdiagonal = sum(
        (
            read_vector(reader, variable, cid)[0]
            for variable in source["offdiagonal_variables"]
        ),
        start=np.zeros(3, dtype=np.float64),
    )
    return np.asarray(
        (
            diagonal[0],
            diagonal[1],
            diagonal[2],
            stored_offdiagonal[2],
            stored_offdiagonal[1],
            stored_offdiagonal[0],
        ),
        dtype=np.float64,
    )


def _read_volumetric_field_grid(reader, variable, centering):
    """Read one field-grid vector array in cell-centred volumetric form.

    The installed Analysator API materializes a complete field grid for this
    operation. The caller copies only requested same-cell vectors and then
    releases this source-dtype array before preparing another producer,
    avoiding both repeated reads and an additional full float64 copy.

    Parameters
    ----------
    reader : analysator.vlsvfile.VlsvReader
        Open VLSV source providing field-grid dimensions and values.
    variable : str
        Resolved ``fg_*`` vector producer.
    centering : {"volume", "face", "edge"}
        Existing Analysator centering operation appropriate to the producer.

    Returns
    -------
    numpy.ndarray
        Source-dtype field grid with shape ``(nx, ny, nz, 3)``. The caller
        copies only requested VDF-cell vectors and releases this transient
        complete array immediately.
    """

    values = (
        reader.read_fsgrid_variable(variable, operator="pass")
        if centering == "volume"
        else reader.read_fg_variable_as_volumetric(
            variable,
            centering=centering,
            operator="pass",
        )
    )
    mesh_size = tuple(int(value) for value in reader.get_fsgrid_mesh_size())
    return np.asarray(values).reshape(*mesh_size, 3)


def _field_grid_index_for_cell(reader, cid):
    """Return the field-grid index at one VDF cell centre.

    Parameters
    ----------
    reader : analysator.vlsvfile.VlsvReader
        Open source providing SpatialGrid coordinates and fsgrid indexing.
    cid : int
        SpatialGrid cell ID owning the selected VDF.

    Returns
    -------
    tuple of int
        Three-dimensional fsgrid index calculated from the cell-centre
        coordinate in metres using Analysator's current indexing convention.
    """

    coordinate_m = np.asarray(
        reader.get_cell_coordinates(int(cid)),
        dtype=np.float64,
    )
    return tuple(
        int(value)
        for value in reader.get_fsgrid_indices(coordinate_m)
    )


def _read_field_grid_vector(cid, source):
    """Return one copied field-grid vector prepared for a selected CID.

    Parameters
    ----------
    cid : int
        SpatialGrid cell ID owning the selected VDF.
    source : dict
        Resolved field-grid source containing selected cell-centre vectors.

    Returns
    -------
    numpy.ndarray
        Independent three-component vector for the requested VDF cell. The
        prepared source-cache entry is removed after this read; its values are
        stored in the aligned context row. Magnetic, electric, or velocity
        physical units follow the producer.
    """

    return np.array(source["cell_values"].pop(int(cid)), copy=True)


def _format_selected_sources(sources):
    """Format one concise producer summary for an opened VLSV source.

    Parameters
    ----------
    sources : dict
        File-scoped magnetic, electric, velocity, density, and pressure
        producer mapping.

    Returns
    -------
    str
        One line printed once per reader. Legacy momentum density includes
        both numerator and denominator names, and summed pressure families
        list every participating tensor producer.
    """

    pressure = sources["pressure_tensor"]
    velocity = sources["bulk_velocity"]
    velocity_name = (
        f"{velocity['variable']}/{velocity['density_variable']}"
        if velocity["mode"] == "momentum_density"
        else velocity["variable"]
    )
    pressure_names = (
        *pressure["diagonal_variables"],
        *pressure["offdiagonal_variables"],
    )
    return (
        "Plasma context sources: "
        f"B={sources['magnetic_field']['variable']}, "
        f"E={sources['electric_field']['variable']}, "
        f"V={velocity_name}, "
        f"n={sources['number_density']}, "
        f"P={'+'.join(pressure_names)}"
    )


__all__ = [
    "PLASMA_CONTEXT_FEATURE_NAMES",
    "create_plasma_context_row",
    "prepare_plasma_context_sources_for_cells",
    "read_plasma_values_for_cell",
    "resolve_plasma_context_sources",
]
