"""Read typed same-cell physical values from an open VLSV reader.

Plasma-context extraction, Hermite rotation, prediction, and plotting reuse
these focused adapters after file-scoped source resolution.
Configured-population bulk velocity prefers ``<population>/vg_v``,
``<population>/V``, ``vg_v``, ``V``, and field-grid ``fg_v``; historical
files derive the complete metres-per-second vector from the matched
``rho_v/rho`` number-flux relation. Scalar, vector, and pressure producers
remain on their owning SpatialGrid or field grid as selected by the caller.

Values stay in the reader's physical units and become Python scalars or
float64 vectors. The saved float32 ``(n_samples, 16)`` context is assembled
elsewhere from complete B/E/V components, number density, and six pressure
components. Separate B/E/V magnitudes are not stored.
"""

import numpy as np


def resolve_bulk_velocity_source(reader, population):
    """Select the total bulk-velocity source for one open VLSV file.

    Newer Vlasiator generations store a population-scoped or unnamespaced
    velocity vector directly in metres per second. Older files store number
    flux as ``rho_v`` and number density as ``rho``; their ratio is the total
    bulk velocity. The explicit candidate order excludes thermal,
    nonthermal, backstream, and drift velocities.

    Parameters
    ----------
    reader : analysator.vlsvfile.VlsvReader
        Open source reader reused for physical values and spatial plotting.
    population : str
        Active VDF population used for population-scoped variable names.

    Returns
    -------
    mode : str
        ``"direct"`` for a stored velocity vector or
        ``"momentum_density"`` for the legacy ratio.
    primary_variable : str
        Direct velocity variable or legacy momentum variable.
    density_variable : str or None
        Legacy number-density variable when division is required.
    """

    direct_candidates = (
        f"{population}/vg_v",
        f"{population}/V",
        "vg_v",
        "V",
        "fg_v",
    )
    direct_variable = next(
        (
            variable
            for variable in direct_candidates
            if reader.check_variable(variable)
        ),
        None,
    )
    if direct_variable is not None:
        return "direct", direct_variable, None
    return next(
        (
            "momentum_density",
            momentum_variable,
            density_variable,
        )
        for momentum_variable, density_variable in (("rho_v", "rho"),)
        if reader.check_variable(momentum_variable)
        and reader.check_variable(density_variable)
    )


def read_bulk_velocity(reader, cid, source):
    """Read total bulk velocity at one spatial cell in metres per second.

    Optional Hermite rotation uses this same-cell value only to define the
    perpendicular bulk-flow axis. A direct velocity producer is returned
    unchanged; the legacy producer divides ``rho_v`` by ``rho`` exactly once.

    Parameters
    ----------
    reader : analysator.vlsvfile.VlsvReader
        Open VLSV reader used for the selected physical VDF.
    cid : int
        Spatial cell identifier owning the VDF.
    source : tuple
        Result of :func:`resolve_bulk_velocity_source` for this reader.

    Returns
    -------
    numpy.ndarray
        Total velocity vector ``[v_x, v_y, v_z]`` in metres per second.
    """

    mode, primary_variable, density_variable = source
    primary_values, _ = read_vector(reader, primary_variable, int(cid))
    if mode == "direct":
        return primary_values
    density, _ = read_scalar(reader, density_variable, int(cid))
    return primary_values / density


def read_scalar(reader, variable, cid):
    """Read one scalar variable and its reported units.

    Same-cell sparsity-threshold loading uses this projection of the general
    Analysator variable-info result after producer resolution.

    Parameters
    ----------
    reader : analysator.vlsvfile.VlsvReader
        Open source reader.
    variable : str
        Resolved VLSV variable name.
    cid : int
        Positive spatial cell identifier.

    Returns
    -------
    value : float
        Scalar value at the selected cell.
    units : str
        Units reported by the reader, or an empty string when absent.
    """

    values, units = read_variable_info(reader, variable, cid)
    return float(np.asarray(values).reshape(-1)[0]), units


def read_vector(reader, variable, cid):
    """Read one physical three-vector and its reported units.

    Direct magnetic-field loading uses this single-cell form before Hermite
    rotation-frame construction.

    Parameters
    ----------
    reader : analysator.vlsvfile.VlsvReader
        Open source reader.
    variable : str
        Resolved VLSV variable name.
    cid : int
        Positive spatial cell identifier.

    Returns
    -------
    values : numpy.ndarray
        Float64 vector with shape ``(3,)``.
    units : str
        Units reported by the reader, or an empty string when absent.
    """

    values, units = read_variable_info(reader, variable, cid)
    return np.asarray(values, dtype=np.float64), units


def read_vectors(reader, variable, cellids):
    """Read aligned physical three-vectors for several spatial cells.

    Component-face magnetic reconstruction uses this batched form to read the
    selected cell and its required +x and +z neighbours in one operation.

    Parameters
    ----------
    reader : analysator.vlsvfile.VlsvReader
        Open source reader.
    variable : str
        Resolved VLSV variable name.
    cellids : sequence of int
        Spatial cell identifiers in required output order.

    Returns
    -------
    values : numpy.ndarray
        Float64 array with shape ``(n_cells, 3)``.
    units : str
        Units reported by the reader, or an empty string when absent.
    """

    values, units = read_variable_info(reader, variable, cellids)
    return np.asarray(values, dtype=np.float64), units


def read_variable_info(reader, variable, cellids):
    """Read VLSV data with pass-through operator and normalized units.

    This is the shared Analysator boundary for scalar and vector adapters. It
    leaves numerical data in the reader's native physical convention and
    converts only a missing or present units attribute to a stable string.

    Parameters
    ----------
    reader : analysator.vlsvfile.VlsvReader
        Open source reader.
    variable : str
        Resolved VLSV variable name.
    cellids : int or sequence of int
        Spatial cell identifier or ordered identifiers.

    Returns
    -------
    values : object
        Data object returned by Analysator.
    units : str
        Units reported by Analysator, or an empty string when absent.
    """

    variable_info = reader.read_variable_info(
        variable,
        cellids=cellids,
        operator="pass",
    )
    units = (
        ""
        if variable_info.units is None
        else str(variable_info.units)
    )
    return variable_info.data, units
