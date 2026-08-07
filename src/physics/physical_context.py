"""Read same-cell physical values for rotation and raw VDF plotting.

Optional Hermite rotation reads the magnetic field in teslas and total bulk
velocity in metres per second at the selected VDF cell. Raw VDF figures keep
their separate source sparsity-threshold read in ``s^3 m^-6``; that threshold
is never an input to the physical-VDF Hermite transform.
"""

from src.physics.magnetic_field import (
    get_cell_centered_magnetic_field,
)
from src.physics.vlsv_physical_values import (
    read_bulk_velocity,
    read_scalar,
)


SPARSITY_THRESHOLD_UNITS = "s3/m6"


def resolve_sparsity_threshold_variable(reader, population):
    """Select the VDF sparsity-threshold producer for one source file.

    Vlasiator output generations store the numerical floor used by sparse
    velocity-space data under different names. Older BCH files expose the
    unnamespaced ``MinValue`` variable, while population-based files may
    expose ``<population>/MinValue``,
    ``<population>/EffectiveSparsityThreshold``, or
    ``<population>/vg_effectivesparsitythreshold``. The candidate order
    follows the supported Analysator-era output families.

    Parameters
    ----------
    reader : analysator.vlsvfile.VlsvReader
        Open source reader reused by extraction and prediction plotting.
    population : str
        VDF population used to construct population-scoped variable names.

    Returns
    -------
    str
        First available threshold-variable name. Prediction source loading
        retains this name for every displayed spatial cell in the file.

    Notes
    -----
    The selected value remains in physical VDF units of ``s^3 m^-6``. This
    function does not estimate a threshold from distribution values and does
    not modify the source VDF.
    """

    candidates = (
        "MinValue",
        f"{population}/MinValue",
        f"{population}/EffectiveSparsityThreshold",
        f"{population}/vg_effectivesparsitythreshold",
    )
    return next(
        variable
        for variable in candidates
        if reader.check_variable(variable)
    )


def get_sparsity_threshold(
    reader,
    cid,
    *,
    variable,
):
    """Read the exact same-cell plotting threshold for a raw VDF panel.

    Prediction plotting passes the exact producer resolved when its VLSV
    source was opened. The returned value controls only the shared physical
    VDF mask and remains independent from Hermite coefficient generation.

    Parameters
    ----------
    reader : analysator.vlsvfile.VlsvReader
        Open VLSV reader used for sample extraction.
    cid : int
        Positive selected spatial cell identifier.
    variable : str
        Exact file-scoped producer selected before reading any source cells.

    Returns
    -------
    min_value : float
        Threshold value in ``s^3 m^-6``.
    provenance : dict
        Selected variable, units, operator, population, and centring.

    """

    cid = int(cid)
    variable = str(variable)
    population = (
        variable.split("/", 1)[0]
        if "/" in variable
        else "unnamespaced"
    )
    min_value, units = read_scalar(reader, variable, cid)
    return min_value, {
        "variable": variable,
        "operator": "pass",
        "units": units,
        "stored_centering": "selected_spatial_cell_native",
        "output_centering": "selected_spatial_cell",
        "population": population,
        "fallback": "none",
    }


def get_hermite_rotation_context(
    reader,
    cid,
    resolved_source,
    velocity_source,
):
    """Read the two same-cell vectors used for optional Hermite rotation.

    Extraction and prediction call this only when ``hermite.rotate`` is true.
    The magnetic field defines the parallel direction and the total bulk flow
    defines the first perpendicular direction after removing its parallel
    component. Neither vector replaces the drift and thermal moments computed
    from the rotated physical VDF itself.

    Parameters
    ----------
    reader : analysator.vlsvfile.VlsvReader
        Open VLSV reader also used for raw VDF extraction.
    cid : int
        Positive selected spatial cell identifier.
    resolved_source : ResolvedVlsvSource
        File-scoped magnetic-field producer reused across samples.
    velocity_source : tuple
        Direct-velocity or legacy ``rho_v / rho`` producer selected once for
        this source reader.

    Returns
    -------
    magnetic_field : numpy.ndarray
        Magnetic vector ``[B_x, B_y, B_z]`` in teslas.
    bulk_velocity : numpy.ndarray
        Total flow vector ``[v_x, v_y, v_z]`` in metres per second.
    """

    cid = int(cid)
    magnetic_field, _ = (
        get_cell_centered_magnetic_field(
            reader,
            cid,
            resolved_source=resolved_source,
        )
    )
    bulk_velocity = read_bulk_velocity(
        reader,
        cid,
        velocity_source,
    )
    return magnetic_field, bulk_velocity
