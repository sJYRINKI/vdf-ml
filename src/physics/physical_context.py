"""Read the same-cell sparsity threshold used by raw VDF plotting.

Raw VDF figures retain their source sparsity threshold in ``s^3 m^-6``.
Same-cell magnetic, electric, velocity, density, and pressure values are
resolved and read through :mod:`src.physics.plasma_context`, which is the
single current owner of same-cell plasma producers.
"""

from src.physics.vlsv_physical_values import read_scalar


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
