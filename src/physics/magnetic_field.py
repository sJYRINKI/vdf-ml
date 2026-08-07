"""Produce one selected cell-centred magnetic field for Hermite rotation.

Direct volume-average fields are preferred. The established
``legacy_b_polar_2d`` reconstruction averages component-specific face values
only on an unrefined polar ``(nx, 1, nz)`` SpatialGrid.
"""

import numpy as np

from src.physics.vlsv_physical_values import (
    read_vector,
    read_vectors,
)


MAGNETIC_FIELD_CANDIDATES = ("vg_b_vol", "B_vol")
LEGACY_B_POLAR_2D = "legacy_b_polar_2d"
MAGNETIC_FIELD_MAGNITUDE_RTOL = 1e-6
MAGNETIC_FIELD_DIRECTION_ATOL = 1e-6
MAGNETIC_FIELD_UNITS = "T"


def get_cell_centered_magnetic_field(reader, cid, resolved_source=None):
    """Read the selected VDF cell's magnetic field in teslas.

    Hermite physical-context loading calls this for the same spatial cell as
    the extracted VDF. Direct volume-average variables take precedence; when
    both names are present, ``vg_b_vol`` remains the selected value and their
    comparison is descriptive only. Otherwise the established polar-plane
    component-face reconstruction is used.

    Parameters
    ----------
    reader : analysator.vlsvfile.VlsvReader
        Open VLSV reader used for raw VDF extraction.
    cid : int
        Positive selected spatial cell identifier.
    resolved_source : ResolvedVlsvSource, optional
        File-scoped producer resolution reused for all samples. Direct calls
        without it retain focused reader-based selection.

    Returns
    -------
    magnetic_field : numpy.ndarray
        Float64 vector in teslas with shape ``(3,)``.
    provenance : dict
        Selected variable, centring, units, and reconstruction description.

    """

    cid = int(cid)
    magnetic_producer = _resolved_magnetic_producer(resolved_source)
    if magnetic_producer is None:
        direct_variables = tuple(
            variable
            for variable in MAGNETIC_FIELD_CANDIDATES
            if reader.check_variable(variable)
        )
    else:
        producer_variables = tuple(
            producer.variable
            for producer in magnetic_producer.variables
        )
        direct_variables = tuple(
            variable
            for variable in MAGNETIC_FIELD_CANDIDATES
            if variable in producer_variables
        )

    if len(direct_variables) == 1:
        variable = direct_variables[0]
        magnetic_field, units = read_vector(reader, variable, cid)
        return magnetic_field, _magnetic_field_provenance(
            variable=variable,
            units=units,
            stored_centering="cell_centered_volume_average",
            reconstruction="none",
            reconstruction_version=None,
        )

    if len(direct_variables) == 2:
        fields = {}
        units = {}
        for variable in direct_variables:
            fields[variable], units[variable] = read_vector(
                reader,
                variable,
                cid,
            )
        comparison = _compare_direct_magnetic_fields(
            fields["vg_b_vol"],
            fields["B_vol"],
        )
        variable = "vg_b_vol"
        provenance = _magnetic_field_provenance(
            variable=variable,
            units=units[variable],
            stored_centering="cell_centered_volume_average",
            reconstruction="none",
            reconstruction_version=None,
        )
        provenance["candidate_comparison"] = {
            **comparison,
            "variables": ["vg_b_vol", "B_vol"],
            "units": {
                name: units[name]
                for name in ("vg_b_vol", "B_vol")
            },
        }
        return fields[variable], provenance

    magnetic_field, units = _reconstruct_legacy_polar_magnetic_field(
        reader,
        cid,
    )
    return magnetic_field, _magnetic_field_provenance(
        variable="B",
        units=units,
        stored_centering="face_centered_components",
        reconstruction="polar_2d_component_face_average",
        reconstruction_version=LEGACY_B_POLAR_2D,
    )


def _resolved_magnetic_producer(resolved_source):
    """Return the file-scoped magnetic producer selected by data stage 1.

    Direct plotting or physics calls may omit the source record, in which case
    the caller performs the same reader-based candidate selection.
    """

    if resolved_source is None:
        return None
    return resolved_source.magnetic_field


def _reconstruct_legacy_polar_magnetic_field(reader, cid):
    """Average component-specific face fields to one polar cell centre.

    For the unrefined ``(nx, 1, nz)`` SpatialGrid convention, ``Bx`` uses the
    selected and +x cells, ``By`` uses the selected cell, and ``Bz`` uses the
    selected and +z cells. The returned vector retains the reader's field
    units, normally teslas.
    """

    mesh_size = np.asarray(reader.get_spatial_mesh_size(), dtype=np.int64)
    nx, ny, nz = (int(value) for value in mesh_size)
    zero_based_cid = cid - 1
    x_index = zero_based_cid % nx
    z_index = zero_based_cid // (nx * ny)
    neighbor_cellids = [cid, cid + 1, cid + nx]
    face_fields, units = read_vectors(
        reader,
        "B",
        neighbor_cellids,
    )
    magnetic_field = np.asarray(
        [
            0.5 * (face_fields[0, 0] + face_fields[1, 0]),
            face_fields[0, 1],
            0.5 * (face_fields[0, 2] + face_fields[2, 2]),
        ],
        dtype=np.float64,
    )
    return magnetic_field, units


def _compare_direct_magnetic_fields(first, second):
    """Describe magnitude and direction differences between direct fields.

    When both volume-average variable names exist, provenance records these
    diagnostics but the successful workflow still selects ``vg_b_vol``.
    Neither input vector is modified and the comparison does not choose or
    average a field.
    """

    first_magnitude = float(np.linalg.norm(first))
    second_magnitude = float(np.linalg.norm(second))
    absolute_magnitude_difference = abs(first_magnitude - second_magnitude)
    relative_magnitude_difference = (
        absolute_magnitude_difference
        / max(first_magnitude, second_magnitude)
    )
    first_direction = first / first_magnitude
    second_direction = second / second_magnitude
    normalized_direction_difference = float(
        np.linalg.norm(first_direction - second_direction)
    )
    direction_dot_product = float(
        np.clip(np.dot(first_direction, second_direction), -1.0, 1.0)
    )
    direction_angle_rad = float(np.arccos(direction_dot_product))
    materially_consistent = (
        relative_magnitude_difference <= MAGNETIC_FIELD_MAGNITUDE_RTOL
        and normalized_direction_difference
        <= MAGNETIC_FIELD_DIRECTION_ATOL
    )
    return {
        "absolute_magnitude_difference": float(
            absolute_magnitude_difference
        ),
        "relative_magnitude_difference": float(
            relative_magnitude_difference
        ),
        "normalized_direction_difference": (
            normalized_direction_difference
        ),
        "direction_angle_rad": direction_angle_rad,
        "magnitude_rtol": MAGNETIC_FIELD_MAGNITUDE_RTOL,
        "direction_atol": MAGNETIC_FIELD_DIRECTION_ATOL,
        "materially_consistent": bool(materially_consistent),
    }


def _magnetic_field_provenance(
    variable,
    units,
    stored_centering,
    reconstruction,
    reconstruction_version,
):
    """Create the descriptive record for the field used by Hermite rotation.

    The record preserves the exact variable, units, stored centring, and any
    reconstruction convention beside same-cell physical context. It contains
    no field values and owns no reader resource.
    """

    return {
        "variable": variable,
        "operator": "pass",
        "units": units,
        "stored_centering": stored_centering,
        "output_centering": "selected_spatial_cell_center",
        "reconstruction": reconstruction,
        "reconstruction_version": reconstruction_version,
    }
