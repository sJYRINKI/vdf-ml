"""Expose reusable physical calculations used by VDF workflows.

Dataset planning calls the x-z flux topology and point-selection modules,
while Hermite extraction and prediction share the physical-VDF Hermite
transform and optional bulk-flow magnetic rotation. These calculations remain
outside workflow orchestration so endpoint coordinates, physical units, axis
order, and numerical ordering have one implementation.
"""

from src.physics.hermite_basis import DEFAULT_HERMITE_ORDER
from src.physics.hermite_rotation import rotate_vdf
from src.physics.hermite_transform import vdf_to_hermite
from src.physics.magnetic_field import (
    get_cell_centered_magnetic_field,
)
from src.physics.physical_context import get_hermite_rotation_context

__all__ = [
    "DEFAULT_HERMITE_ORDER",
    "get_cell_centered_magnetic_field",
    "get_hermite_rotation_context",
    "rotate_vdf",
    "vdf_to_hermite",
]
