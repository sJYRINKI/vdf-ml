"""Expose reusable physical calculations used by VDF workflows.

Dataset extraction and VLSV prediction resolve same-cell magnetic field,
electric field, configured-population bulk velocity, number density, and total
pressure producers in the focused plasma-context module. Aligned model context
stores complete B/E/V vectors in Cartesian order, density, and
``(Pxx, Pyy, Pzz, Pxy, Pxz, Pyz)`` as a float32 ``(n_samples, 16)`` array.
Separate magnitudes are not stored, and the same B/V vectors support optional
Hermite rotation.

Dataset planning separately calls x-z flux topology and point selection.
Keeping these calculations outside orchestration gives source resolution,
endpoint coordinates, units, axis order, and numerical ordering one current
implementation.
"""

from src.physics.hermite_basis import DEFAULT_HERMITE_ORDER
from src.physics.hermite_rotation import rotate_vdf
from src.physics.hermite_transform import vdf_to_hermite
from src.physics.magnetic_field import (
    get_cell_centered_magnetic_field,
)

__all__ = [
    "DEFAULT_HERMITE_ORDER",
    "get_cell_centered_magnetic_field",
    "rotate_vdf",
    "vdf_to_hermite",
]
