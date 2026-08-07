"""Name current dataset metadata fields and representation constants.

These constants define the row order written to ``metadata.csv`` and the
six auxiliary topology targets consumed by the CNN. They describe core
saved data ownership.
"""


METADATA_SCHEMA_VERSION = "1.0"
TOPOLOGY_SCHEMA_VERSION = "1.0"
HERMITE = "hermite"

SAMPLE_IDENTITY_COLUMNS = (
    "sample_index",
    "timestep",
    "cid",
    "file_location",
)
CLASSIFICATION_COLUMNS = (
    "class_id",
    "class_name",
)
VDF_COORDINATE_COLUMNS = (
    "vdf_x_re",
    "vdf_y_re",
    "vdf_z_re",
)
TOPOLOGY_AVAILABILITY_COLUMNS = (
    "x_topology_available",
    "o_topology_available",
)
TOPOLOGY_TARGET_COLUMNS = (
    "distance_to_x_point_re",
    "distance_to_o_point_re",
    "vdf_to_x_point_dx_re",
    "vdf_to_x_point_dz_re",
    "vdf_to_o_point_dx_re",
    "vdf_to_o_point_dz_re",
)
X_TOPOLOGY_TARGET_COLUMNS = (
    "distance_to_x_point_re",
    "vdf_to_x_point_dx_re",
    "vdf_to_x_point_dz_re",
)
O_TOPOLOGY_TARGET_COLUMNS = (
    "distance_to_o_point_re",
    "vdf_to_o_point_dx_re",
    "vdf_to_o_point_dz_re",
)
TOPOLOGY_TARGET_GROUPS = {
    "x": X_TOPOLOGY_TARGET_COLUMNS,
    "o": O_TOPOLOGY_TARGET_COLUMNS,
}
METADATA_COLUMNS = (
    *SAMPLE_IDENTITY_COLUMNS,
    *CLASSIFICATION_COLUMNS,
    *VDF_COORDINATE_COLUMNS,
    *TOPOLOGY_AVAILABILITY_COLUMNS,
    *TOPOLOGY_TARGET_COLUMNS,
)
