"""Expose focused dataset, VDF-slice, and prediction plotting entry points.

The package keeps visualization separate from scientific extraction and
model inference. Public functions consume current saved arrays, metadata, or
already-decoded prediction rows and create figures without changing the
underlying physical or numerical results.
"""

from src.plotting.dataset_colormap import (
    DatasetColormapResult,
    plot_dataset_colormap,
)
from src.plotting.dataset_selection_colormap import (
    create_bulk_velocity_expression,
    render_dataset_velocity_background,
)
from src.plotting.dataset_vdf_frame import (
    DatasetVdfFrameResult,
    plot_dataset_vdf_frames,
    render_dataset_vdf_frame,
)
from src.plotting.frame_animation import create_frame_animation
from src.plotting.hermite_coefficients import (
    DatasetHermiteFrameResult,
    plot_dataset_hermite_frames,
    render_hermite_coefficient_frame,
)
from src.plotting.point_selection_areas import (
    draw_point_selection_areas,
)
from src.plotting.prediction_plot_geometry import (
    build_class_color_map,
    select_prediction_representatives,
)
from src.plotting.prediction_vdf_cuts import (
    PeakVdfCuts,
    extract_peak_vdf_cuts,
    plot_peak_vdf_planes,
)
from src.plotting.prediction_visualization import (
    save_coordinate_prediction_figure,
    save_region_prediction_figure,
)
from src.plotting.vdf_slices import (
    DEFAULT_VDF_PLANES,
    PhysicalVdfPlane,
    PreparedPhysicalVdfPlane,
    VdfSlicePlotResult,
    create_physical_vdf_plane,
    draw_vdf_plane,
    extract_peak_vdf_planes,
    format_vdf_plane_axes,
    get_vdf_plot_velocity_cell_size_ms,
    load_vdf_plotting_thresholds,
    plot_dataset_vdf_slices,
    prepare_physical_vdf_plane,
    select_vdf_samples,
)


__all__ = [
    "DatasetColormapResult",
    "DatasetHermiteFrameResult",
    "DatasetVdfFrameResult",
    "DEFAULT_VDF_PLANES",
    "PeakVdfCuts",
    "PhysicalVdfPlane",
    "PreparedPhysicalVdfPlane",
    "VdfSlicePlotResult",
    "build_class_color_map",
    "create_bulk_velocity_expression",
    "create_frame_animation",
    "create_physical_vdf_plane",
    "draw_point_selection_areas",
    "draw_vdf_plane",
    "extract_peak_vdf_planes",
    "extract_peak_vdf_cuts",
    "format_vdf_plane_axes",
    "get_vdf_plot_velocity_cell_size_ms",
    "load_vdf_plotting_thresholds",
    "plot_dataset_colormap",
    "plot_dataset_hermite_frames",
    "plot_dataset_vdf_frames",
    "plot_dataset_vdf_slices",
    "plot_peak_vdf_planes",
    "prepare_physical_vdf_plane",
    "render_dataset_vdf_frame",
    "render_hermite_coefficient_frame",
    "render_dataset_velocity_background",
    "save_coordinate_prediction_figure",
    "save_region_prediction_figure",
    "select_vdf_samples",
    "select_prediction_representatives",
]
