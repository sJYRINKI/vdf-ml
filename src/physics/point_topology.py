"""Detect and classify magnetic X/O critical points in the x-z plane.

Zero contours of the smoothed flux-function gradients are intersected.
The local two-dimensional Hessian identifies saddles as X points and local
maxima as O points under the established determinant and ``Hxx`` signs.
"""

import numpy as np

from src.data.step_02_find_vdf_cells import (
    R_EARTH,
    find_matching_region_name_re,
)
from src.physics.flux_grid import (
    calculate_hessian,
    find_flux_gradient_intersections,
    interpolate_flux,
    read_smoothed_flux_grid,
)
from src.physics.island_contours import (
    add_o_point_island_contours,
)
from src.physics.point_selection import get_point_selection_method


PROTON_MASS = 1.67262192369e-27
ELEMENTARY_CHARGE = 1.602176634e-19
MU0 = 4.0e-7 * np.pi


def find_point_coords_re(reader, flux_file_location, points_config=None):
    """Find physical X- and O-point coordinates in Earth radii.

    Plotting and lightweight callers use this coordinate-only projection of
    the full topology records. Detection and Hessian classification remain
    identical to dataset planning, but selection geometry is discarded from
    the return value.

    Parameters
    ----------
    reader : analysator.vlsvfile.VlsvReader
        Open VLSV reader for the matching timestep.
    flux_file_location : str or pathlib.Path
        Matching raw magnetic-flux file.
    points_config : dict, optional
        Point regions and X/O selection settings.

    Returns
    -------
    x_point_coords_re : list of list of float
        X-point coordinates ``[x, y, z]`` in Earth radii.
    o_point_coords_re : list of list of float
        O-point coordinates ``[x, y, z]`` in Earth radii.
    """

    x_point_records, o_point_records = find_point_records(
        reader=reader,
        flux_file_location=flux_file_location,
        points_config=points_config,
    )
    return (
        [record["coord_re"] for record in x_point_records],
        [record["coord_re"] for record in o_point_records],
    )


def find_point_records(
    reader,
    flux_file_location,
    points_config=None,
    *,
    include_physical_geometry=False,
):
    """Detect X/O points and attach selection-specific physical geometry.

    Dataset labeling calls this complete x-z topology workflow before mapping
    critical points to VDF cells. It smooths the flux grid, intersects zero
    gradient contours, classifies candidates with the local Hessian, records
    physical coordinates and flux, and calculates only the X/O geometry
    required by the configured selection methods.

    Parameters
    ----------
    reader : analysator.vlsvfile.VlsvReader
        Open VLSV reader for spatial indices and physical density.
    flux_file_location : str or pathlib.Path
        Matching raw double-precision flux-function file.
    points_config : dict, optional
        Point regions and physical/manual selection settings.
    include_physical_geometry : bool, optional
        Whether to retain X ion-inertial-length boxes and O island contours
        for dataset-frame overlays even when the active sample-selection
        method is manual.

    Returns
    -------
    x_point_records : list of dict
        X points with coordinates, flux, Hessian eigensystem, and optional
        local ion inertial length.
    o_point_records : list of dict
        O points with coordinates, flux, and optional island contours.

    Notes
    -----
    Negative Hessian determinant identifies an X point. Positive determinant
    with negative ``Hxx`` identifies an O point. Points within 100 spatial
    cells of an x/z boundary remain excluded.
    """

    x_array, z_array, flux_function_zx = read_smoothed_flux_grid(
        reader=reader,
        flux_file_location=flux_file_location,
    )
    x_cells = len(x_array)
    z_cells = len(z_array)
    dx = float(x_array[1] - x_array[0])
    xmin = float(x_array[0])
    zmin = float(z_array[0])
    intersections = find_flux_gradient_intersections(
        x_array,
        z_array,
        flux_function_zx,
    )
    x_point_records = []
    o_point_records = []
    flux_function_xz = flux_function_zx.T
    x_selection_method = get_point_selection_method(
        config=points_config,
        point_kind="x",
    )
    o_selection_method = get_point_selection_method(
        config=points_config,
        point_kind="o",
    )
    physical_selection_methods = {
        "physical",
        "consensus",
        "union",
    }
    x_physical_geometry = (
        x_selection_method in physical_selection_methods
        or include_physical_geometry
    )
    o_physical_geometry = (
        o_selection_method in physical_selection_methods
        or include_physical_geometry
    )

    for x_coord, z_coord in intersections:
        coords = [x_coord, 0, z_coord]
        cellid = reader.get_cellid(coords)
        i = int((cellid - 1) % x_cells)
        j = (int(cellid) - 1) // x_cells
        if i <= 100 or j <= 100 or i >= x_cells - 100 or j >= z_cells - 100:
            continue

        hessian = calculate_hessian(
            flux_function_xz=flux_function_xz,
            i=i,
            j=j,
            dx=dx,
        )
        det_hessian = np.linalg.det(hessian)
        eigvals, eigvecs = np.linalg.eigh(hessian)
        coord_re = [
            float(coords[0] / R_EARTH),
            float(coords[1] / R_EARTH),
            float(coords[2] / R_EARTH),
        ]
        region_name = find_matching_region_name_re(
            coord_re=coord_re,
            points_config=points_config,
            names_key="point_region_names",
        )
        if region_name is None:
            continue
        interpolated_flux = interpolate_flux(
            flux_function_xz=flux_function_xz,
            x=float(coords[0]),
            z=float(coords[2]),
            xmin=float(xmin),
            zmin=float(zmin),
            dx=float(dx),
        )

        if det_hessian < 0:
            point_record = create_point_record(
                point_kind="x",
                coord_m=coords,
                coord_re=coord_re,
                cellid=cellid,
                flux=interpolated_flux,
                hessian=hessian,
                eigvals=eigvals,
                eigvecs=eigvecs,
            )
            point_record["region_name"] = region_name
            if x_physical_geometry:
                add_ion_inertial_length(
                    reader=reader,
                    point_record=point_record,
                    points_config=points_config,
                )
            x_point_records.append(point_record)

        if det_hessian > 0 and hessian[0, 0] < 0:
            point_record = create_point_record(
                point_kind="o",
                coord_m=coords,
                coord_re=coord_re,
                cellid=cellid,
                flux=interpolated_flux,
                hessian=hessian,
                eigvals=eigvals,
                eigvecs=eigvecs,
            )
            point_record["region_name"] = region_name
            o_point_records.append(point_record)

    if o_physical_geometry:
        add_o_point_island_contours(
            o_point_records=o_point_records,
            x_point_records=x_point_records,
            flux_function_zx=flux_function_zx,
            x_array=x_array,
            z_array=z_array,
            points_config=points_config,
        )
        if o_selection_method == "physical":
            o_point_records = [
                point_record
                for point_record in o_point_records
                if point_record.get("contour_vertices_re") is not None
            ]
    return x_point_records, o_point_records


def is_in_point_region(coord_re, points_config):
    """Return whether an x-z critical point belongs to a configured region.

    Topology detection uses region membership after Hessian classification so
    out-of-scope candidates never enter point labeling or island association.

    Parameters
    ----------
    coord_re : array-like
        Point coordinate ``[x, y, z]`` in Earth radii.
    points_config : dict
        Point-region configuration.

    Returns
    -------
    bool
        Whether one configured point region contains the coordinate.
    """

    return find_matching_region_name_re(
        coord_re=coord_re,
        points_config=points_config,
        names_key="point_region_names",
    ) is not None


def create_point_record(
    point_kind,
    coord_m,
    coord_re,
    cellid,
    flux,
    hessian,
    eigvals,
    eigvecs,
):
    """Create one magnetic critical-point record.

    The topology loop calls this immediately after Hessian classification.
    Keeping continuous coordinates, source-grid cell ID, interpolated flux,
    and the eigensystem together allows later selection geometry to reuse the
    exact candidate calculation.

    Parameters
    ----------
    point_kind : {"x", "o"}
        Hessian classification.
    coord_m : array-like
        Physical ``[x, y, z]`` coordinate in metres.
    coord_re : array-like
        Same coordinate in Earth radii.
    cellid : int
        Spatial cell containing the point.
    flux : float
        Bilinearly interpolated magnetic-flux-function value in the source
        flux file's units.
    hessian : numpy.ndarray
        Symmetric x-z flux Hessian with shape ``(2, 2)``.
    eigvals : numpy.ndarray
        Two Hessian eigenvalues.
    eigvecs : numpy.ndarray
        Two column eigenvectors in x-z coordinates.

    Returns
    -------
    dict
        Point identity, coordinates, flux, and Hessian geometry.
    """

    return {
        "is_point_record": True,
        "point_kind": point_kind,
        "coord_m": [float(value) for value in coord_m],
        "coord_re": [float(value) for value in coord_re],
        "cellid": int(cellid),
        "flux": float(flux),
        "hessian": np.asarray(hessian, dtype=float),
        "eigvals": np.asarray(eigvals, dtype=float),
        "eigvecs": np.asarray(eigvecs, dtype=float),
    }


def add_ion_inertial_length(reader, point_record, points_config):
    """Add selected-cell proton density and local ``d_i`` to an X point.

    Physical X-point selection needs a local length scale for its
    Hessian-aligned box. This function reads the configured number density at
    the candidate's spatial cell and mutates the record with density, metres,
    and Earth-radius forms of the proton inertial length.

    Parameters
    ----------
    reader : analysator.vlsvfile.VlsvReader
        Open source reader.
    point_record : dict
        X-point record updated in place.
    points_config : dict
        Configuration containing the density variable.
    """

    x_selection_config = (points_config or {}).get("x_selection", {})
    density_variable = x_selection_config.get("density_variable", "rho")
    number_density = float(
        np.asarray(
            reader.read_variable(
                density_variable,
                int(point_record["cellid"]),
            )
        ).squeeze()
    )
    di_m = compute_ion_inertial_length(number_density)
    point_record["rho"] = number_density
    point_record["di_m"] = di_m
    point_record["di_re"] = di_m / R_EARTH


def compute_ion_inertial_length(number_density):
    """Calculate proton ion inertial length from number density.

    X-point selection uses ``sqrt(m_p / (mu_0 n e^2))`` as the physical scale
    multiplying configured Hessian-eigenvector half-widths.

    Parameters
    ----------
    number_density : float
        Positive proton number density in ``m^-3``.

    Returns
    -------
    float
        Ion inertial length in metres.

    """

    return float(
        np.sqrt(
            PROTON_MASS / (MU0 * number_density * ELEMENTARY_CHARGE**2)
        )
    )
