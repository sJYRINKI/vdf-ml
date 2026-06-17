import analysator as pt
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
import sys
from scipy.ndimage import convolve
from scipy.signal import convolve2d
from shapely import geometry
from numpy import linalg as LA
import logging

from src.vdf_helpers import R_EARTH, get_cellid_with_vdf
from src.timesteps import create_timestep_path

def create_labeled_coords_for_timestep(config, timestep):
    """
    Create labeled coordinate data for one timestep.

    Parameters
    ----------
    config : dict
        Dataset config.
    timestep : int
        Timestep to process.

    Returns
    -------
    tuple
        Pair of ``(timestep, label_data)``. ``label_data`` contains
        ``labeled_coords`` and point-conflict ``rejected_cellids``.
    """

    timestep_labeled_coords = list(iter_labeled_coords(config))
    point_labeled_coords, rejected_cellids = create_point_labeled_coords(
        config=config,
        timestep=timestep,
    )
    timestep_labeled_coords.extend(point_labeled_coords)

    return int(timestep), {
        "labeled_coords": timestep_labeled_coords,
        "rejected_cellids": rejected_cellids,
    }

def create_label_data_by_timestep(config, timesteps):
    """
    Create labeled coordinate data for each timestep.

    Parameters
    ----------
    config : dict
        Dataset config.
    timesteps : iterable of int
        Timesteps to process.

    Returns
    -------
    dict
        Mapping from timestep to labeled coordinate data.
    """

    return dict(
        create_labeled_coords_for_timestep(
            config=config,
            timestep=timestep,
        )
        for timestep in timesteps
    )

def iter_labeled_coords(config):
    """
    Iterate over labeled coordinates for each class from config.

    Parameters
    ----------
    config : dict
        Dataset config dictionary.

    Yields
    ------
    class_name : str
        Name of the class.
    label : int
        Integer label for the class.
    coord_re : list of float
        Coordinate in Earth radii, given as ``[x, y, z]``.
    """

    labels = config["labels"]
    class_coords = config.get("class_coords_re", {})

    for class_name, coords in class_coords.items():
        label = labels[class_name]

        for coord_re in coords:
            yield class_name, int(label), coord_re

def iter_point_labeled_coords(config, timestep):
    """
    Iterate over labeled X-point and O-point coordinates for one timestep.

    Parameters
    ----------
    config : dict
        Dataset config dictionary.
    timestep : int
        Timestep to process.

    Yield
    -----
    class_name : str
        Name of the class.
    label : int
        Integer label for the class.
    coord_re : list of float
        Coordinate in Earth radii, given as ``[x, y, z]``.
    """

    point_labeled_coords, _ = create_point_labeled_coords(
        config=config,
        timestep=timestep,
    )

    yield from point_labeled_coords


def create_point_labeled_coords(config, timestep, reader=None):
    """
    Create point labeled coordinates and rejected point cell IDs.

    Parameters
    ----------
    config : dict
        Dataset config dictionary.
    timestep : int
        Timestep to process.
    reader : analysator.vlsvfile.VlsvReader, optional
        Existing reader for the timestep bulk file.

    Returns
    -------
    point_labeled_coords : list of tuple
        Tuples of ``(class_name, label, coord_re)`` for accepted X/O points.
    rejected_cellids : set of int
        VDF cell IDs rejected because X and O points mapped to the same cell.
    """

    points_config = config.get("points")
    labels = config["labels"]

    x_class_name = points_config["x_class_name"]
    o_class_name = points_config["o_class_name"]

    flux_file_location = create_timestep_path(
        path_template=config["file_template_flux"],
        timestep=timestep,
    )

    if reader is None:
        bulk_file_location = create_timestep_path(
            path_template=config["file_template_bulk"],
            timestep=timestep,
        )
        reader = pt.vlsvfile.VlsvReader(str(bulk_file_location))

    x_point_coords_re, o_point_coords_re = find_point_coords_re(
        reader=reader,
        flux_file_location=flux_file_location,
        points_config=points_config,
    )

    x_point_coords_re, o_point_coords_re, rejected_cellids = remove_shared_cellid_points(
        reader=reader,
        x_point_coords_re=x_point_coords_re,
        o_point_coords_re=o_point_coords_re,
    )

    point_labeled_coords = []

    for coord_re in x_point_coords_re:
        point_labeled_coords.append((x_class_name, int(labels[x_class_name]), coord_re))

    for coord_re in o_point_coords_re:
        point_labeled_coords.append((o_class_name, int(labels[o_class_name]), coord_re))

    return point_labeled_coords, rejected_cellids

def remove_shared_cellid_points(reader, x_point_coords_re, o_point_coords_re):
    """
    Remove duplicate or contradictory point labels by VDF cell ID.

    Coordinates are grouped with ``get_cellid_with_vdf``, which is the same
    cell-ID lookup used when samples are saved to metadata. If an X-point and
    an O-point map to the same VDF cell ID, both are discarded. If multiple
    points from the same class map to one VDF cell ID, only the first point is
    kept.

    Parameters
    ----------
    reader : analysator.vlsvfile.VlsvReader
        Open VLSV file reader for the matching timestep.
    x_point_coords_re : list of list of float
        X-point coordinates in Earth radii, each given as ``[x, y, z]``.
    o_point_coords_re : list of list of float
        O-point coordinates in Earth radii, each given as ``[x, y, z]``.

    Returns
    -------
    filtered_x_point_coords_re : list of list of float
        One X-point coordinate per non-conflicting VDF cell ID.
    filtered_o_point_coords_re : list of list of float
        One O-point coordinate per non-conflicting VDF cell ID.
    shared_cellids : set of int
        VDF cell IDs rejected because both point classes mapped to them.
    """

    x_points_by_cellid = group_coords_by_cellid(reader, x_point_coords_re)
    o_points_by_cellid = group_coords_by_cellid(reader, o_point_coords_re)

    shared_cellids = set(x_points_by_cellid) & set(o_points_by_cellid)

    filtered_x_point_coords_re = [
        coords_re[0]
        for cellid, coords_re in x_points_by_cellid.items()
        if cellid not in shared_cellids
    ]

    filtered_o_point_coords_re = [
        coords_re[0]
        for cellid, coords_re in o_points_by_cellid.items()
        if cellid not in shared_cellids
    ]

    return filtered_x_point_coords_re, filtered_o_point_coords_re, shared_cellids

def group_coords_by_cellid(reader, coords_re):
    """
    Group coordinates by the VDF cell ID used for saved samples.

    Parameters
    ----------
    reader : analysator.vlsvfile.VlsvReader
        Open VLSV file reader used to convert coordinates to VDF cell IDs.
    coords_re : list of list of float
        Coordinates in Earth radii, each given as ``[x, y, z]``.

    Returns
    -------
    dict[int, list of list of float]
        Mapping from VDF cell ID to all coordinates that resolve to that cell.
    """

    coords_by_cellid = {}

    for coord_re in coords_re:
        cellid = get_cellid_with_vdf(reader, coord_re)
        coords_by_cellid.setdefault(cellid, []).append(coord_re)

    return coords_by_cellid

def find_point_coords_re(reader, flux_file_location, points_config=None):
    """
    Find X- and O-point coordinates from a flux file.

    parameters
    ----------
    reader : analysatir.vlsvfile.VlsvReader
        Open VLSV file reader for matching timestep.
    flux_file_location : pathlib.Path
        Flux filepath
    points_config : dict, optional
        Point configuration.

    returns
    -------
    x_point_coords_re : list of list of float
        X-point coordinates in Earth radii
    o_point_coords_re : list of list of float
        O-point coordinates in Earth radii
    """

    x_cells=int(reader.get_spatial_mesh_size()[0])
    z_cells=int(reader.get_spatial_mesh_size()[2])
    xsize = reader.read_parameter("xcells_ini")
    xmax =  reader.read_parameter("xmax")
    xmin =  reader.read_parameter("xmin")
    zmin =  reader.read_parameter("zmin")
    zmax =  reader.read_parameter("zmax")
    dx = (xmax-xmin)/xsize 

    ## DEFINE ARRAYS FOR AXIS
    x_array=np.array(range(int(xmin), int(xmax), int(dx)))
    z_array=np.array(range(int(zmin), int(zmax), int(dx)))

    flux_function = np.fromfile(
        flux_file_location,
        dtype='double'
    ).reshape(z_cells,x_cells)

    # Smooth fluxfunction
    kernel_size=5
    fkernel = np.ones((kernel_size,kernel_size))/(kernel_size**2)
    raw_flux_function = flux_function
    flux_function= convolve2d(flux_function, fkernel, 'same')

    dfdx,dfdz=np.gradient(flux_function)

    #calculate the 0 contours of df/dx and df/dz
    plt.figure(1)
    contour1 = plt.contour(x_array, z_array, dfdx, [0])
    contour1_paths = get_contour_paths(contour1)

    contour2 = plt.contour(x_array, z_array, dfdz, [0])
    contour2_paths = get_contour_paths(contour2)

    x_coords=[]
    z_coords=[]

    # find the intersection points of the 
    for path1 in contour1_paths:
        for path2 in contour2_paths:
            if path1.intersects_path(path2) and len(path1)>1 and len(path2)>1:
                intersection = find_intersection(path1.vertices, path2.vertices)

                for x_coord, z_coord in intersection_to_points(intersection):
                    x_coords.append(x_coord)
                    z_coords.append(z_coord)


    # DEFINE the type of the gradient(flux)=0 ##
    x_point_coords_re=[]
    o_point_coords_re=[]
    x_point_fluxes=[]
    o_point_fluxes=[]
    flux_function=flux_function.T

    for k in range(len(x_coords)):
        #cellid = 1+i+j*x_cells
        coords=[x_coords[k],0,z_coords[k]]
        cellid=reader.get_cellid(coords)
        i=int((cellid-1)%x_cells)
        j=(int(cellid)-1)//x_cells

        ## the limist for i and j have the value 100, to save time, I have not been interested about x's and o's within 100 cell from boundaries.
        ## If you are then change the limits
        if i > 100 and j > 100 and i < x_cells-100 and j < z_cells-100:


            # Hessian matrix using central difference formulas for the second partial derivatives
            deltaPsi_xx= (
                flux_function[i+1,j]
                -2*flux_function[i,j]
                +flux_function[i-1,j]
            )/dx**2
            
            deltaPsi_zz= (
                flux_function[i,j+1]
                -2*flux_function[i,j]
                +flux_function[i,j-1]
            )/dx**2

            deltaPsi_xz= (
                flux_function[i+1,j+1]
                -flux_function[i+1,j-1]
                -flux_function[i-1,j+1]
                +flux_function[i-1,j-1]
            )/(4*dx**2)

            DetHess = deltaPsi_xx*deltaPsi_zz-deltaPsi_xz*deltaPsi_xz

            coord_re = [
                float(coords[0] / R_EARTH),
                float(coords[1] / R_EARTH),
                float(coords[2] / R_EARTH),
            ]

            if not is_in_point_region(coord_re, points_config):
                continue

            i_i = int(coords[0]/dx)
            i_f = coords[0]/dx - i_i
            j_i = int(coords[1]/dx)
            j_f = coords[1]
            interpolated_flux = (1.-j_f) * ((1. - i_f) * flux_function[i,j] + i_f * flux_function[i+1,j]) + j_f* ((1. - i_f) * flux_function[i,j+1] + i_f * flux_function[i+1,j+1])

            if DetHess < 0:
                x_point_coords_re.append(coord_re)
                x_point_fluxes.append(interpolated_flux)          

            ## NOTE if you want the o-points to be local maxima use deltaPsi_xx < 0, if you want them to be local minima use  deltaPsi_xx > 0                  
            #if DetHess > 0 and deltaPsi_xx > 0:
            #   minima_location.append(coords)
            if DetHess > 0 and deltaPsi_xx < 0:
                o_point_coords_re.append(coord_re)
                o_point_fluxes.append(interpolated_flux) 
                

    return x_point_coords_re, o_point_coords_re

def get_contour_paths(contour):
    """
    Get contour paths from a Matplotlib contour object.

    Parameters
    ----------
    contour : matplotlib.contour.QuadContourSet
        Contour object returned by ``plt.contour``.

    Returns
    -------
    list
        Contour path objects.
    """

    if hasattr(contour, "collections"):
        return contour.collections[0].get_paths()

    paths = []

    for path in contour.get_paths():
        for vertices in path.to_polygons(closed_only=False):
            if len(vertices) > 1:
                paths.append(MplPath(vertices))

    return paths

def find_intersection(v1,v2):
    """
    Find intersection points of two contour paths.

    Parameters
    ----------
    v1 : numpy.ndarray
        Vertices of the first contour path.
    v2 : numpy.ndarray
        Vertices of the second contour path.

    Returns
    -------
    intersection : shapely.geometry.LineString
        The intersection of the two contour paths.
    """
    poly1 = geometry.LineString(v1)
    poly2 = geometry.LineString(v2)

    intersection = poly1.intersection(poly2)
    return intersection

def intersection_to_points(intersection):
    """
    Convert a Shapely intersection geometry to coordinate points.

    Parameters
    ----------
    intersection : shapely geometry
        Intersection geometry.

    Returns
    -------
    list of tuple
        List of ``(x, z)`` coordinate pairs.
    """

    if intersection.is_empty:
        return []

    if intersection.geom_type == "Point":
        return [(float(intersection.x), float(intersection.y))]

    if intersection.geom_type == "MultiPoint":
        return [
            (float(point.x), float(point.y))
            for point in intersection.geoms
        ]

    if intersection.geom_type in {"LineString", "LinearRing"}:
        return [
            (float(point[0]), float(point[1]))
            for point in intersection.coords
        ]

    if hasattr(intersection, "geoms"):
        points = []

        for geom in intersection.geoms:
            points.extend(intersection_to_points(geom))

        return points

    return []

def is_in_point_region(coord_re, points_config):
    """
    Check whether a point coordinate is in the configured region.

    Parameters
    ----------
    coord_re : list of float
        Coordinate in Earth radii, given as ``[x, y, z]``.
    points_config : dict
        Point configuration.

    Returns
    -------
    bool
        Whether the coordinate should be used as a labeled point.
    """

    region_re = (points_config or {}).get("region_re", {})

    x_between = region_re.get("x_between")
    x_min = region_re.get("x_min")
    x_max = region_re.get("x_max")
    z_abs_max = region_re.get("z_abs_max")

    x_re = coord_re[0]
    z_re = coord_re[2]

    if x_between is not None:
        left_x, right_x = x_between
        x_min = min(left_x, right_x)
        x_max = max(left_x, right_x)

    if x_min is not None and x_re < x_min:
        return False

    if x_max is not None and x_re > x_max:
        return False

    if z_abs_max is not None and abs(z_re) > z_abs_max:
        return False

    return True
