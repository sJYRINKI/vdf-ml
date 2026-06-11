import analysator as pt
import numpy as np
import matplotlib.pyplot as plt
import sys
from scipy.ndimage import convolve
from scipy.signal import convolve2d
from shapely import geometry
from numpy import linalg as LA
import logging

from src.vdf_helpers import R_EARTH
from src.timesteps import create_timestep_path

def create_labeled_coords_by_timestep(config, timesteps):
    """
    Create labeled coordinates for each timestep.

    Parameters
    ----------
    config : dict
        Dataset config.
    timesteps : iterable of int
        Timesteps to process.

    Returns
    -------
    dict
        Mapping from timestep to labeled coordinates.
    """

    manual_labeled_coords = list(iter_labeled_coords(config))
    labeled_coords_by_timestep = {}

    for timestep in timesteps:
        timestep_labeled_coords = list(manual_labeled_coords)
        timestep_labeled_coords.extend(
            iter_point_labeled_coords(
                config=config,
                timestep=timestep
            )
        )

        labeled_coords_by_timestep[int(timestep)] = timestep_labeled_coords

    return labeled_coords_by_timestep
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

    flux_points_config = config.get("flux_points")

    labels = config["labels"]

    x_class_name = flux_points_config["x_class_name"]
    o_class_name = flux_points_config["o_class_name"]

    bulk_file_location = create_timestep_path(
        path_template=config["file_template_bulk"],
        timestep=timestep,
    )

    flux_file_location = create_timestep_path(
        path_template=config["file_template_flux"],
        timestep=timestep,
    )

    reader = pt.vlsvfile.VlsvReader(str(bulk_file_location))

    x_point_coords_re, o_point_coords_re = find_point_coords_re(
        reader=reader,
        flux_file_location=flux_file_location,
        flux_points_config=flux_points_config,
    )

    for coord_re in x_point_coords_re:
        yield x_class_name, int(labels[x_class_name]), coord_re

    for coord_re in o_point_coords_re:
        yield o_class_name, int(labels[o_class_name]), coord_re

def find_point_coords_re(reader, flux_file_location, flux_points_config=None):
    """
    Find X- and O-point coordinates from a flux file.

    parameters
    ----------
    reader : analysatir.vlsvfile.VlsvReader
        Open VLSV file reader for matching timestep.
    flux_file_location : pathlib.Path
        Flux filepath
    flux_points_config : dict, optional
        Flux point configuration.

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

            if not is_in_flux_point_region(coord_re, flux_points_config):
                continue

            if DetHess < 0:
                x_point_coords_re.append(coord_re)           

            ## NOTE if you want the o-points to be local maxima use deltaPsi_xx < 0, if you want them to be local minima use  deltaPsi_xx > 0                  
            #if DetHess > 0 and deltaPsi_xx > 0:
            #   minima_location.append(coords)
            if DetHess > 0 and deltaPsi_xx < 0:
                o_point_coords_re.append(coord_re)

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

    return contour.get_paths()

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

def is_in_flux_point_region(coord_re, flux_points_config):
    """
    Check whether a flux point coordinate is in the configured region.

    Parameters
    ----------
    coord_re : list of float
        Coordinate in Earth radii, given as ``[x, y, z]``.
    flux_points_config : dict
        Flux point configuration.

    Returns
    -------
    bool
        Whether the coordinate should be used as a labeled flux point.
    """

    region_re = (flux_points_config or {}).get("region_re", {})

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