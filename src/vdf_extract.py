import numpy as np
import analysator as pt

def extract_vdf(file, cid, box=-1, pop="avg"):
    """
    Extract a 3D VDF from one Vlasiator cell.

    This function reads the sparse velocity-space data stored in a `.vlsv`
    file, places the values into a full velocity-space grid, sorts the grid by
    velocity coordinates, and returns the VDF as a dence NumPy array.

    Parameters
    ----------
    file : str
        Path to the `.vlsv` file.
    cid : int
        Spatial cell ID from which the VDF is extracted.
    box : int, optional
        If positive, crop the VDF around its maximum value using this value as
        the half-width of the crop in index space. If `-1`, return VDF.
    pop : str, optional
        Particle population name used by Analysator.

    Returns
    -------
    numpy.ndarray
        Dense 3D VDF array with dtype `float32`. The returned axis order `[vx, vy, vz]`.
    """

    assert cid>0
    reader = pt.vlsvfile.VlsvReader(file)
    #read phase space density
    vcells = reader.read_velocity_cells(cid, pop)
    keys = list(vcells.keys())
    values = list(vcells.values())

    #generate a velocity space
    size = reader.get_velocity_mesh_size(pop)
    vids = np.arange(4 * 4 * 4 * int(size[0]) * int(size[1]) * int(size[2]))

    #put phase space density into array
    dist = np.zeros_like(vids,dtype=float)
    dist[keys] = values

    #sort vspace by velocity
    v = reader.get_velocity_cell_coordinates(vids, pop)

    i = np.argsort(v[:,0],kind='stable')
    v = v[i]
    #vids = vids[i]
    dist = dist[i]

    j = np.argsort(v[:,1],kind='stable')
    v = v[j]
    #vids = vids[j]
    dist = dist[j]

    k = np.argsort(v[:,2],kind='stable')
    v = v[k]
    #vids = vids[k]
    dist = dist[k]
    dist = dist.reshape(4*int(size[0]),4*int(size[1]),4*int(size[2]))
    vdf=dist
    i,j,k = np.unravel_index(np.argmax(vdf), vdf.shape)
    len=int(box)
    if box >0:
        data=vdf[(i-len):(i+len),(j-len):(j+len),(k-len):(k+len)]
    else:
        data =vdf

    data=np.swapaxes(data,2,0)
    return np.array(data,dtype=np.float32)