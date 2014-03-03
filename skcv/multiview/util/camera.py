import numpy as np
from numpy.linalg import norm

def project(points, cameras):
    """ Generates point projections from a set of cameras
       
        Parameters
        ----------
        points: numpy array
        Array (3,N) of points representing a cloud
        
        cameras: list
        List of Q camera matrices to compute the projections
        
        Returns
        -------
        list of Q projections for the points 
    """
    
    #list of projections
    projections = []
    
    for camera in cameras:
        projections.append(np.dot(camera, points))
    
    return projections


def look_at_matrix(center, look_at, up_vector = np.array((0,1,0))):
    """ Generates camera matrix using a center at a look at point
       
        Parameters
        ----------
        center: numpy array
        Vector representing the camera center
        
        look_at: numpy array
        Vector representing the point to look at

        up_vector: numpy array, option
        The camera up vector

        Returns
        -------
        External camera matrix
    """
    
    # form the pointing vector. the camera looks at -w
    w = center - look_at
    nw = w / norm(w)
    
    # form the up vector
    u = np.cross(up_vector, nw)
    nu = u / norm(u)
    
    # form the last vector
    v = np.cross(nw, nu)
    nv = v / norm(v)
    
    #build the camera matrix
    external = np.vstack((nu, nv, nw))
    rt = np.dot(external, -center)
    external = np.hstack((external, rt[:, np.newaxis]))
    
    return external
