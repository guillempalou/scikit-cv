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


def look_at_matrix(center, look_at):
    """ Generates camera matrix using a center at a look at point
       
        Parameters
        ----------
        center: numpy array
        Vector representing the camera center
        
        look_at: numpy array
        Vector representing the point to look at
        
        Returns
        -------
        External camera matrix
    """
    
    # form the pointing vector. the camera looks at -w
    w = center - look_at
    w = w / norm(w)
    
    # form the up vector
    u = np.cross(np.array((0, 0, 1)), w)
    u = u / norm(u)
    
    # form the last vector
    v = np.cross(w,u)
    v = v / norm(v)
    
    #build the camera matrix
    external = np.vstack((u,v,w))
    rt = np.dot(external, -center)
    external = np.hstack((external,rt[:,np.newaxis]))
    
    return external
        