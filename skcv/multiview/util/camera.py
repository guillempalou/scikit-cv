import numpy as np
from numpy.linalg import norm, svd, qr, det


def project(points, cameras):
    """ Generates point projections from a set of cameras
       
        Parameters
        ----------
        points: numpy array
        Array (4,N) of points representing a cloud in homogeneous coordinates
        
        cameras: list
        List of Q camera matrices to compute the projections
        
        Returns
        -------
        list of Q projections for the points 
    """

    #list of projections
    projections = []

    for camera in cameras:
        p = np.dot(camera, points)
        projections.append(p / p[2, :])

    return projections


def calibration_matrix(focal, skew=0, center=(0, 0), focal_y=None):
    """ Builds a calibration matrix from parameters

    Parameters
    ----------
    focal: float
    focal length

    skew: float, optional
    skew of the pixels, normally 0

    center: numpy array, optional
    center of the projection

    focal_y: float, optional
    focal length of the y axis

    """
    k = np.zeros((3, 3))

    k[0, 0] = focal
    if focal_y is not None:  # pragma: no cover
        k[1, 1] = focal_y
    else:                   # pragma: no cover
        k[1, 1] = focal

    k[0, 1] = skew
    k[0, 2] = center[0]
    k[1, 2] = center[1]
    k[2, 2] = 1

    return k


def internal_parameters(k_matrix):
    """ Extracts the intrinsic parameters from the matrix

        Parameters
        ----------
        k_matrix: numpy array


        Returns
        -------
        list of parameters

    """
    return k_matrix[0, 0], k_matrix[1, 1], \
           (k_matrix[0, 2], k_matrix[1, 2]), k_matrix[0, 1]


def camera_center(camera):
    """ Computes the camera center

    Parameters
    ----------
    camera: numpy array
    camera matrix

    Returns
    -------
    center of the camera

    """

    # camera center
    u, d, vh = svd(camera)
    center = vh[3, :]

    return center[:3] / center[3]


def camera_parameters(camera):
    """ Computes the camera center

    Parameters
    ----------
    camera: numpy array
    camera matrix

    Returns
    -------
    parameters of the camera

    """

    # get the center of the camera
    center = camera_center(camera)

    # get the left square matrix
    m = camera[:3, :3]

    #perform a RQ decomposition from a QR
    q, r = qr(np.flipud(m).T)
    r = np.flipud(r.T)
    q = q.T

    k = r[:, ::-1]
    r = q[::-1, :]

    #return the calibration matrix with positive focal lengths
    t = np.diag(np.sign(np.diag(k)))

    k = np.dot(k, t)
    r = np.dot(t, r)  #T is its own inverse

    if det(r) < 0:
        r *= -1

    return k, r, center


def look_at_matrix(center, look_at, up_vector=np.array((0, 1, 0))):
    """ Generates camera matrix using a center at a look at point
        the camera is assumed to be looking initially at (0,0,1)
        following the model of Zisserman, "Multiple View Geometry"
       
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
    w = look_at - center
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
