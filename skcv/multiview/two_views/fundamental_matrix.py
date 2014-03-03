import numpy as np
from numpy.linalg import svd

from skcv.multiview.util import euclidean_to_homogeneous
from skcv.multiview.util import homogeneous_to_euclidean
from skcv.multiview.util import normalize_points


def eight_point_algorithm(x1, x2):
    """ Computes the fundamental matrix from 8 (or more) projection
    point pairs

    Parameters
    ----------
    x1: numpy array
    projections of points in the first image, in homogeneous coordinates

    x2: numpy array
    projections of points in the second image, in homogeneous coordinates

    Returns
    -------
    F, the fundamental matrix satisfying x2.T * F * x1 = 0

    """

    n_points = x1.shape[1]
    if x2.shape != n_points:
        raise ValueError("Shape must be the same")

    # normalize points
    x1n, t1 = normalize_points(x1, is_homogeneous=True)
    x2n, t2 = normalize_points(x2, is_homogeneous=True)

    # build the vector
    a = np.vstack((x2n[0, :] * x1n,
                   x2n[1, :] * x1n,
                   x2n[2, :] * x1n))

    # find F in the normalized coordinates and transform it
    u, d, v = svd(a.T, full_matrices=True)
    f_matrix = np.reshape(v[8,:], (3, 3))

    # force the rank 2 constraint
    u, d, v = svd(f_matrix, full_matrices=True)
    d[2] = 0
    f_matrix = np.dot(u, np.dot(np.diag(d), v))

    # transform coordinates
    f_matrix = np.dot(t2.T, np.dot(f_matrix, t1))
    
    return f_matrix


def robust_F_estimation(x1, x2, max_iter=1000, inlier_threshold=2):
    """ Computes the fundamental matrix using the eight point algorithm
    (Hartley 1997)

    Parameters
    ----------
    x1: numpy array
    projections of points in the first image

    x2: numpy array
    projections of points in the second image

    max_iter: int, optional
    maximum number of iterations of the ransac algorithm

    inlier_threshold: float, optional
    maximum distance to consider a point pair inlier

    Returns
    -------
    F, the fundamental matrix satisfying x2.T * F * x1 = 0

    """

    iteration = 0
    while iteration < max_iter:
        pass