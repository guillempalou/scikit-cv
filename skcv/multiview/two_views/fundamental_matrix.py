import numpy as np
from numpy.linalg import svd
from math import log

from skcv.multiview.util import normalize_points

def fundamental_matrix_from_two_cameras(camera1, camera2):
    """ Computes the fundamental matrix from two projection
    matrices

    Parameters
    ----------
    camera1: numpy array
    Projection matrix of first camera

    camera2: numpy array
    Projection matrix of second camera

    Returns
    -------
    fundamental matrix

    """

    Pp = np.linalg.pinv(camera1)

    # camera center
    u, d, vh = svd(camera1)
    center = vh[3, :]

    # epipole on the second image
    e = np.dot(camera2, center)

    se = np.array(((0, -e[2], e[1]),
                   (e[2], 0, -e[0]),
                   (-e[1], e[0], 0)))

    f_matrix = np.dot(se, np.dot(camera2, Pp))

    return f_matrix


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
    if x2.shape[1] != n_points:  # pragma: no cover
        raise ValueError("Shape must be the same")

    # normalize points
    x1n, t1 = normalize_points(x1, is_homogeneous=True)
    x2n, t2 = normalize_points(x2, is_homogeneous=True)

    # build the vector
    a = np.vstack((x2n[0, :] * x1n,
                   x2n[1, :] * x1n,
                   x2n[2, :] * x1n))

    # find F in the normalized coordinates and transform it
    u, d, vh = svd(a.T, full_matrices=True)
    f_matrix = np.reshape(vh[8, :], (3, 3))

    # force the rank 2 constraint
    u, d, vh = svd(f_matrix, full_matrices=True)
    d[2] = 0
    f_matrix = np.dot(u, np.dot(np.diag(d), vh))

    # transform coordinates
    f_matrix = np.dot(t2.T, np.dot(f_matrix, t1))

    return f_matrix


def right_epipole(f_matrix):
    """
    Computes the right epipole (first image) of fundamental matrix
    the right epipole satisfies Fe = 0

    """

    u, d, vh = svd(f_matrix)
    return vh[2, :]


def left_epipole(f_matrix):
    """
    Computes the right epipole (first image) of fundamental matrix
    the right epipole satisfies Fe = 0

    """

    u, d, vh = svd(f_matrix)
    return u[:, 2]


def canonical_cameras_from_f(f_matrix):
    """
    Retrieves the two canonical cameras given a fundamental matrix

    """
    # the first camera is the identity
    camera1 = np.eye(3, 4)

    e = left_epipole(f_matrix)

    se = np.array(((0, -e[2], e[1]),
                   (e[2], 0, -e[0]),
                   (-e[1], e[0], 0)))

    camera2 = np.hstack((np.dot(se, f_matrix), e[:, np.newaxis]))

    return camera1, camera2

def sampson_error(x1, x2, f_matrix):
    """
    Computes the sampson error for a set of point pairs

    Parameters
    ----------
    x1: numpy array
    projections of points in the first image, in homogeneous coordinates

    x2: numpy array
    projections of points in the second image, in homogeneous coordinates

    f_matrix: numpy_array
    fundamental matrix

    Returns
    -------
    sampson error of each point pair

    """

    f_x1 = np.dot(f_matrix, x1)
    f_x2 = np.dot(f_matrix.T, x2)

    #get the denominator
    den = np.sum(f_x1[:2, :] ** 2, axis=0) +\
          np.sum(f_x2[:2, :] ** 2, axis=0)

    #get the numerator
    num = np.sum((x2 * f_x1), axis=0)**2

    return num / den


def reprojection_error(x1, x2, f_matrix):
    """
    Computes the sampson error for a set of point pairs

    Parameters
    ----------
    x1: numpy array
    projections of points in the first image, in homogeneous coordinates

    x2: numpy array
    projections of points in the second image, in homogeneous coordinates

    f_matrix: numpy_array
    fundamental matrix

    Returns
    -------
    reprojection error of each point pair

    """


def robust_f_estimation(x1, x2,
                        max_iter=1000,
                        distance='sampson',
                        n_samples=8,
                        prob = 0.99,
                        inlier_threshold=2):
    """ Computes the fundamental matrix using the eight point algorithm
    (Hartley 1997)

    Parameters
    ----------
    x1: numpy array
    projections of points in the first image, in homogeneous coordinates

    x2: numpy array
    projections of points in the second image, in homogeneous coordinates

    max_iter: int, optional
    maximum number of iterations of the ransac algorithm

    distance: string, option
    distance to use to find inliers/outliers

    n_samples: int, optional
    number of points to samples at each RANSAC iteration

    prob: float, optional
    probability of having a free from outliers sample

    inlier_threshold: float, optional
    maximum distance to consider a point pair inlier

    Returns
    -------
    F, the fundamental matrix satisfying x2.T * F * x1 = 0

    """

    iteration = 0
    n_points = x1.shape[1]

    is_inlier = np.zeros(n_points, dtype=bool)

    # variables to store the best result found
    best_inliers = is_inlier
    best_n_inliers = 0

    while iteration < max_iter:
        #select 8 points at random
        idx = np.random.choice(n_points, n_samples, replace=False)

        selected_x1 = x1[:, idx]
        selected_x2 = x2[:, idx]

        #get inliers
        f_matrix = eight_point_algorithm(selected_x1,
                                         selected_x2)
        # find the error distance
        if distance == 'sampson':
            e = sampson_error(x1, x2, f_matrix)
        else:  # pragma : no cover
            raise ValueError()

        is_inlier = e < inlier_threshold

        n_inliers = np.count_nonzero(is_inlier)

        if n_inliers > best_n_inliers:
            best_inliers = is_inlier
            best_n_inliers = n_inliers

        #update max_iterations if estimation is improved
        # the epsilon (1e-10) is added in case of all inliers
        eps = 1 - n_inliers / n_points + 1e-10
        new_iter = log(1 - prob) / log(1 - (1-eps)**n_samples)

        if new_iter < max_iter:
            max_iter = new_iter

        print(n_inliers, max_iter)
        iteration += 1

    #refine the estimate using all inliers
    best_x1 = x1[:, best_inliers]
    best_x2 = x2[:, best_inliers]

    f_matrix = eight_point_algorithm(best_x1, best_x2)

    return f_matrix