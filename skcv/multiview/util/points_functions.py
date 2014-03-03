import numpy as np


def euclidean_to_homogeneous(x):
    """
    Transforms X to be in homogeneous coordinates

    Parameters
    ----------
    x: numpy array
    each column of the array is a point


    Returns
    -------
    xh: numpy array,
    x in euclidean coordinates
    """

    xe = np.vstack((x, np.ones(x.shape[1])))

    return xe

def homogeneous_to_euclidean(xh):
    """
    Transforms X to be in euclidean coordinates

    Parameters
    ----------
    x: numpy array (3,N), (4,N)
    each column of the array is a point

    Returns
    -------
    xh: numpy array,
    x in homogeneous coordinates
    """

    return xh[0:-1, :]/xh[-1, :]


def normalize_points(x, is_homogeneous=False):
    """
    Normalizes points so that they have mena 0 and variance 1
    accross dimensions

    Parameters
    ----------
    x: numpy array
    array (D, N) with the points. Each columns is a
    D-dimensional point

    Returns
    -------
    Xn: numpy array,
    normalized points

    """
    dimensions = x.shape[0]

    mu_x = np.mean(x, axis=1)
    std_x = np.std(x, axis=1)

    # if x is in homogeneous coordinates
    if is_homogeneous:
        dimensions -= 1
        mu_x = mu_x[:-1]
        std_x = std_x[:-1]

    size = dimensions + 1

    # build the transformation matrix
    t = np.eye(size)
    t[:dimensions, -1] = -mu_x/std_x
    diag = np.arange(0, dimensions)
    t[diag, diag] = 1/std_x

    # normalize the points
    mu_x_mat = np.repeat(mu_x[:, np.newaxis], x.shape[1], axis=1)
    x_normalized = (x[:dimensions,:] - mu_x_mat) / std_x[:, np.newaxis]

    # put 1 to the last coordinate if we work in homogeneous coordinates
    if is_homogeneous:
        x_normalized = np.vstack((x_normalized, np.ones(x.shape[1])))

    return x_normalized, t
