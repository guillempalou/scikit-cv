import numpy as np
from numpy.linalg import svd, inv
from skcv.multiview.util import camera_parameters


def linear_autocalibration(cameras, internal_parameters, n_iterations=50):
    """
    Computes the homography H to transform a projective reconstruction
    into a metric such that:
    Pm = P*H
    Xm = H^-1*X

    Parameters
    ----------
    cameras: list
    List of camera matrices

    internal_parameters: numpy array
    Approximate internal camera matrix

    n_iterations: int, optional
    number of iterations for varying the variance of the focal length

    Returns
    -------

    Homography satisfying the above equations

    """

    n_views = len(cameras)

    k_pars = internal_parameters
    ki = inv(k_pars)

    ratio = k_pars[1, 1] / k_pars[0, 0]

    norm_cameras = [np.dot(ki,cam) for cam in cameras]

    betas = 0.1*np.exp(0.3*np.linspace(0, n_iterations))

    min_cost = 1e200
    best_t = np.array(())

    # transform a 16 vector of a symmatric matrix to a 10-vector
    # it could be precomputed
    idx = np.array((0, 1, 2, 3, 1, 4, 5, 6, 2, 5, 7, 8, 3, 6, 8, 9))
    h = np.zeros((16, 10))
    for i in range(10):
        h[idx == i, i] = 1

    # assumptions abour the deviation of the normalized internal parameters
    skew_sigma = 0.01
    center_sigma = 0.1
    focal_sigma = 0.2

    for beta in betas:

        # build the least squares problem
        chi = np.zeros((6*n_views, 10))
        for i in range(n_views):
            p1 = norm_cameras[i][0, :]
            p2 = norm_cameras[i][1, :]
            p3 = norm_cameras[i][2, :]

            #linearize the absolute quadric constraints
            chi[6*i, :] = (1./skew_sigma) * np.dot(np.kron(p1, p2), h)
            chi[6*i+1, :] = (1./center_sigma) * np.dot(np.kron(p1, p3), h)
            chi[6*i+2, :] = (1./center_sigma) * np.dot(np.kron(p2, p3), h)
            chi[6*i+3, :] = (1./focal_sigma) * np.dot(np.kron(p1, p1) - np.kron(p2, p2), h)
            chi[6*i+4, :] = (1./beta) * np.dot(np.kron(p1, p1) - np.kron(p3, p3), h)
            chi[6*i+5, :] = (1./beta) * np.dot(np.kron(p2, p2) - np.kron(p3, p3), h)

        # solve the system and build H from svd
        u, d, vh = svd(chi)

        # the quadric is the last eigenvector (null-space)
        q = vh[9, :]
        quadric = np.array(((q[0], q[1], q[2], q[3]),
                            (q[1], q[4], q[5], q[6]),
                            (q[2], q[5], q[7], q[8]),
                            (q[3], q[6], q[8], q[9])))

        u, d, vh = svd(quadric)
        t = np.zeros((4, 4))
        t[:, :3] = np.dot(u, np.diag(np.sqrt(d)))[:, :3]
        t[3, 3] = 1
        print(quadric)

        # compute the cost and keep the minimum
        cost = 0
        for i in range(n_views):

            c = np.dot(cameras[i], t)
            k, r, center = camera_parameters(c)
            k /= k[2, 2]

            print("Parameters:\n", k)

            cost += (k[0, 1]**2 + (k[1, 1] / k[0, 0] - ratio) +
                     k[0, 2]**2 + k[1, 2]**2) / k[0, 0]**2

        print(cost)

        if cost < min_cost:
            best_t = t
            min_cost = cost

    return best_t