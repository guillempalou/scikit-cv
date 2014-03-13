import numpy as np
from numpy.linalg import inv, svd
from skcv.multiview.two_views.fundamental_matrix import *
#from skcv.multiview.two_views._triangulate_kanatani_cython import _triangulate_kanatani_cython

def _triangulate_hartley(x1, x2, f_matrix, P1, P2):
    """
    triangulates points according to

    Richard Hartley and Andrew Zisserman (2003). \"Multiple View Geometry in computer vision.\"

    """

    n_points = x1.shape[1]

    #3D points
    x_3d = np.zeros((4, n_points))

    for i in range(n_points):
        t = np.eye(3)
        tp = np.eye(3)

        # define transformation
        t[0, 2] = -x1[0, i]
        t[1, 2] = -x1[1, i]
        tp[0, 2] = -x2[0, i]
        tp[1, 2] = -x2[1, i]

        # translate matrix F
        f = np.dot(inv(tp).T, np.dot(f_matrix, inv(t)))

        # find normalized epipoles
        e = right_epipole(f)
        ep = left_epipole(f)

        e /= (e[0] ** 2 + e[1] ** 2)
        ep /= (ep[0] ** 2 + ep[1] ** 2)

        r = np.array(((e[0], e[1], 0), (-e[1], e[0], 0), (0, 0, 1)))
        rp = np.array(((ep[0], ep[1], 0), (-ep[1], ep[0], 0), (0, 0, 1)))

        f = np.dot(rp, np.dot(f, r.T))

        f1 = e[2]
        f2 = ep[2]
        a = f[1, 1]
        b = f[1, 2]
        c = f[2, 1]
        d = f[2, 2]

        # build a degree 6 polynomial
        coeffs = np.zeros(7)
        coeffs[0] = -(2 * a ** 2 * c * d * f1 ** 4 - 2 * a * b * c ** 2 * f1 ** 4)
        coeffs[1] = -(-2 * a ** 4 - 4 * a * 2 * c ** 2 * f2 ** 2 + 2 * a ** 2 * d ** 2 * f1 ** 4 -
                      2 * b ** 2 * c ** 2 * f1 ** 4 - 2 * c ** 4 * f2 ** 4)
        coeffs[2] = - (-8 * a ** 3 * b + 4 * a ** 2 * c * d * f1 ** 2 -
                       8 * a ** 2 * c * d * f2 ** 2 - 4 * a * b * c ** 2 * f1 ** 2 -
                       8 * a * b * c ** 2 * f2 ** 2 + 2 * a * b * d ** 2 * f1 ** 4 -
                       2 * b ** 2 * c * d * f1 ** 4 - 8 * c ** 3 * d * f2 ** 4)
        coeffs[3] = - (-12 * a ** 2 * b ** 2 + 4 * a ** 2 * d ** 2 * f1 ** 2 -
                       4 * a ** 2 * d ** 2 * f2 ** 2 - 16 * a * b * c * d * f2 ** 2 -
                       4 * b ** 2 * c ** 2 * f1 ** 2 - 4 * b ** 2 * c ** 2 * f2 ** 2 -
                       12 * c ** 2 * d ** 2 * f2 ** 4)
        coeffs[4] = - (2 * a ** 2 * c * d - 8 * a * b ** 3 - 2 * a * b * c ** 2 +
                       4 * a * b * d ** 2 * f1 ** 2 - 8 * a * b * d ** 2 * f2 ** 2 -
                       4 * b ** 2 * c * d * f1 ** 2 - 8 * b ** 2 * c * d * f2 ** 2 -
                       8 * c * d ** 3 * f2 ** 4)
        coeffs[5] = - (2 * a ** 2 * d ** 2 - 2 * b ** 4 - 2 * b ** 2 * c ** 2 -
                       4 * b ** 2 * d ** 2 * f2 ** 2 - 2 * d ** 4 * f2 ** 4)
        coeffs[6] = -2 * a * b * d ** 2 + 2 * b ** 2 * c * d

        roots = np.roots(coeffs)

        # evaluate the polinomial at the roots and +-inf
        vals = np.hstack((roots, [1e20]))

        min_s = 1e200
        min_v = 0

        # check all the polynomial roots
        for k in range(len(vals)):
            x = np.real(vals[k])

            s_t = x ** 2 / (1 + f1 ** 2 * x ** 2) + (c * x + d) ** 2 / \
                  ((a * x + b) ** 2 + f2 ** 2 * ((c * x + d) ** 2))

            if s_t < min_s:
                min_v = np.real(vals[k])
                min_s = s_t

        if min_v < 1e10:
            l = np.array((min_v * f1, 1, -min_v))
            lp = np.array((0, min_v, 1))
        else:
            l = np.array((f1, 0, -1))
            lp = np.array((0, 1, 0))

        lp = np.dot(f, lp)

        # find the point closest to the lines
        x = np.array((-l[0]*l[2], -l[1]*l[2], l[0]**2 + l[1]**2))
        xp = np.array((-lp[0]*lp[2], -lp[1]*lp[2], lp[0]**2 + lp[1]**2))

        x = np.dot(inv(t), np.dot(r.T, x))
        xp = np.dot(inv(tp), np.dot(rp.T, xp))

        # triangulate
        x_3d[:, i] = triangulate(x, xp, P1, P2)

    # return points
    return x_3d / x_3d[3, :]


# def _triangulate_kanatani(x1, x2, f_matrix, P1, P2,
#                           convergence=1e-6, max_iterations=10):
#     """
#     Triangulates according to
#     Triangulation from Two Views Revisited: Hartley-Sturm vs. Optimal Correction
#     Kenichi Kanatani
#
#     """
#
#
#     n_points = x1.shape[1]
#
#     #3D points
#     x_3d = np.zeros((4, n_points))
#
#     # sub matrix f_matrix
#
#     sub_f = f_matrix[:2, :2]
#
#     for i in range(n_points):
#         # initial energy
#         e0 = 1e10
#
#         # variables
#         dx1 = np.zeros(2)
#         dx2 = np.zeros(2)
#
#         # corrected coordinates
#         x1a = x1[:, i]
#         x2a = x2[:, i]
#         e = 0
#
#         xFx = np.dot(x2[:, i], np.dot(f_matrix, x1[:, i]))
#
#         iterations = 0
#
#         while np.abs(e - e0) > convergence and \
#                         iterations < max_iterations:
#
#             # variables
#             nk1 = np.dot(f_matrix.T, x2a)[:2]
#             nk2 = np.dot(f_matrix, x1a)[:2]
#             n = np.dot(nk1, nk1) + np.dot(nk2, nk2)
#
#             l = (xFx - np.dot(dx2, np.dot(sub_f, dx1))) / n
#
#             dx1 = l*nk1
#             dx2 = l*nk2
#
#             x1a = x1[:, i] - np.hstack((dx1[:2], 0))
#             x2a = x2[:, i] - np.hstack((dx2[:2], 0))
#
#             # check for convergence
#             e = np.dot(dx1, dx1) + np.dot(dx2, dx2)
#             if np.abs(e - e0) / e0 < convergence:
#                 break
#
#             e0 = e
#
#             iterations += 1
#
#         # triangulate
#         x_3d[:, i] = triangulate(x1a, x2a, P1, P2)
#
#     return x_3d / x_3d[3, :]
#
#
def triangulate(x1, x2, P1, P2):
    """
    Triangulates the 3D position from two projections and two cameras

    Parameters
    ----------

    x1: numpy array
        Projections on the first image

    x2: numpy array
        Projections on the second image


    Returns
    -------
    The 3D point in homogeneous coordinates

    """
    a = np.zeros((4, 4))
    a[0, :] = x1[0] * P1[2, :] - P1[0, :]
    a[1, :] = x1[1] * P1[2, :] - P1[1, :]
    a[2, :] = x2[0] * P2[2, :] - P2[0, :]
    a[3, :] = x2[1] * P2[2, :] - P2[1, :]

    u, d, v = svd(a)

    # the point lies on the null space of matrix a
    return v[3,:]


def optimal_triangulation(x1, x2, f_matrix, cameras=None, method='Hartley'):
    """
    Triangulates point projections using an optimal solution

    Parameters
    ----------

    x1: numpy array
        Projections on the first image

    x2: numpy array
        Projections on the second image

    f_matrix: numpy array
        Fundamental matrix

    cameras: 2-tuple, optional
        cameras from the two projections
        if none are provided, the two canonical are obtained

    Returns
    -------

    """
    #xn, t = normalize_points(x, is_homogeneous=True)

    if cameras is None:
        p1, p2 = canonical_cameras_from_f(f_matrix)
    else:
        p1, p2 = cameras

    if method == 'Hartley':
        x_3d = _triangulate_hartley(x1, x2, f_matrix, p1, p2)
    elif method == 'Kanatani':
        x_3d = _triangulate_kanatani(x1, x2, f_matrix, p1, p2)

    return x_3d

def find_projection_matrix_from_points(x, x_3d):
    pass