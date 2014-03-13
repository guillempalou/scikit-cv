import numpy as np

from .triangulation import triangulate

# def triangulate_kanatani(double[:, :] x1,
#                           double[:, :] x2,
#                           double[:, :] f_matrix,
#                           double[:, :] P1,
#                           double[:, :] P2,
#                           double convergence=1e-6, int max_iterations=10):
#     """
#     Triangulates according to
#     Triangulation from Two Views Revisited: Hartley-Sturm vs. Optimal Correction
#     Kenichi Kanatani
#
#     """
#
#
#     cdef int n_points = x1.shape[1]
#
#     #iteration variables
#     cdef int i, iterations
#
#     #gradient information
#     cdef double[:] dx1
#     cdef double[:] dx2
#     cdef double[:] x1a
#     cdef double[:] x2a
#     cdef double[:, :] sub_f
#     cdef double[:] nk1 = np.zeros(2)
#     cdef double[:] nk2 = np.zeros(2)
#
#     # to check for convergence
#     cdef float e0, e
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
#         # perform the x2'*F*x1 product
#         xFx = np.dot(x2[:, i], np.dot(f_matrix, x1[:, i]))
#
#         iterations = 0
#
#         while np.abs(e - e0) > convergence and \
#               iterations < max_iterations:
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


def triangulate_kanatani(x1, x2, f_matrix, P1, P2,
                        convergence=1e-6, max_iterations=10):
    """
    Triangulates according to
    Triangulation from Two Views Revisited: Hartley-Sturm vs. Optimal Correction
    Kenichi Kanatani

    """


    n_points = x1.shape[1]

    #3D points
    x_3d = np.zeros((4, n_points))

    # sub matrix f_matrix

    sub_f = f_matrix[:2, :2]

    for i in range(n_points):
        # initial energy
        e0 = 1e10

        # variables
        dx1 = np.zeros(2)
        dx2 = np.zeros(2)

        # corrected coordinates
        x1a = x1[:, i]
        x2a = x2[:, i]
        e = 0

        xFx = np.dot(x2[:, i], np.dot(f_matrix, x1[:, i]))

        iterations = 0

        while np.abs(e - e0) > convergence and \
                        iterations < max_iterations:

            # variables
            nk1 = np.dot(f_matrix.T, x2a)[:2]
            nk2 = np.dot(f_matrix, x1a)[:2]
            n = np.dot(nk1, nk1) + np.dot(nk2, nk2)

            l = (xFx - np.dot(dx2, np.dot(sub_f, dx1))) / n

            dx1 = l*nk1
            dx2 = l*nk2

            x1a = x1[:, i] - np.hstack((dx1[:2], 0))
            x2a = x2[:, i] - np.hstack((dx2[:2], 0))

            # check for convergence
            e = np.dot(dx1, dx1) + np.dot(dx2, dx2)
            if np.abs(e - e0) / e0 < convergence:
                break

            e0 = e

            iterations += 1

        # triangulate
        x_3d[:, i] = triangulate(x1a, x2a, P1, P2)

    return x_3d / x_3d[3, :]