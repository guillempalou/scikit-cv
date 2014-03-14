#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False

import numpy as np
from numpy.linalg import svd

cdef inline double _abs(double a): return a if a >= 0. else -a

def _triangulate_kanatani(double[:, ::1] x1,
                          double[:, ::1] x2,
                          double[:, ::1] f_matrix,
                          double[:, ::1] P1,
                          double[:, ::1] P2,
                          double convergence=1e-6, int max_iterations=10):
    """
    Triangulates according to
    Triangulation from Two Views Revisited: Hartley-Sturm vs. Optimal Correction
    Kenichi Kanatani

    """


    cdef int n_points = x1.shape[1]

    #iteration variables
    cdef int i, j, k, iterations

    cdef float n, xFx, dfd

    #gradient information
    cdef double[::1] dx1
    cdef double[::1] dx2
    cdef double[::1] x1a = np.zeros(3)
    cdef double[::1] x2a = np.zeros(3)
    cdef double[:, ::1] sub_f
    cdef double[::1] nk1 = np.zeros(2)
    cdef double[::1] nk2 = np.zeros(2)

    # to check for convergence
    cdef float e0, e

    #triangulation matrix
    cdef double[:, ::1] a = np.zeros((4, 4))

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
        x1a[0] = x1[0, i]
        x1a[1] = x1[1, i]
        x1a[2] = x1[2, i]

        x2a[0] = x2[0, i]
        x2a[1] = x2[1, i]
        x2a[2] = x2[2, i]

        e = 0

        # perform the x2'*F*x1 product
        xFx = 0

        for k in range(3):
            xFx += x1a[k]*(x2a[0]*f_matrix[0, k] +
                           x2a[1]*f_matrix[1, k] +
                           x2a[2]*f_matrix[2, k])

        iterations = 0

        while _abs(e - e0) > convergence and \
              iterations < max_iterations:

            # variables
            #nk1 = np.dot(f_matrix.T, x2a)[:2]
            nk1[0] = f_matrix[0, 0]*x2a[0] + f_matrix[1, 0] * x2a[1] + f_matrix[2, 0] * x2a[2]
            nk1[1] = f_matrix[0, 1]*x2a[0] + f_matrix[1, 1] * x2a[1] + f_matrix[2, 1] * x2a[2]
            #nk2 = np.dot(f_matrix, x1a)[:2]
            nk2[0] = f_matrix[0, 0]*x1a[0] + f_matrix[0, 1] * x1a[1] + f_matrix[0, 2] * x1a[2]
            nk2[1] = f_matrix[1, 0]*x1a[0] + f_matrix[1, 1] * x1a[1] + f_matrix[1, 2] * x1a[2]

            # n = np.dot(nk1, nk1) + np.dot(nk2, nk2)
            n = nk1[0]*nk1[0] + nk1[1]*nk1[1] + nk2[0]*nk2[0] + nk2[1]*nk2[1]

            # np.dot(dx2, np.dot(sub_f, dx1))
            dfd = dx1[0] * (dx2[0]*f_matrix[0, 0] + dx2[1]*f_matrix[1, 0]) + \
                  dx1[1] * (dx2[0]*f_matrix[0, 1] + dx2[1]*f_matrix[1, 1])

            l = (xFx - dfd) / n

            dx1[0] = l*nk1[0]
            dx1[1] = l*nk1[1]
            dx2[0] = l*nk2[0]
            dx2[1] = l*nk2[1]

            #x1a = x1[:, i] - np.hstack((dx1[:2], 0))
            x1a[0] = x1[0, i] - dx1[0]
            x1a[1] = x1[1, i] - dx1[1]

            #x2a = x2[:, i] - np.hstack((dx2[:2], 0))
            x2a[0] = x2[0, i] - dx2[0]
            x2a[1] = x2[1, i] - dx2[1]

            # check for convergence
            e = dx1[0]*dx1[0] + dx1[1]*dx1[1] + dx2[0]*dx2[0] + dx2[1]*dx2[1]

            if abs(e - e0) / e0 < convergence:
                break

            e0 = e

            iterations += 1

        for k in range(4):
            a[0, k] = x1a[0] * P1[2, k] - P1[0, k]
            a[1, k] = x1a[1] * P1[2, k] - P1[1, k]
            a[2, k] = x2a[0] * P2[2, k] - P2[0, k]
            a[3, k] = x2a[1] * P2[2, k] - P2[1, k]

        u, d, vh = svd(a)

        # the point lies on the null space of matrix a
        x_3d[:, i] = vh[3, :]

    return x_3d / x_3d[3, :]