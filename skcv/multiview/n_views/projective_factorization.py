import numpy as np
from numpy.linalg import svd, norm, inv
from skcv.multiview.util import normalize_points
from skcv.multiview.two_views import eight_point_algorithm, right_epipole, left_epipole

def projective_factorization(x, max_iterations=1):
    """
    Computes the structure from point projections and camera matrices

    x: list
    list of numpy arrays, representing the points projections

    max_iterations: int, optional
    maximum number of iterations

    """

    n_views = len(x)
    n_points = x[0].shape[1]

    iterations = 0

    #lambda matrix, approximate depths
    l = np.ones((n_views, n_points))

    #normalization matrices
    norm_matrices = []

    # normalize coordinates
    xn = np.zeros((3*n_views, n_points))
    for i in range(n_views):

        #find normalization matrix for projections i
        x_norm, T = normalize_points(x[i], is_homogeneous=True)
        xn[3*i:3*(i+1), :] = x_norm
        norm_matrices.append(T)

    while iterations < max_iterations:
        # normalize the lambda matrix
        lr_norm = norm(l, axis=1)
        ln = l / lr_norm[:, np.newaxis]
        lc_norm = norm(ln, axis=0)
        ln /= lc_norm

        # repeat the lambdas
        ln = np.repeat(ln, 3, axis=0)

        #build the factorization matrix
        fact_matrix = ln*xn

        u, d, vh = svd(fact_matrix)

        print(d[3] / d[4])
        d = d[:4]/d[0]

        # from the svd decomposition we can find the projections and 3d points
        p_matrices = u[:, :4]
        x_3d = np.dot(np.diag(d), vh[:4, :])

        iterations += 1
        if iterations != max_iterations:

            w_matrix = np.dot(p_matrices, x_3d)

            for i in range(n_views):
                l[i, :] = w_matrix[3*i+2, :]

    cameras = []

    for i in range(n_views):
        # denormalize camera matrices
        c_matrix = np.dot(inv(norm_matrices[i]), p_matrices[3*i:3*(i+1), :])

        cameras.append(c_matrix)

    return cameras, x_3d