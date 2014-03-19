import numpy as np
from numpy.linalg import svd, norm, inv
from skcv.multiview.util import normalize_points

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

    #lambda matrix
    l = np.ones((n_points, n_views))

    #normalization matrices
    norm_matrices = []

    # normalize coordinates
    xn = np.zeros((3*n_points, n_views))
    for i in range(n_views):

        #find normalization matrix for projections i
        x_norm, T = normalize_points(x[i])
        xn[3*i:3*(i+1), :] = x_norm
        norm_matrices.append(T)

    while iterations < max_iterations:

        # normalize the lambda matrix
        lr_norm = norm(l, axis=1)
        ln = l / lr_norm
        lc_norm = norm(ln, axis=0)
        ln /= lc_norm

        #build the factorization matrix
        fact_matrix = np.kron(ln, xn)

        u, d, vh = svd(fact_matrix)

        d[4:] = 0

        # from the svd decomposition we can find the projections and 3d points
        p_matrices = np.dot(u, d)
        x_3d = vh
        iterations += 1

        #reproject each point and recompute lambdas
        if iterations != max_iterations:
            projs = np.dot(u, np.dot(d, vh))
            l = projs[::3, :]

    cameras = []
    for i in range(n_views):
        #denormalize camera matrices
        c_matrix = np.dot(inv(norm_matrices[i]), p_matrices[4*i:4*(i+1), :])
        cameras.append(c_matrix)

    return cameras, x_3d