import numpy as np

def eight_point_algorithm(x1, x2, max_iter=1000, inlier_threshold=2):
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

        while i < max_iter: