import numpy as np
from numpy.testing import assert_equal, assert_almost_equal
from skcv.multiview.util.points_functions import *


def test_coordinate_transformation():
    x = np.arange(10)
    y = np.arange(10)
    points = np.vstack((x, y))

    points_h = euclidean_to_homogeneous(points)

    # check the conversion
    assert_equal(points_h[:2, :], points)
    assert_equal(points_h[2, :], np.ones(10))

    points_e = homogeneous_to_euclidean(points_h)

    assert_almost_equal(points_e, points)


def test_normalize_points():
    x = np.arange(10)
    y = np.arange(10)
    ones = np.ones(10)

    points = np.vstack((x, y))

    # result of the transformation
    t_gt = np.array([[0.34815531, 0., -1.5666989],
                  [0., 0.34815531, -1.5666989],
                  [0., 0., 1.]])

    # normalized points
    x_gt = np.array([[-1.5666989, -1.21854359, -0.87038828, -0.52223297, -0.17407766,
                      0.17407766, 0.52223297, 0.87038828, 1.21854359, 1.5666989],
                     [-1.5666989, -1.21854359, -0.87038828, -0.52223297, -0.17407766,
                      0.17407766, 0.52223297, 0.87038828, 1.21854359, 1.5666989]])

    x_n, t = normalize_points(points, is_homogeneous=False)

    assert_almost_equal(x_n, x_gt)
    assert_almost_equal(t, t_gt)

    #do the normalization in homogeneous coordinates
    points_h = np.vstack((x, y, ones))

    x_n, t = normalize_points(points_h, is_homogeneous=True)
    assert_almost_equal(x_n, np.vstack((x_gt, ones)))
    assert_almost_equal(t, t_gt)
