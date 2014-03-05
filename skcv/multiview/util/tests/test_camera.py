import numpy as np
from numpy.testing import assert_equal, assert_almost_equal
from skcv.multiview.util.points_functions import *
from skcv.multiview.util.camera import *


def test_project():
    camera = np.array(((100, 0, 0, 0),
                       (0, 100, 0, 0),
                       (0, 0, 1, 0)))

    X1 = np.array((1, 2, 2, 1))

    x1 = project(X1, [camera])
    x1 = hnormalize(x1[0])

    projection = np.array((50, 100, 1))

    assert_almost_equal(x1, projection)


def test_camera_center():
    center1 = np.array((10,10,10))
    look_at = np.zeros(3)

    camera1 = look_at_matrix(center1, look_at)

    t = camera_center(camera1)

    assert_almost_equal(t, center1)


def test_camera_parameters():
    center1 = np.array((10, 10, 10))
    look_at = np.zeros(3)

    camera1 = look_at_matrix(center1, look_at)
    K = np.array(((100, 1, 10), (0, 150, 15), (0, 0, 1)))

    c1 = np.dot(K, camera1)

    k, r, t = camera_parameters(c1)

    assert_almost_equal(k, K)
    assert_almost_equal(r, camera1[:,:3])
    assert_almost_equal(center1, t)


def test_internal_parameters():
    k_matrix = calibration_matrix(100, focal_y=150, skew=1, center=(10, 15))
    K = np.array(((100, 1, 10), (0, 150, 15), (0, 0, 1)))

    focal_x, focal_y, center, skew = internal_parameters(K)

    assert_equal(focal_x, 100)
    assert_equal(focal_y, 150)
    assert_equal(skew, 1)
    assert_equal(center, (10, 15))

    assert_equal(K, k_matrix)



def test_look_at():
    K = np.array(((100, 0, 0, 0),
                  (0, 100, 0, 0),
                  (0, 0, 1, 0)))

    camera = np.array(((1, 0, 0, 0),
                       (0, 1, 0, 0),
                       (0, 0, 1, 0)))

    center = np.array((0, 0, 0))
    look_at = np.array((0, 0, 10))

    camera1 = look_at_matrix(center, look_at)

    assert_equal(camera, camera1)
