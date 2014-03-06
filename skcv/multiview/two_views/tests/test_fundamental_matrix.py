import numpy as np
from numpy.testing import assert_almost_equal, assert_allclose
import pickle
import os
from skcv import data_dir
from skcv.multiview.two_views.fundamental_matrix import *
from skcv.multiview.util.points_functions import *
from skcv.multiview.util.camera import *


def test_fundamental_from_cameras():
    center1 = np.array((0, 0, 0))
    center2 = np.array((1, 0, 0))

    look_at1 = np.array((0, 0, 10))
    look_at2 = np.array((1, 0, 10))

    camera1 = look_at_matrix(center1, look_at1)
    camera2 = look_at_matrix(center2, look_at2)

    f_matrix = fundamental_matrix_from_two_cameras(camera1, camera2)


def test_epipoles():
    f_matrix = np.array(((0, 0, 0), (0, 0, -1), (0, 1, 0)))

    re = right_epipole(f_matrix)
    le = left_epipole(f_matrix)

    assert_almost_equal(re, [1, 0, 0])
    assert_almost_equal(le, [1, 0, 0])


def test_canonical_cameras():
    camera1 = np.array([[1., 0., 0., 0.],
                        [0., 1., 0., 0.],
                        [0., 0., 1., 0.]])
    camera2 = np.array([[0., 0., 0., 1.],
                        [0., -1., 0., 0.],
                        [0., 0., -1., 0.]])

    f_matrix = np.array([[0, 0, 0],
                         [0, 0, -1],
                         [0, 1, 0]])

    p1, p2 = canonical_cameras_from_f(f_matrix)

    assert_almost_equal(p1, camera1)
    assert_almost_equal(p2, camera2)


def test_sampson_error():
    f_matrix = np.array([[0, 0, 0],
                         [0, 0, -1],
                         [0, 1, 0]])

    x = np.linspace(0, 10, 11)
    y = np.linspace(0, 10, 11)
    ones = np.ones(11)

    x1 = np.vstack((x, y, ones))
    x2 = np.vstack((x + 1, y + 1, ones))
    x3 = np.vstack((x + 1, y, ones))

    error = sampson_error(x1, x2, f_matrix)

    gt_err = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])

    assert_almost_equal(gt_err, error)

    error = sampson_error(x1, x3, f_matrix)

    assert_almost_equal(np.zeros(11), error)


def test_eight_point_algorithm():
    projections_file = os.path.join(data_dir, 'two_view_projections.dat')

    (x1e, x2e) = pickle.load(open(projections_file, 'rb'))

    x1h = euclidean_to_homogeneous(x1e)
    x2h = euclidean_to_homogeneous(x2e)

    f_matrix = eight_point_algorithm(x1h, x2h)

    # fundamental matrix corresponding to an horizontal displacement
    f_groundtruth = np.array(((0, 0, 0),
                              (0, 0, -1),
                              (0, 1, 0)))

    f_matrix /= np.max(f_matrix)

    assert_almost_equal(f_matrix, f_groundtruth)


def test_robust_f_estimation():
    projections_file = os.path.join(data_dir, 'two_view_projections.dat')

    (x1e, x2e) = pickle.load(open(projections_file, 'rb'))

    #add gaussian noise to x1e and x2e
    dev = 0.1
    x1e += np.random.normal(0, dev, size=x1e.shape)
    x2e += np.random.normal(0, dev, size=x2e.shape)

    x1h = euclidean_to_homogeneous(x1e)
    x2h = euclidean_to_homogeneous(x2e)

    f_matrix = robust_f_estimation(x1h, x2h)

    # fundamental matrix corresponding to an horizontal displacement
    f_groundtruth = np.array(((0, 0, 0),
                              (0, 0, -1),
                              (0, 1, 0)))

    #the sampson error should be equal to the noise variance
    e_gt = sampson_error(x1h, x2h, f_groundtruth)
    e = sampson_error(x1h, x2h, f_matrix)

    assert_allclose(np.sqrt(np.mean(e_gt)), dev, rtol=1e-1)
    assert_allclose(np.sqrt(np.mean(e)), dev, rtol=1e-1)

    f_matrix /= np.max(f_matrix)

    assert_allclose(f_matrix, f_groundtruth, atol=0.1)


