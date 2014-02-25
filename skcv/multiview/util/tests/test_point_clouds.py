import numpy as np
from numpy.testing import assert_almost_equal, assert_array_less, assert_equal
from skcv.multiview.util.synthetic_point_cloud import *

def test_random_sphere():
    n_points = 10
    radius = 7
    center = np.array((1, 2, 3))

    points = random_sphere(n_points, radius=radius, center=center)

    norm = np.linalg.norm(points - center[:, np.newaxis], axis=0)

    assert_almost_equal(radius*np.ones(n_points), norm)


def test_random_ball():
    n_points = 10
    radius = 7
    center = np.array((1, 2, 3))

    points = random_ball(n_points, radius=radius, center=center)

    norm = np.linalg.norm(points - center[:, np.newaxis], axis=0)

    assert_array_less(norm, radius*np.ones(n_points))


def test_random_cube():

    n_points = 10
    size = 5
    center = np.array((1, 2, 3))

    points = random_cube(n_points, size, center)
    points_c = points - center[:, np.newaxis]

    in_cube = np.zeros(10, dtype=np.bool)

    for i in range(3):
        b = (abs(points_c[i, :]) - 0.5*size) < 1e-6
        in_cube = np.logical_or(in_cube, b)

    assert_equal(np.all(in_cube), True)