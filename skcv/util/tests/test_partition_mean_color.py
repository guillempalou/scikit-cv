import numpy as np
from numpy.testing import assert_equal
from skcv.util.partition_mean_color import partition_mean_color


def test_false_color():
    N = 100
    M = 100

    part = np.zeros((N,M))

    part[:N/2, :M/2] = 0
    part[N/2:, :M/2] = 1
    part[:N/2, M/2:] = 2
    part[N/2:, M/2:] = 3

    img = np.zeros((N,M,3))

    img[..., 0] = np.fromfunction(lambda i, j: i+j+0, (N,M))
    img[..., 1] = np.fromfunction(lambda i, j: i+j+1, (N,M))
    img[..., 2] = np.fromfunction(lambda i, j: i+j+2, (N,M))

    mean_color = partition_mean_color(img, part)

    colors = np.zeros((3,4))
    colors[:, 0] = np.array([49, 50, 51])
    colors[:, 1] = np.array([99, 100, 101])
    colors[:, 2] = np.array([99, 100, 101])
    colors[:, 3] = np.array([149, 150, 151])

    for i in range(4):
        coords = np.where(part == i)
        for ch in range(3):
            assert_equal(len(np.unique(mean_color[coords[0], coords[1], ch])), 1)
            assert_equal(mean_color[coords[0][0], coords[1][0], ch], colors[ch, i])
