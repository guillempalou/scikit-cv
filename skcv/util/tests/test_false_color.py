import numpy as np
from numpy.testing import assert_equal
from skcv.util.false_color import false_color


def test_false_color():
    N = 100
    M = 100

    part = np.zeros((N,M))

    part[:N/2, :M/2] = 0
    part[N/2:, :M/2] = 1
    part[:N/2, M/2:] = 2
    part[N/2:, M/2:] = 3

    img = false_color(part)

    for i in range(4):
        coords = np.where(part == i)
        for ch in range(3):
            assert_equal(len(np.unique(img[coords[0], coords[1], ch])), 1)
