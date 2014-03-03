import numpy as np
from numpy.testing import assert_almost_equal, assert_equal

from skcv.image.segmentation.bpt import BPT
from skcv.image.segmentation.region_distances import mean_color_distance


def test_bpt():
    N = 100
    M = 100

    part = np.zeros((N,M))

    part[:N/2,:M/2] = 0
    part[N/2:,:M/2] = 1
    part[:N/2,M/2:] = 2
    part[N/2:,M/2:] = 3

    f = lambda r, c, d: (part[r.astype(np.int),c.astype(np.int)]/4 + d/12)

    img = np.fromfunction(f,
                          (N, M, 3),
                          dtype=np.float64)

    b = BPT(img,part,mean_color_distance,update_partition=True)

    assert_equal(b.nodes[0]["childs"], [])
    assert_equal(b.nodes[1]["childs"], [])
    assert_equal(b.nodes[2]["childs"], [])
    assert_equal(b.nodes[3]["childs"], [])
    assert_equal(b.nodes[0]["parent"], 5)
    assert_equal(b.nodes[1]["parent"], 5)
    assert_equal(b.nodes[2]["parent"], 4)
    assert_equal(b.nodes[3]["parent"], 4)
    assert_equal(b.nodes[4]["childs"], [2, 3])
    assert_equal(b.nodes[5]["childs"], [0, 1])
    assert_equal(b.nodes[4]["parent"], 6)
    assert_equal(b.nodes[5]["parent"], 6)
    assert_equal(b.nodes[6]["childs"], [5, 4])