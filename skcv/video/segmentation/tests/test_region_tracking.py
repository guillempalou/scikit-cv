import numpy as np
from numpy.testing import assert_equal
from skcv.video.segmentation.region_tracking import *

def test_bipartite_region_tracking():
    N = 10
    M = 10

    part = np.fromfunction(lambda i, j: i*N+j, (N, M))

    gt_part = np.zeros((2, N, M))

    gt_part[0, ...] = part
    gt_part[1, ...] = part

    gt_part[0, N-1, M-3:M] = 98

    video_part = gt_part.copy()
    video_part[1, ...] += N*M;

    flow = np.zeros((2, N, M, 2))
    rel = np.ones((2, N, M, 2))
    tracked = bipartite_region_tracking(video_part, flow, rel)

    gt_part[1, N-1, M-2] = 101
    gt_part[1, N-1, M-3] = 100
    gt_part[1, N-1, M-1] = 98

    assert_equal(tracked, gt_part)