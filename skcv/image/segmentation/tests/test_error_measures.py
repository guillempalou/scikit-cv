import numpy as np
from numpy.testing import assert_equal
from skcv.image.segmentation.error_measures import undersegmentation_error

def test_undersegmentation_error():
    N = 100
    M = 100

    part = np.zeros((N, M))

    part[:N/2, :M/2] = 0
    part[N/2:, :M/2] = 1
    part[:N/2, M/2:] = 2
    part[N/2:, M/2:] = 3

    part_gt = part.copy()

    part[N/2, M/4] = 0
    part[N/4-1, M/2-1] = 2

    ue = undersegmentation_error(part, part_gt)

    assert_equal(ue,0.0002)

    ue = undersegmentation_error(part, part_gt, tolerance=0)

    assert_equal(ue,0.5001)