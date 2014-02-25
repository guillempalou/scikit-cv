import numpy as np
from numpy.testing import assert_equal
from skcv.image.segmentation.error_measures import undersegmentation_error
from skcv.image.segmentation.error_measures import boundary_detection
from skcv.image.segmentation.error_measures import segmentation_accuracy
from skcv.image.segmentation.error_measures import explained_variation

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

    assert_equal(ue, 0.0002)

    # test with lists
    ue = undersegmentation_error(part, [part_gt])

    assert_equal(ue, 0.0002)

    ue = undersegmentation_error(part, part_gt, tolerance=0)

    assert_equal(ue, 0.5001)


def test_segmentation_accuracy():
    N = 100
    M = 100

    part = np.zeros((N, M))

    part[:N/2, :M/2] = 0
    part[N/2:, :M/2] = 1
    part[:N/2, M/2:] = 2
    part[N/2:, M/2:] = 3

    part_gt = part.copy()

    accu = segmentation_accuracy(part, part_gt)
    assert_equal(accu, 1)

    part[N/2, M/4] = 0
    part[N/4-1, M/2-1] = 2

    accu = segmentation_accuracy(part, part_gt)

    assert_equal(accu, 0.99980000000000002)

    # test with lists
    accu = segmentation_accuracy(part, [part_gt])

    assert_equal(accu, 0.99980000000000002)


def test_boundary_detection():
    n = 10
    m = 10

    part = np.zeros((n, m))

    part[:n/2, :m/2] = 0
    part[n/2:, :m/2] = 1
    part[:n/2, m/2:] = 2
    part[n/2:, m/2:] = 3

    part_gt = part.copy()

    part[n/2, m/4] = 0
    part[n/4-1, m/2-1] = 2

    precision, recall = boundary_detection(part, part_gt, 0)

    assert_equal(precision, 0.75)
    assert_equal(recall, 0.9)

def test_explained_variation():
    n = 10
    m = 10

    part = np.zeros((n, m))

    part[:n/2, :m/2] = 0
    part[n/2:, :m/2] = 1
    part[:n/2, m/2:] = 2
    part[n/2:, m/2:] = 3

    img = np.fromfunction(lambda i, j, k: i+j, (n, m, 3), dtype=int)

    ev = explained_variation(img, part)
    assert_equal(ev, 0.24242424242424243)

    img[:, :, 1] = part
    img[:, :, 2] = part
    img[:, :, 3] = part

    ev = explained_variation(img, part)
    assert_equal(ev, 0)