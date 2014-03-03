import numpy as np
from numpy.testing import assert_almost_equal

from skcv.image.segmentation.region_descriptors import region_mean_color
from skcv.image.segmentation.region_descriptors import region_color_histograms
from skcv.image.segmentation.region_descriptors import region_dominant_colors

def test_mean_color():
    img = np.zeros((10,10,3))
    img[5:10,5:10,:] = 2
    region1 = {'coords' : np.where(img[:,:,1] == 2)}
    region2 = {'coords' : np.where(img[:,:,1] == 0)}
    all = {'coords' : np.hstack((region1['coords'],region2['coords']))}
    avg1 = region_mean_color(img,region1)
    avg2 = region_mean_color(img, region2)
    avg_all = region_mean_color(img, all)

    assert_almost_equal(avg1, [2, 2, 2])
    assert_almost_equal(avg2, [0, 0, 0])
    assert_almost_equal(avg_all, [0.5, 0.5, 0.5])

def test_histograms():

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

    k = 0
    for i in range(4):
        region = {"coords": np.where(part == i)}
        h, _ = region_color_histograms(img, region, bins=12)

        for hst in h:
            hist_gt = np.zeros(12)
            hist_gt[k] += 2500
            assert_almost_equal(hst,hist_gt)
            k += 1

def test_dominant_colors():

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

    k = 0
    for i in range(4):
        region = {"coords": np.where(part == i)}
        colors, error = region_dominant_colors(img, region)
        #it will only find one color
        color = img[region['coords']][0,...]
        assert_almost_equal(colors[0], color)
        assert_almost_equal(error, 0)

    # get a whole region covering the whole image
    region = {"coords": np.where(part < 5)}
    colors, error = region_dominant_colors(img, region)

    gt_colors = np.zeros((4,3))
    gt_colors[0, :] = img[np.where(part == 3)][0, ...]
    gt_colors[1, :] = img[np.where(part == 1)][0, ...]
    gt_colors[2, :] = img[np.where(part == 0)][0, ...]
    gt_colors[3, :] = img[np.where(part == 2)][0, ...]
    assert_almost_equal(error, 0)

    pass