import numpy as np
from pycv.image.segmentation.region_descriptors import region_mean_color

def test_mean_color():
    img = np.zeros((10,10,3))
    img[5:10,5:10,:] = 2
    region1 = {'coords' : np.where(img[:,:,1] == 2)}
    region2 = {'coords' : np.where(img[:,:,1] == 0)}
    all = {'coords' : np.hstack((region1['coords'],region2['coords']))}
    avg1 = region_mean_color(img,region1)
    avg2 = region_mean_color(img, region2)
    avg_all = region_mean_color(img, all)
    for i in range(3):
        assert(avg1[i] == 2)
        assert(avg2[i] == 0)
        assert(avg_all[i] == 0.5)

