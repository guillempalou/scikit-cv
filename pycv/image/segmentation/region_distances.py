__author__ = 'guillem'

from region_descriptors import region_mean_color
import numpy as np;

def mean_color_distance(img,r1,r2):
    """ Returns the color mean of two regions
    @param r1: Region 1
    @param r2: Region 2
    @return:euclidean distance between region color means
    """
    if ("mean_color" not in r1):
        r1["mean_color"] = region_mean_color(img,r1)

    if ("mean_color" not in r2):
        r2["mean_color"] = region_mean_color(img,r2)

    return np.linalg.norm(r1["mean_color"]-r2["mean_color"])