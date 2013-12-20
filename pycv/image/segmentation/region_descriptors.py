__author__ = 'guillem'

import numpy as np

def region_mean_color(img, region):
    avg = np.mean(img[region["coords"][0], region["coords"][1],:],axis=0)