import numpy as np
from scipy.cluster.vq import kmeans

def region_mean_color(img, region):
    """ Region mean color

    Parameters
    ----------
    img: numpy array (N,M,D)
        color/gray image
    region: dict
        dictionary containing the coordinates of the region

    Returns
    -------
    avg: numpy 1D vector of D elements
        color mean of the region

    """
    avg = np.mean(img[region["coords"][0], region["coords"][1],:],axis=0)
    return avg

def region_color_histograms(img, region, bins = 10):
    """ Region mean color

    Parameters
    ----------
    img: numpy array (N,M,D)
        color/gray image with [0..1] range for each channel
    region: dict
        dictionary containing the coordinates of the region
    bins: int, optional
        number of bins for each channel

    Returns
    -------
    hist: list
        list of ndarrays representing histograms
    edges: list
        list of ndarrays representing the edge bins

    """
    hist = []
    edges = []
    channels = img.shape[2]

    for i in range(channels):
        values = img[region["coords"][0], region["coords"][1],i]
        h,e = np.histogram(values,bins=bins,range=(0,1))
        hist.append(h)
        edges.append(e)

    return hist,edges

def region_dominant_colors(img, region, colors = 8):
    """ Region mean color

    Parameters
    ----------
    img: numpy array (N,M,D)
        color/gray image
    region: dict
        dictionary containing the coordinates of the region
    bins:

    Returns
    -------
    cb: list
        list of ndarrays representing the centroid
    error: float
        squared error of the clustering process
    """

    values = img[region["coords"][0], region["coords"][1],:]
    cb, error =  kmeans(values,colors)
    return cb, error