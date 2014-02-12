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
    # construct adavanced indexing
    coords = [c for c in region["coords"]] + [slice(img.shape[-1])]
    avg = np.mean(img[coords], axis=0)
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
    channels = img.shape[-1]

    # construct adavanced indexing
    coords = [c for c in region["coords"]]

    for i in range(channels):
        c = coords + [i]
        values = img[c]
        h,e = np.histogram(values, bins=bins, range=(0, 1))
        hist.append(h)
        edges.append(e)

    return hist, edges

def region_dominant_colors(img, region, colors=8):
    """ Region mean color

    Parameters
    ----------
    img: numpy array (N,M,D)
        color/gray image
    region: dict
        dictionary containing the coordinates of the region
    color: int, optional
        number of color clusters

    Returns
    -------
    cb: list
        list of ndarrays representing the centroid
    error: float
        squared error of the clustering process
    """
     # construct adavanced indexing
    coords = [c for c in region["coords"]] + [slice(img.shape[-1])]

    values = img[coords]
    cb, error = kmeans(values, colors)
    return cb, error