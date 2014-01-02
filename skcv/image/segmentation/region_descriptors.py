import numpy as np

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
        color/gray image
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
        h,e = np.histogram(values,bins=bins,density=True)
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
    hist: list
        list of ndarrays representing histograms
    edges: list
        list of ndarrays representing the edge bins

    """