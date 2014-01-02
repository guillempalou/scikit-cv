__author__ = 'guillem'

import math
import numpy as np

#private function for colorwheel
def _colorwheel():
    """ Created the color wheel for flow color. Private, auxiliary function
    Return
    ------
    colorwheel: array of (Ncolors,3) with a set of colors.
    """
    colors  = np.array([15,6,4,11,13,6])
    ncolors = np.sum(colors)

    #array to be returned containing the different colors
    colorwheel = np.zeros((ncolors,3))
    actual_color = 0


    #fill the wheel with colors
    idx = np.arange(actual_color,actual_color+colors[0])
    colorwheel[idx,0] = 255
    colorwheel[idx,1] = np.floor(255*np.linspace(0,1,colors[0],endpoint=False))
    actual_color = actual_color + colors[0]

    idx = np.arange(actual_color,actual_color+colors[1])
    colorwheel[idx,0] = 255 - np.floor(255*np.linspace(0,1,colors[1],endpoint=False))
    colorwheel[idx,1] = 255
    actual_color = actual_color + colors[1]

    idx = np.arange(actual_color,actual_color+colors[2])
    colorwheel[idx,1] = 255
    colorwheel[idx,2] = np.floor(255*np.linspace(0,1,colors[2],endpoint=False))
    actual_color = actual_color + colors[2]

    idx = np.arange(actual_color,actual_color+colors[3])
    colorwheel[idx,1] = 255 - np.floor(255*np.linspace(0,1,colors[3],endpoint=False))
    colorwheel[idx,2] = 255
    actual_color = actual_color + colors[3]

    idx = np.arange(actual_color,actual_color+colors[4])
    colorwheel[idx,0] = np.floor(255*np.linspace(0,1,colors[4],endpoint=False))
    colorwheel[idx,2] = 255
    actual_color = actual_color + colors[4]

    idx = np.arange(actual_color,actual_color+colors[5])
    colorwheel[idx,0] = 255
    colorwheel[idx,2] = 255 - np.floor(255*np.linspace(0,1,colors[5],endpoint=False))

    return colorwheel


def flow_to_image(flow):
    """ Converts a flow array into a RGB image according to middlebury color scheme

    Parameters
    ----------
    flow: flow array (M,N,2) where M and N are the width and height respectively
    """

    if (len(flow.shape) != 3):
        raise ValueError("Flow must be of the form (M,N,2)")

    if (flow.shape[2] != 2):
        raise ValueError("Flow must be of the form (M,N,2)")

    #copy data so we do not change values
    u = np.copy(flow[:,:,0])
    v = np.copy(flow[:,:,1])

    #flow threshold for unknown values
    flow_threshold = 1e9

    #fix unknown values
    idx_unknown = (np.abs(u)> flow_threshold) | (np.abs(v)> flow_threshold)
    idx_nan     = (np.isnan(u) | np.isnan(v));
    u[idx_unknown] = 0
    v[idx_unknown] = 0

    #get flow extreme values
    maxu = u.max()
    maxv = v.max()

    minu = v.min()
    minv = v.min()

    #get the norm of each vector flow
    maxnorm = np.max(u*u + v*v)

    print "Flow range u={hmin} .. {hmax}; v = {vmin} .. {vmax}".format(hmin = minu, hmax = maxu, vmin = minv, vmax = maxv)

    eps = 1e-10
    u = u / (maxnorm + eps)
    v = v / (maxnorm + eps)

    norm = np.sqrt(u*u + v*v)

    #get the color wheel
    colorwheel = _colorwheel()
    ncolors = colorwheel.shape[0]

    #map each angle to a color
    uv_angle = np.arctan2(-v,-u)/math.pi
    fk = (uv_angle+1)*0.5*(ncolors-1)
    k0 = np.floor(fk)
    k0 = k0.astype(np.uint32);
    k1 = k0 + 1
    k1[k1==(ncolors)] = 1
    f = fk - k0

    #compute color for each channel
    flow_img = np.zeros((flow.shape[0],flow.shape[1],3))
    #get indexes with valid flow values
    idx = (norm <= 1)

    for i in [0,1,2]:
        t = colorwheel[:,i]
        col0 = t[k0]*1.0/255
        col1 = t[k1]*1.0/255
        col = (1-f)*col0 + f*col1

        col[idx] = 1 - np.exp(-norm[idx])*(1-col[idx]);
        col[~idx] = col[~idx]*0.75

        flow_img[:,:,i] = np.floor(255*col*(1-idx_nan))

    return flow_img