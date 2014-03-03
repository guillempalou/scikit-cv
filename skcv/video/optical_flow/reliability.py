__author__ = 'guillem'

import numpy as np
from scipy.interpolate import griddata
from skimage import filter

import matplotlib.pyplot as plt


def variation_reliability(flow, gamma=1):
    """ Calculates the flow variation reliability
    Parameters
    ----------
    flow: numpy array
    flow values

    gamma: float, optional
    soft threshold

    Returns
    -------
    variation reliability map (0 less reliable, 1 reliable)
    """

    #compute central differences
    gradx = np.gradient(flow[:, :, 0])
    grady = np.gradient(flow[:, :, 1])

    norm_grad = (gradx[0] ** 2 + gradx[1] ** 2 +
                 grady[0] ** 2 + grady[1] ** 2) / (0.01 * np.sum(flow ** 2, axis=2) + 0.002)

    norm_grad[norm_grad > 1e2] = 0

    return np.exp(-norm_grad / gamma)


def occlusion_reliability(forward_flow, backward_flow, gamma=1):
    """ Calculates the flow variation reliability
    Parameters
    ----------
    forward_flow: numpy array with forward flow values
    backward_flow: numpy array with backward flow values
    gamma: soft threshold

    Return
    ------
    variation_reliability: reliability map (0 less reliable, 1 reliable)
    """

    #check dimensions
    if (forward_flow.shape != backward_flow.shape): #pragma: no cover
        raise ValueError("Array sizes should be the same")

    #compute warping flow
    xcoords = np.arange(0, forward_flow.shape[0])
    ycoords = np.arange(0, forward_flow.shape[1])
    xx, yy = np.meshgrid(ycoords, xcoords)
    coords = (xx.flatten(), yy.flatten())

    #find the warped flow
    warped_flow = np.zeros_like(forward_flow)
    warped_flow[:, :, 0] = xx + forward_flow[:, :, 0]
    warped_flow[:, :, 1] = yy + forward_flow[:, :, 1]
    warped_coords = (warped_flow[:, :, 0].flatten(), warped_flow[:, :, 1].flatten())

    #interpolate flow values
    fx = griddata(coords, backward_flow[:, :, 0].flatten(), warped_coords, method='linear', fill_value=0)
    fy = griddata(coords, backward_flow[:, :, 1].flatten(), warped_coords, method='linear', fill_value=0)

    interpolated_flow = np.zeros_like(forward_flow)
    interpolated_flow[:, :, 0] = fx.reshape(backward_flow.shape[:2])
    interpolated_flow[:, :, 1] = fy.reshape(backward_flow.shape[:2])

    #find the forward-backward consistency
    result = np.sum((forward_flow + interpolated_flow) ** 2, axis=2) / \
             (0.01 * (np.sum(forward_flow ** 2, axis=2) +
                      np.sum(interpolated_flow ** 2, axis=2)) + 0.5)

    return np.exp(-result / gamma)


def structure_reliability(img, gamma=1):
    """ Calculates the flow structure reliability
    Parameters
    ----------
    img: numpy array
    image to compute the structure

    gamma: float, optional
    soft threshold

    Return
    ------
    reliability map (0 less reliable, 1 reliable)

    """

    #compute gradient of the image in the three channels

    #kernel for blurring

    st = np.zeros((img.shape[0], img.shape[1]))

    eps = 1e-6

    for k in np.arange(img.shape[-1]):
        grad = np.gradient(img[:, :, k])
        #compute components of the structure tensor
        wxx = filter.gaussian_filter(grad[0] ** 2, 1)
        wxy = filter.gaussian_filter(grad[0] * grad[1], 1)
        wyy = filter.gaussian_filter(grad[1] ** 2, 1)

        #determinant and trace
        wdet = wxx * wyy - wxy ** 2
        wtr = wxx + wyy

        st += wdet / (wtr + eps)

    avg = st.mean()

    return 1 - np.exp(-st / (0.7 * avg * gamma))


def flow_reliability(img, forward_flow, backward_flow, use_structure=True):
    """

    Parameters
    ----------
    @param img: image frame
    @param forward_flow: flow from the current frame to the other
    @param backward_flow: flow from the next frame and the current
    @param use_structure: use structure to compute the minimum

    Returns
    -------
    @return: the minimum of the different reliabilities
    """

    #soft threshold
    gamma = 1

    if use_structure:
        st = structure_reliability(img, gamma)
    else:
        st = np.ones((img.shape[0], img.shape[1]))

    #compute the different reliabilities
    var = variation_reliability(forward_flow, gamma)
    occ = occlusion_reliability(forward_flow, backward_flow)

    #return the minimum of the three
    return np.minimum(st, np.minimum(var, occ))
