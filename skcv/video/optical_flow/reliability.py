__author__ = 'guillem'

import numpy as np
from scipy.interpolate import griddata
from skimage import filter

def variation_reliability(flow, gamma=1):
    """ Calculates the flow variation reliability
    Parameters
    ----------
    flow: numpy array with flow values
    gamma: soft threshold

    Returns
    -------
    variation_reliability: reliability map (0 less reliable, 1 reliable)
    """

    #compute central differences
    grad = np.gradient(flow[:,:,0])

    norm_grad = grad[0]**2 + grad[1]**2 / (0.01*np.sum(flow**2,axis=2)+0.002)

    norm_grad[norm_grad > 1e2] = 0

    return np.exp(-norm_grad/gamma)

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
    if (forward_flow.shape != backward_flow.shape):
        raise ValueError("Array sizes should be the same")

    #compute warping flow
    xcoords = np.arange(0,forward_flow.shape[0])
    ycoords = np.arange(0,forward_flow.shape[1])
    xx,yy = np.meshgrid(xcoords,ycoords)
    coords = (xx.flatten(), yy.flatten())

    #find the warped flow
    warped_flow = np.zeros(forward_flow.shape)
    warped_flow[:,:,0] = xx + forward_flow[:,:,0]
    warped_flow[:,:,1] = yy + forward_flow[:,:,0] 
    warped_coords = (warped_flow[:,:,0].flatten(), warped_flow[:,:,1].flatten())

    #interpolate flow values
    fx = griddata(coords, backward_flow[:,:,0].flatten(), warped_coords, method='linear',fill_value=0)
    fy = griddata(coords, backward_flow[:,:,1].flatten(), warped_coords, method='linear',fill_value=0)
    interpolated_flow = np.zeros(forward_flow.shape)
    interpolated_flow[:,:,0] = fx.reshape(10,10)
    interpolated_flow[:,:,1] = fy.reshape(10,10)

    #find the forward-backward consistency
    result = np.sum((forward_flow + interpolated_flow)**2,axis=2) / (0.01*(np.sum(forward_flow**2,axis=2) + np.sum(interpolated_flow**2,axis=2)) + 0.5)

    return np.exp(-result/gamma)


def structure_reliability(img, gamma=1):
    """ Calculates the flow structure reliability
    Parameters
    ----------
    two_frames: numpy array containing the current and next frame
    flow: numpy array with flow values
    gamma: soft threshold

    Return
    ------
    variation_reliability: reliability map (0 less reliable, 1 reliable)
    """

    #compute gradient of the image in the three channels

    #kernel for blurring

    structure_reliability = np.zeros((img.shape[0],img.shape[1]))

    for k in np.arange(img.shape[2]):
        grad = np.gradient(img[:,:,k])
        #compute components of the structure tensor
        Wxx = filter.gaussian_filter(grad[0]**2,1)
        Wxy = filter.gaussian_filter(grad[0]*grad[1],1)
        Wyy = filter.gaussian_filter(grad[1]**2,1)

        #determinant and trace
        Wdet = Wxx*Wyy - Wxy**2
        Wtr = Wxx + Wyy

        structure_reliability = structure_reliability + Wdet/Wtr

    avg = structure_reliability.mean();

    return 1-np.exp(-(structure_reliability)/(0.7*avg*gamma))

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

    if (use_structure):
        structure_reliability = structure_reliability(img, gamma)
    else:
        structure_reliability = np.ones((img.shape[0],img.shape[1]))

    #compute the different reliabilities
    var = variation_reliability(forward_flow,gamma)
    occ = occlusion_reliability(forward_flow, backward_flow)

    #return the minimum of the three
    return np.mininum(structure_reliability,np.minimum(var,occ))
