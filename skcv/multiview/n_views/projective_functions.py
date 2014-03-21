import numpy as np

def swap_signs(cameras, x3d):
    """
    Swaps signs of the camera and 3D points
    so the projective depths are positive

    Parameters
    ----------
    camera: list
        Camera matrices

    x3d: numpy array
        array containing 3D points


    Returns
    -------
    camera: cameras with the correct sign. empty if error
    x3d: points with the correct sign. empty if error

    """

    n_views = len(cameras)
    n_points = x3d.shape[1]

    signs = np.zeros((n_views, n_points))

    for i in range(n_views):
        signs[i, :] = np.sign(np.dot(cameras[i], x3d))[2, :]

    signp = signs[:, 0]
    signs *= signp

    signx = signs[0, :]
    signs *= signx

    if np.any(signs < 0):
        return [], []

    x3d_signed = x3d * signx
    cameras_signed = [cameras[i]*signp[i] for i in range(n_views)]

    return cameras_signed, x3d_signed