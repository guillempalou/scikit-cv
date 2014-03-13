import numpy as np


def partition_mean_color(img, id_map):
    """ Returns a numpy array of false color

    Parameters
    ---------
    img : array
        image or volume with color values

    id_map : array
        2D or 3D numpy array with id values

    Returns
    -------
    mean_color: array with the same shape than input with 3 values for each position
    """

    ids = np.unique(id_map)
    nids = len(ids)

    # create a mean color image of the original size
    image_mean_color = np.zeros_like(img)

    for label, i in zip(ids, range(nids)):
        coords = np.where(id_map == label)
        coords = [c for c in coords] + [slice(img.shape[-1])]

        mean_color = np.mean(img[coords], axis=0)
        image_mean_color[coords] = mean_color

    # return the false color image
    return image_mean_color

