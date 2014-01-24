__author__ = 'guillem'

import numpy as np


def false_color(id_map):
    """ Returns a numpy array of false color
    
    Parameters
    ---------
    id_map : 2D or 3D numpy array with id values

    Returns
    -------
    false_color: array with the same shape than input with 3 values for each position
    """

    ids = np.unique(id_map)
    nids = len(ids)

    # assign a random color to each id
    colors = np.random.randint(0, 256, (nids, 3))

    # check dimensions (should be 3 or 4)
    if len(id_map.shape) == 2:
        id_map = id_map[np.newaxis, ...]

    # create a false color image of the original size and 3 channels
    image_false_color = np.zeros((id_map.shape[0],
                                  id_map.shape[1],
                                  id_map.shape[2],
                                  3))

    dimensions = 3
    for label, i in zip(ids, range(nids)):
        (px, py, pz) = np.where(id_map == label)
        image_false_color[px, py, pz, :] = colors[i, :]

    #return the false color image
    return np.squeeze(image_false_color)
