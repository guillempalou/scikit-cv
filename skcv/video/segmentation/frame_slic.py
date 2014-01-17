__author__ = 'guillem'

import numpy as np;
from skimage.segmentation import slic

def oversegment_video(video, n_segments, compactness = 10):
    """
    Oversegments a collection of frames using SLIC

    Parameters
    ----------

    video: numpy array (width, height, frames, channels)
        3 or 4 dimensional array representing the video

    n_segments: int
        Number of segments desired for each frame

    compactness: float, optional
        Compactness parameter for the SLIC algorithm

    Returns
    -------
        partition: numpy array (width, height, frames)
            Array representing the partition

    """
    d = len(video.shape)

    width = video.shape[0]
    height = video.shape[1]

    if d == 3:
        video = video[..., np.newaxis]
    elif d != 4:
        raise ValueError('Video should have 3 or 4 dimensions')

    n_frames = video.shape[3]

    partition = np.zeros((width, height, n_frames))
    for n in range(n_frames):
        frame = video[:, :, n, :]
        partition[:, :, n] = slic(frame, n_segments, compactness)

    return partition

