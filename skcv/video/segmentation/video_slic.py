import numpy as np
from skimage.segmentation import slic


def video_slic(video, n_segments, compactness=10):
    """
    Oversegments a collection of frames using SLIC

    Parameters
    ----------

    video: numpy array
        3 or 4 dimensional array representing the video, in CIE LAB

    n_segments: int
        Number of segments desired for each frame

    compactness: float, optional
        Compactness parameter for the SLIC algorithm

    Returns
    -------
        partition: numpy array
            Array representing the partition

    """
    d = len(video.shape)

    width = video.shape[1]
    height = video.shape[2]

    if d == 3:  # pragma: no cover
        video = video[..., np.newaxis]
    elif d != 4:  # pragma: no cover
        raise ValueError('Video should have 3 or 4 dimensions')

    n_frames = video.shape[0]

    partition = np.zeros((n_frames, width, height))
    current_label = 0
    for n in range(n_frames):
        frame = video[n, ...]
        partition[n, ...] = current_label + slic(frame, n_segments, compactness,
                                                 convert2lab=False,
                                                 enforce_connectivity=True)
        current_label = np.max(partition[n, ...]) + 1

    return partition

