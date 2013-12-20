__author__ = 'guillem'

import numpy as np


def read_flow_file(path):
    """ Reads flow file and returns 2D numpy array
    Parameters
    ----------
    path: file path to read

    Return
    ------
    flow: a (M,N,2) numpy array containing the 2D flow vectors for each position (x,y)
    """
    #open the file
    f = open(path, "rb")

    if (not f):
        raise IOError("File cannot be opened")

    #read the tag
    tag = f.read(4)

    if (tag != "PIEH"):
        raise TypeError("File type does not correspond to a flow file")

    #read the width and height
    width = np.fromfile(f, dtype=np.uint32, count=1)
    height = np.fromfile(f, dtype=np.uint32, count=1)

    if (width < 1 or width > 99999 or height < 1 or height > 99999):
        raise ValueError("Width and height file not correct")

    #read flow data
    flow = np.fromfile(f, dtype=np.float32, count=width[0] * height[0] * 2)

    if (flow.size != width[0] * height[0] * 2):
        raise ValueError("Data flow too small %d != %d" % (flow.size, width[0] * height[0] * 2))

    #reshape the flow so that its shape is (height,width,2)
    flow_reshaped = np.reshape(flow, (height[0], width[0], 2), order='C')

    #close the file
    f.close()

    return flow_reshaped


def write_flow_file(path, flow):
    """ Writes flow file and returns 2D numpy array
    Parameters
    ----------
    path: file path to write
    flow:

    """

    #open the file for writing
    f = open(path, "wb")

    if (not f):
        raise IOError("File cannot be opened")

    #read the tag
    tag = f.write("PIEH")


    #write first the width and then the height
    shape = np.array((2,1),dtype=np.uint32)
    shape[0] = flow.shape[1]
    shape[1] = flow.shape[0]
    shape.tofile(f)

    #write the flow data
    flow.astype(np.float32).tofile(f)