import os
import numpy as np
from numpy.testing import assert_equal, assert_almost_equal
from skcv import data_dir
from skcv.video.optical_flow.io import *


def test_io():
    N = 11
    M = 11

    x_flow = np.linspace(0, N, N, endpoint=False)
    y_flow = np.linspace(0, M, M, endpoint=False)

    xv, yv = np.meshgrid(x_flow, y_flow)

    flow = np.zeros((N, M, 2))
    flow[..., 0] = xv - N/2
    flow[..., 1] = yv - M/2

    test_file = os.path.join(data_dir, 'flow_test.flo')

    write_flow_file(test_file, flow)

    flow_read = read_flow_file(test_file)

    assert_equal(flow, flow_read)