import numpy as np
from numpy.testing import assert_almost_equal
import pickle
import os
from skcv import data_dir
from skcv.multiview.two_views.fundamental_matrix import *
from skcv.multiview.util.camera import *

def test_eight_point_algorithm():
    projections_file = os.path.join(data_dir, 'two_view_projections.dat')

    (x1e, x2e) = pickle.load(open(projections_file))

    x1h = euclidean_to_homogeneous(x1e)
    x2h = euclidean_to_homogeneous(x2e)

    f_matrix = eight_point_algorithm(x1h, x2h)

    f_groundtruth = np.array(((0, 0, 0),
                              (0, 0, -1),
                              (0, 1, 0)))

    f_matrix /= np.max(f_matrix)

    assert_almost_equal(f_matrix, f_groundtruth)