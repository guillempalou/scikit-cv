import numpy as np
import pickle
import os

from numpy.testing import assert_array_almost_equal

from skcv import data_dir
from skcv.multiview.two_views.fundamental_matrix import *
from skcv.multiview.util.points_functions import *
from skcv.multiview.util.camera import *
from skcv.multiview.two_views import triangulation

def test_triangulation_hartley():

    projections_file = os.path.join(data_dir, 'two_view_projections.dat')

    (x1e, x2e) = pickle.load(open(projections_file, 'rb'))

    #add gaussian noise to x1e and x2e
    dev = 0.1
    x1e += np.random.normal(0, dev, size=x1e.shape)
    x2e += np.random.normal(0, dev, size=x2e.shape)

    x1h = euclidean_to_homogeneous(x1e)
    x2h = euclidean_to_homogeneous(x2e)

    f_matrix = robust_f_estimation(x1h, x2h)

    p1, p2 = canonical_cameras_from_f(f_matrix)

    X = triangulation.optimal_triangulation(x1h, x2h, f_matrix, cameras=(p1,p2), method='Hartley')

    x1p = np.dot(p1, X)
    x2p = np.dot(p2, X)

    ratio1 = x1p / x1h
    ratio2 = x2p / x2h


def test_triangulation_kanatani():

    projections_file = os.path.join(data_dir, 'two_view_projections.dat')

    (x1e, x2e) = pickle.load(open(projections_file, 'rb'))

    #add gaussian noise to x1e and x2e
    dev = 0.1
    x1e += np.random.normal(0, dev, size=x1e.shape)
    x2e += np.random.normal(0, dev, size=x2e.shape)

    x1h = euclidean_to_homogeneous(x1e)
    x2h = euclidean_to_homogeneous(x2e)

    f_matrix = robust_f_estimation(x1h, x2h)

    p1, p2 = canonical_cameras_from_f(f_matrix)

    X = triangulation.optimal_triangulation(x1h, x2h, f_matrix, cameras=(p1,p2), method='Kanatani')

    x1p = np.dot(p1, X)
    x2p = np.dot(p2, X)

    ratio1 = x1p / x1h
    ratio2 = x2p / x2h
