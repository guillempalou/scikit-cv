import numpy as np
from numpy.testing import assert_allclose
import pickle
import os
import matplotlib.pyplot as plt

from skcv import data_dir
from skcv.multiview.autocalibration import linear_autocalibration
from skcv.multiview.n_views import projective_factorization
from skcv.multiview.util.points_functions import *
from skcv.multiview.util.camera import *


def test_linear_autocalibration():

    dump_path = os.path.join(data_dir, "multiview_projections.dat")
    dump_file = open(dump_path, "rb")

    n_views = pickle.load(dump_file)

    internals = []
    rotations = []
    centers = []
    gt_cameras = []
    depths = []

    for i in range(n_views):
        (k, r, c) = pickle.load(dump_file)
        internals.append(k)
        rotations.append(r)
        centers.append(c)

        cm = np.eye(3, 4)
        cm[:, 3] = -c
        cm = np.dot(k, np.dot(r, cm))
        gt_cameras.append(cm)

    (projections, x3d) = pickle.load(dump_file)

    x3dh = euclidean_to_homogeneous(x3d)

    #get the true depths
    for i in range(n_views):
        projs = np.dot(gt_cameras[i], x3dh)
        depth_i = (projections[i] / projs)[1,:] / np.linalg.norm(gt_cameras[i][2,:])
        depths.append(depth_i)

    # get a random homography to generate a projective reconstruction
    h = np.random.random((4, 4))
    x_3d = np.dot(np.linalg.inv(h), x3dh)
    cameras = []

    for i in range(n_views):
        cameras.append(np.dot(gt_cameras[i], h))

    t = linear_autocalibration(cameras, internals[0])

    for i in range(n_views):
        p = np.dot(cameras[i], t)
        k, r, c = camera_parameters(p)
        k /= k[2, 2]

        #check the calibration got correct internal parameters
        assert_allclose(k, internals[i], rtol=1e-2, atol=1e-6)

    dump_file.close()
