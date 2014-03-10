import numpy as np
from skcv.multiview.util.points_functions import *
#from skcv.multiview.two_views.fundamental_matrix import *

def triangulate_from_projections(x, cameras):
    xn, t = normalize_points(x, is_homogeneous=True)

    x_3d = _triangulate_cython(xn, cameras)


def find_projection_matrix_from_points(x, x_3d):
    pass