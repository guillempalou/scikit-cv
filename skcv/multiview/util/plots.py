import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plot_point_cloud(x):  # pragma: no cover
    """
    Plots point cloud as a scattered 3D plot
    Cannot be tested in a terminal-only framework
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x[0, :], x[1, :], x[2, :], '*')
    plt.show()