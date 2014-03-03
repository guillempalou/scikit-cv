import numpy as np
from numpy.testing import assert_equal
from skcv.video.optical_flow.reliability import flow_reliability
from skcv.video.segmentation.tbpt import TBPT


def test_tbpt():
    N = 99
    M = 99
    n_frames = 2

    fflow = np.zeros((n_frames, N, M, 2))
    bflow = np.zeros((n_frames, N, M, 2))

    fflow[0, N / 3:2 * N / 3, M / 3:2 * M / 3, 0] = 1
    fflow[0, N / 3:2 * N / 3, M / 3:2 * M / 3, 1] = 1

    bflow[1, 1 + N / 3:2 * N / 3, 1 + M / 3:2 * M / 3, 0] = -1
    bflow[1, 1 + N / 3:2 * N / 3, 1 + M / 3:2 * M / 3, 1] = -1

    video = np.zeros((2, N, M, 3))

    fcoords = np.where(fflow[0, ..., 0] == 1)
    bcoords = np.where(bflow[1, ..., 0] == -1)

    video[0, fcoords[0], fcoords[1], :] = 200
    video[1, bcoords[0], bcoords[1], :] = 200

    rel = np.zeros((n_frames, N, M))
    for frame in range(n_frames-1):
        rel[frame, ...] = flow_reliability(video[frame, ...],
                                           fflow[frame, ...],
                                           bflow[frame + 1, ...],
                                           use_structure=False)

    part = (video[..., 1] != 0).astype(np.int)

    #define a distance for the TBPT
    #arguments: video, flow, region1, region2
    distance = lambda v, fflow, r1, r2: 1

    tbpt = TBPT(video, part, distance, optical_flow=fflow)

    #check regions
    assert_equal(tbpt.nodes[0]["parent"], 2)
    assert_equal(tbpt.nodes[1]["parent"], 2)
    assert_equal(tbpt.nodes[2]["childs"], [0, 1])