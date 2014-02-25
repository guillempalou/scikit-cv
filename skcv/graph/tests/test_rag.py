import numpy as np
from skcv.graph.rag import rag
from numpy.testing import assert_equal

def test_rag():

    #test rag for 2D
    ids = np.linspace(0,1000,100, endpoint=True)
    p_2d = np.reshape(ids, (10, 10)).astype(np.int)

    graph, regions = rag(p_2d)

    assert_equal(len(graph.nodes()), 100)
    assert_equal(len(regions), 100)
    assert_equal(len(graph.edges()), 180)

    graph, regions = rag(p_2d, discard_axis=[0])

    assert_equal(len(graph.edges()), 90)

    #test rag for 3D
    ids = np.linspace(0,1000,1000, endpoint=False)
    p_3d = np.reshape(ids, (10, 10, 10)).astype(np.int)

    graph, regions = rag(p_3d)

    assert_equal(len(graph.nodes()), 1000)
    assert_equal(len(regions), 1000)
    assert_equal(len(graph.edges()), 2700)

    pass