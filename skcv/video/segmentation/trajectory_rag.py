import networkx as nx
import numpy as np

def trajectory_rag(partition, optical_flow, reliability, consistent=False,
                   matching_th = 0.1, reliability_th=0.3):
    """
    Parameters
    ----------
    partition: numpy array
    A 3D label array where each label represents a region

    optical_flow: numpy array
    A 3D,2 array representing optical flow values for each frame

    reliability: numpy array
    A 3D array representing the flow reliability

    consistent: bool, optional
    whether the original partition is consistent from frame to frame

    matching_th: float, optional
    matching threshold for the bipartite matching

    reliability_th: float, optional
    reliability threshold to stop tracking

    Returns
    -------
    A NetworkX graph object with adjacency relations

    Raises
    ------
    ValueError:
        if partition has dimension 1

    Notes
    -----
    The regions correspond to labels, not connected components

    Examples
    --------

    """

    dimensions = len(partition.shape)

    if dimensions != 3:
        raise ValueError("Dimensions must be 3")

    # create a RAG
    rag = nx.Graph()

    # link regions across frames if consistent=False
    # perform a weighted bipartite matchings
    if not consistent:
        frames = partition.shape[0]

        for frame in range(frames-1):

            labels = np.unique(partition[frame, ...])
            labels_next = np.unique(partition[frame+1, ...])

            max_label = max(labels)+1

            # create a graph matching contours
            bipartite = nx.Graph()
            bipartite.add_nodes_from([l for l in labels])
            bipartite.add_nodes_from([l+max_label for l in labels_next])

            # find the correspondence of each label to the next frame
            for label in labels:
                px, py = np.where(partition[frame, ...] == label)

                # find the mean reliability
                rel = np.mean(reliability[frame, px, py])
                if rel < reliability_th:
                    continue

                # find where the regions projects to the next frame
                npx = px + optical_flow[frame, px, py, 0]
                npy = py + optical_flow[frame, px, py, 1]

                count = np.bincount(partition[frame+1,
                                              npx.astype(np.int),
                                              npy.astype(np.int)].astype(np.int))

                # get the count and eliminate weak correspondences
                sum_count = np.sum(count)
                nodes = np.nonzero(count > sum_count*matching_th)[0]

                # update the correspondence values

                sum_count = np.sum(count[nodes])

                weight = count[nodes]/sum_count
                for i, n in enumerate(nodes):
                    print("adding {0}-{1} weight {2}".format(label, n+max_label, weight[i]))
                    bipartite.add_edge(label, n+max_label, weight=weight[i])

            matchings = nx.max_weight_matching(bipartite)

            #TODO assign labels of the matchings
            
    labels = np.unique(partition)

    #create a regions hash table organized by label
    regions = {}
    for label in labels:
        px, py, pz = np.where(partition == label)
        coords = [px, py, pz]
        regions[label] = {"label": label, "coords": coords}

    #create spatial neighbors


    #we need to link frame to frame regions

    #create nodes for the RAG
    rag.add_nodes_from(labels)

    #get adjacencies in each dimension
    endx = []
    startx = []

    #list containing all tuples
    pairs = []
    for d in range(2):
        if d == 0:
            idx = np.where(partition[:-1, :, :] != partition[1:, :, :])
        elif d == 1:
            idx = np.where(partition[:, :-1, :] != partition[:, 1:, :])
        incx = int(d == 0)
        incy = int(d == 1)
        adj = (partition[idx[0], idx[1], idx[2]],
               partition[idx[0] + incx, idx[1] + incy, idx[2]])
        pairs = pairs + \
                [(min(adj[0][i], adj[1][i]),
                  max(adj[0][i], adj[1][i])) for i in range(len(adj[0]))]

    #find unique region pairs
    unique_pairs = set(pairs)

    # compute distances between regions
    edges = [(r[0], r[1]) for r in unique_pairs]

    rag.add_edges_from(edges)

    #return the rag, the regions dictionary and ordered distances
    return rag, regions
