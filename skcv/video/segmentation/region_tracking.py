import networkx as nx
import numpy as np

def bipartite_region_tracking(partition, optical_flow, reliability,
                    matching_th=0.1, reliability_th=0.2):
    """
    Parameters
    ----------
    partition: numpy array
    A 3D label array where each label represents a region

    optical_flow: numpy array
    A 3D,2 array representing optical flow values for each frame

    reliability: numpy array
    A 3D array representing the flow reliability

    matching_th: float, optional
    matching threshold for the bipartite matching

    reliability_th: float, optional
    reliability threshold to stop tracking

    Returns
    -------
    A NetworkX graph object with adjacency relations

    """

    dimensions = len(partition.shape)

    if dimensions != 3: #prama: no cover
        raise ValueError("Dimensions must be 3")

    # link regions across frames
    # perform a weighted bipartite matchings

    frames = partition.shape[0]
    width = partition.shape[1]
    height = partition.shape[2]

    new_partition = np.zeros_like(partition)

    #the first frame is the same
    new_partition[0,...] = partition[0,...]
    current_label = np.max(np.unique(partition[0,...]))+1

    for frame in range(frames-1):
        labels = np.unique(new_partition[frame, ...])
        labels_next = np.unique(partition[frame+1, ...])
        # create a graph matching contours
        bipartite = nx.Graph()
        bipartite.add_nodes_from([l for l in labels])
        bipartite.add_nodes_from([l for l in labels_next])

        # find the correspondence of each label to the next frame
        for label in labels:
            px, py = np.where(new_partition[frame, ...] == label)

            # find the mean reliability
            rel = np.mean(reliability[frame, px, py])

            if rel < reliability_th:
                continue

            # find where the regions projects to the next frame
            npx = px + optical_flow[frame, px, py, 0]
            npy = py + optical_flow[frame, px, py, 1]

            #check for bounds
            in_x = np.logical_and(0 <= npx, npx < width)
            in_y = np.logical_and(0 <= npy, npy < height)
            idx = np.logical_and(in_x, in_y)

            npx = npx[idx]
            npy = npy[idx]

            count = np.bincount(partition[frame+1,
                                          npx.astype(np.int),
                                          npy.astype(np.int)].astype(np.int))

            # get the count and eliminate weak correspondences
            max_count = max(count)
            nodes = np.nonzero(count > max_count*matching_th)[0]

            weight = count[nodes]/max_count
            for i, n in enumerate(nodes):
                bipartite.add_edge(label, n, weight=weight[i])

        # max weighted matching
        matchings = nx.max_weight_matching(bipartite)

        # assign propagated labels to the matchings
        for a in matchings:
            b = matchings[a]
            #print("Match {0}-{1}".format(a,b))
            if b not in labels_next:
                continue

            px, py = np.where(partition[frame+1, ...] == b)
            new_partition[frame+1, px, py] = a

        # assign new labels to non-matched regions
        for n in bipartite.nodes():
            if n not in labels_next:
                continue
            if n not in matchings:
                px, py = np.where(partition[frame+1, ...] == n)
                new_partition[frame+1, px, py] = current_label + 1
                current_label += 1

    return new_partition