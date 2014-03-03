import heapq

import numpy as np

from skcv.graph import rag


class TBPT:
    """
    Class defining a hierarchical segmentation
    The hierarchy is created from an initial partition
    and merging the two most similar regions according
    to a predefined distance until only one region is
    left
    """

    def __init__(self, video, partition, distance,
                 optical_flow=None, update_partition=False, verbose=0):
        """ Creates the Trajectory Binary Partition Tree
            from the initial partition using a specified
            distance

            Parameters
            ----------

            video: array (width, height, n_frames, channels)
                Array containing the video
            partition: array (width, height, n_frames)
                Array with labels used as the initial
                partition
            optical_flow: array (width, height, n_frames, 2), optional
                Array containing the optical flow
            use_flow_for_adjacency: bool, optional
                Whether temporal adjacency uses flow or not
            distance: function (img,region1,region2)
                Distance function between two regions
            update_partition: bool, optional
                Whether the partition gets updated
            verbose: int, optional
                Indicates the level of verbosity
        """

        #initial rag

        r, regions = rag(partition, discard_axis=[0])

        #structures to save the tree topology
        self.nodes = {}
        for reg in regions:
            self.nodes[reg] = {}

        #compute initial distances
        dists = []
        max_label = 0
        for e in r.edges_iter():
            if optical_flow is not None:
                dists.append((distance(video, optical_flow,
                                       regions[e[0]], regions[e[1]]),
                              e[0], e[1]))
            else:
                dists.append((distance(video,
                                       regions[e[0]], regions[e[1]]),
                              e[0], e[1]))

            #store the nodes to a structure
            self.nodes[e[0]]["childs"] = []
            self.nodes[e[1]]["childs"] = []

            #get the maximum used label
            max_label = max(max_label, e[0], e[1])

        #make a heap (priority queue)
        heapq.heapify(dists)

        #contains the regions that are merged
        merged = set()

        #number of regions, N-1 merges
        n_regions = len(regions)
        max_label += 1

        if verbose > 0:  # pragma : no cover
            print("Performing {0} merges".format(n_regions - 1, max_label))

        for n in range(n_regions - 1):

            to_merge = heapq.heappop(dists)

            while (to_merge[1] in merged) or (to_merge[2] in merged):
                to_merge = heapq.heappop(dists)

            if verbose > 1:  # pragma : no cover
                print("Merging {0} and {1} to {2} with distance ".format(
                    to_merge[1],
                    to_merge[2],
                    max_label,
                    to_merge[0]))

            coords1 = regions[to_merge[1]]["coords"]
            coords2 = regions[to_merge[2]]["coords"]

            #create the new region and add a node to the rag
            coords_parent = [np.hstack((coords1[0], coords2[0])),
                             np.hstack((coords1[1], coords2[1])),
                             np.hstack((coords1[2], coords2[2]))]

            regions[max_label] = {"label": max_label, "coords": coords_parent}
            r.add_node(max_label)
            self.nodes[max_label] = {}

            #update tree structures
            self.nodes[to_merge[1]]["parent"] = max_label
            self.nodes[to_merge[2]]["parent"] = max_label
            self.nodes[max_label]["childs"] = (to_merge[1], to_merge[2])

            #iterate through the neighbors of the childs and update links
            edges = r.edges([to_merge[1], to_merge[2]])
            for e in edges:
                r.add_edge(max_label, e[1])
                if optical_flow is not None:
                    dist = distance(video, optical_flow,
                                    regions[e[0]], regions[e[1]])
                else:
                    dist = distance(video,
                                    regions[e[0]], regions[e[1]])

                heapq.heappush(dists, (dist, max_label, e[1]))

            #remove the two nodes and edges
            r.remove_edges_from(edges)
            r.remove_node(to_merge[1])
            r.remove_node(to_merge[2])

            #add the two regions to the set of merged
            merged.add(to_merge[1])
            merged.add(to_merge[2])

            #print(coords_parent,coords_parent.shape)
            if update_partition:
                partition[coords_parent[0][:],
                          coords_parent[1][:],
                          coords_parent[2][:]] = max_label

            max_label += 1
