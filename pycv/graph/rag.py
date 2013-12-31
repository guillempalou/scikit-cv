__author__ = 'guillem'

import networkx as nx
import numpy as np

def rag(partition):
        """
        Parameters
        ----------
        partition: numpy array
        A 2D or 3D label array where each label represents a region

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

        if (dimensions == 2):
            partition = partition[:, :, np.newaxis]


        #create a RAG
        rag = nx.Graph()
        labels = np.unique(partition)

        #create a regions hastable organized by label
        regions = {}
        for label in labels:
            px,py,pz = np.where(partition == label)
            if (dimensions == 2):
                coords = [px,py]
            if (dimensions == 3):
                coords = [px,py,pz]
            regions[label] = {"label" : label, "coords" : coords}

        #create nodes for the RAG
        rag.add_nodes_from(labels)

        #get adjacencies in each dimension
        endx    = []
        startx  = []
        #list containing all tuples
        pairs = [];
        for d in range(3):
            if (d == 0):
                idx = np.where(partition[:-1,:,:]!=partition[1:,:,:])
            elif (d == 1):
                idx = np.where(partition[:,:-1,:]!=partition[:,1:,:])
            elif (d == 2):
                idx = np.where(partition[:,:,:-1]!=partition[:,:,1:])
            incx = int(d == 0)
            incy = int(d == 1)
            incz = int(d == 2)
            adj = (partition[idx[0],idx[1],idx[2]], partition[idx[0]+incx,idx[1]+incy,idx[2]+incz])
            pairs = pairs + [(min(adj[0][i],adj[1][i]), max(adj[0][i],adj[1][i])) for i in range(len(adj[0]))]

        #find unique region pairs
        unique_pairs = set(pairs)

        # compute distances between regions
        edges = [(r[0],r[1]) for r in unique_pairs]

        rag.add_edges_from(edges)

        #return the rag, the regions dictionary and ordered distances
        return (rag, regions)