__author__ = 'guillem'

import heapq
import networkx as nx
import numpy as np

def rag_2d(img, partition, region_distance):
        """
        @param img: image for the rag
        @param partition: partition of the image
        @param region_distance: distance function of two regions
        @return:a rag and a sorted list of distances
        """

        #create a RAG
        rag = nx.Graph()
        labels = np.unique(partition)

        #create a regions hastable organized by label
        regions = {}
        for label in labels:
            regions[label] = {"label" : label, "coords" : np.where(partition == label), "image" : img}

        #create nodes for the RAG
        rag.add_nodes_from(labels)

        #get adjacencies in the first dimension
        idx = np.where(partition[:-1,:]!=partition[1:,:])
        adj = (partition[idx[0],idx[1]], partition[idx[0]+1,idx[1]])
        pairsx = [(min(adj[0][i],adj[1][i]), max(adj[0][i],adj[1][i])) for i in range(len(adj[0]))]

        #get adjacencies in the second dimension
        idx = np.where(partition[:,:-1]!=partition[:,1:])
        adj = (partition[idx[0],idx[1]], partition[idx[0],idx[1]+1])
        pairsy = [(min(adj[0][i],adj[1][i]), max(adj[0][i],adj[1][i])) for i in range(len(adj[0]))]

        unique_pairs = set(pairsx + pairsy)

        # compute distances between regions
        distances = [(region_distance(regions[r[0]],regions[r[1]]),r[0],r[1]) for r in unique_pairs]
        edges = [(r[0],r[1], region_distance(regions[r[0]],regions[r[1]])) for r in unique_pairs]

        rag.add_weighted_edges_from(edges)
        distances = heapq.heapify(distances)

        #return the rag, the regions dictionary and ordered distances
        return (rag, regions, distances)