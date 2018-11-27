import pickle
from sklearn import cluster
import networkx as nx
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics.cluster import adjusted_rand_score
from visualization.visualize import visualize_images


def graph_to_edge_matrix(G, images_list):
    """Convert a networkx graph into an edge matrix.
    See https://www.wikiwand.com/en/Incidence_matrix for a good explanation on edge matrices

    Parameters
    ----------
    G : networkx graph
    """
    # Initialize edge matrix with zeros
    edge_mat = np.zeros((len(G), len(G)), dtype=int)

    # Loop to set 0 or 1 (diagonal elements are set to 1)
    for node in G:
        for neighbor in G.neighbors(node):
            edge_mat[images_list.index(node)][images_list.index(neighbor)] = 1
        edge_mat[images_list.index(node)][images_list.index(node)] = 1

    return edge_mat


img_img_graph = None
with open('pickles/cache/graph-k-10-20181123-165012.pkl', 'rb') as f:
    img_img_graph = pickle.load(f)
with open('pickles/pre-processed/images_list.pkl', 'rb') as f:
    images_list = pickle.load(f)
edge_matrix = graph_to_edge_matrix(img_img_graph, images_list)
k_clusters = 10
results = []
algorithms = {}
algorithms['kmeans'] = cluster.KMeans(n_clusters=k_clusters, n_init=1)
for model in algorithms.values():
    model.fit(edge_matrix)
    results.extend(model.labels_)
clusters = {}
for cluster_id in set(results):
    clusters[cluster_id] = [i for i, x in enumerate(results) if x == cluster_id]
# [main_list[x] for x in indexes]
for cluster_id in clusters.keys():
    visualize_images("Cluster id " + str(cluster_id), [images_list[x] for x in clusters[cluster_id]])
