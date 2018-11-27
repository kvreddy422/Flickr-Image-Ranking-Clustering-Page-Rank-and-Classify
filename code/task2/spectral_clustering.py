import numpy as np
import networkx as nx
import pickle
from visualization.visualize import visualize_images
from collections import defaultdict
from copy import deepcopy
from scipy.sparse.linalg import eigsh
import math
import sys
import random
from sklearn.cluster import KMeans
from scipy.spatial import distance as d
import scipy.spatial
import sklearn
import sys

img_img_graph = None
with open('../task1/pickles/cache/' + str(sys.argv[1]), 'rb') as f:
    img_img_graph = pickle.load(f)
images_list = None
with open('../task1/pickles/pre-processed/images_list.pkl', 'rb') as f:
    images_list = pickle.load(f)


def centroid(data):
    """Find the centroid of the given data."""
    return np.mean(data, 0)


def sse(data):
    """Calculate the SSE of the given data."""
    u = centroid(data)
    return np.sum(np.linalg.norm(data - u, 2, 1))


class KMeansClusterer:
    """The standard k-means clustering algorithm."""

    def __init__(self, data=None, k=2, min_gain=0.01, max_iter=100,
                 max_epoch=10, verbose=True):
        """Learns from data if given."""
        if data is not None:
            self.fit(data, k, min_gain, max_iter, max_epoch, verbose)

    def fit(self, data, k=2, min_gain=0.01, max_iter=100, max_epoch=10,
            verbose=False):
        """Learns from the given data.
        Args:
            data:      The dataset with m rows each with n features
            k:         The number of clusters
            min_gain:  Minimum gain to keep iterating
            max_iter:  Maximum number of iterations to perform
            max_epoch: Number of random starts, to find global optimum
            verbose:   Print diagnostic message if True
        Returns:
            self
        """
        # Pre-process
        self.data = np.matrix(data)
        self.k = k
        self.min_gain = min_gain

        # Perform multiple random init for global optimum
        min_sse = np.inf
        for epoch in range(max_epoch):

            # Randomly initialize k centroids
            indices = np.random.choice(len(data), k, replace=False)
            u = self.data[indices, :]

            # Loop
            t = 0
            old_sse = np.inf
            while True:
                t += 1

                # Cluster assignment
                C = [None] * k
                for x in self.data:
                    j = np.argmin(np.linalg.norm(x - u, 2, 1))
                    C[j] = x if C[j] is None else np.vstack((C[j], x))

                # Centroid update
                for j in range(k):
                    u[j] = centroid(C[j])

                # Loop termination condition
                if t >= max_iter:
                    break
                new_sse = np.sum([sse(C[j]) for j in range(k)])
                gain = old_sse - new_sse
                if verbose:
                    line = "Epoch {:2d} Iter {:2d}: SSE={:10.4f}, GAIN={:10.4f}"
                    print(line.format(epoch, t, new_sse, gain))
                if gain < self.min_gain:
                    if new_sse < min_sse:
                        min_sse, self.C, self.u = new_sse, C, u
                    break
                else:
                    old_sse = new_sse

            if verbose:
                print('')  # blank line between every epoch

        return self


def deterministic_vector_sign_flip(u):
    max_abs_rows = np.argmax(np.abs(u), axis=1)
    signs = np.sign(u[range(u.shape[0]), max_abs_rows])
    u *= signs[:, np.newaxis]
    return u


def make_symmetric(array):
    tol = 1E-10
    diff = array - array.T
    symmetric = np.all(abs(diff.data) < tol)
    if not symmetric:
        conversion = 'to' + array.format
        array = getattr(0.5 * (array + array.T), conversion)()
    return array


def _laplacian_sparse(graph, normed=False, axis=0):
    if graph.format in ('lil', 'dok'):
        m = graph.tocoo()
        needs_copy = False
    else:
        m = graph
        needs_copy = True
    w = m.sum(axis=axis).getA1() - m.diagonal()
    if normed:
        m = m.tocoo(copy=needs_copy)
        isolated_node_mask = (w == 0)
        w = np.where(isolated_node_mask, 1, np.sqrt(w))
        m.data /= w[m.row]
        m.data /= w[m.col]
        m.data *= -1
        m.setdiag(1 - isolated_node_mask)
    else:
        if m.format == 'dia':
            m = m.copy()
        else:
            m = m.tocoo(copy=needs_copy)
        m.data *= -1
        m.setdiag(w)
    return m, w


def _setdiag_dense(A, d):
    A.flat[::len(d) + 1] = d


def set_diag(laplacian, value, norm_laplacian):
    laplacian = laplacian.tocoo()
    if norm_laplacian:
        diag_idx = (laplacian.row == laplacian.col)
        laplacian.data[diag_idx] = value
    n_diags = np.unique(laplacian.row - laplacian.col).size
    if n_diags <= 7:
        laplacian = laplacian.todia()
    else:
        laplacian = laplacian.tocsr()
    return laplacian


def dist(a, b, ax=0):
    return np.linalg.norm(a - b, axis=ax)


import random
from scipy.spatial import distance
import operator


def getDistance(v1, v2):
    return distance.euclidean(v1, v2)


def getMean(Mat):
    finalVector = []

    for i in range(len(Mat[0])):
        finalVector.append(0)

    for i in range(len(Mat)):
        for j in range(len(Mat[0])):
            finalVector[j] = finalVector[j] + Mat[i][j]

    for i in range(len(finalVector)):
        finalVector[i] = finalVector[i] / float(len(Mat))

    return finalVector


def findClusters(centroids, IM, K):
    ClusterImage = {}
    for i in range(K):
        ClusterImage[i] = []

    for i in range(len(IM)):
        queryImage = IM[i]
        unsortedDict = {}
        for j in range(K):
            unsortedDict[j] = getDistance(queryImage, centroids[j])
        sortedDict = sorted(unsortedDict.items(), key=operator.itemgetter(1))
        ClusterImage[sortedDict[0][0]].append(i)

    return ClusterImage


def updateCentroid(centroids, IM, K):
    ClusterImage = {}
    for i in range(K):
        ClusterImage[i] = []

    for i in range(len(IM)):
        queryImage = IM[i]
        unsortedDict = {}
        for j in range(K):
            unsortedDict[j] = getDistance(queryImage, centroids[j])
        sortedDict = sorted(unsortedDict.items(), key=operator.itemgetter(1))
        ClusterImage[sortedDict[0][0]].append(queryImage)

    newCentroids = []
    for i in range(K):
        newCentroids.append(getMean(ClusterImage[i]))

    return newCentroids


def kMeansG(ImageMatrix, K, numOfIter=500):
    print("K:", K)
    m = len(ImageMatrix)
    n = len(ImageMatrix[0])

    print("Number of Iterations:", numOfIter)

    print("ImageMatrix Shape: " + str(m) + " * " + str(n))

    randomIndexes = []

    for i in range(K):
        index = random.randint(0, m - 1)
        while (index in randomIndexes):
            index = random.randint(0, m - 1)
        randomIndexes.append(random.randint(0, m - 1))

    print("Random Indexes:", randomIndexes)

    centroids = []

    for j in randomIndexes:
        centroids.append(ImageMatrix[j])

    for i in range(numOfIter):
        # print(i)
        centroids = updateCentroid(centroids, ImageMatrix, K)

    return findClusters(centroids, ImageMatrix, K)


def function_sai(X, k, n_iter):
    rand_cent_index = np.random.randint(0, len(X), size=k)
    centroids = [X[i] for i in rand_cent_index]
    cdist_of_data = scipy.spatial.distance.cdist(X, centroids, metric='euclidean')
    cluster_dict = defaultdict(list)
    for index, row in enumerate(cdist_of_data):
        cluster_dict[np.argmin(row)].append(index)
    for _ in range(n_iter):
        new_centroids = []
        for index, cluster_id in enumerate(cluster_dict.keys()):
            new_centroids.append(np.mean(np.array([X[d_row] for d_row in cluster_dict[cluster_id]]), axis=0))
        cdist_of_data = scipy.spatial.distance.cdist(X, new_centroids, metric='euclidean')
        cluster_dict = defaultdict(list)
        for index, row in enumerate(cdist_of_data):
            cluster_dict[np.argmin(row)].append(index)
    return cluster_dict

adj = nx.adj_matrix(img_img_graph)  # getting adjenc
adj = make_symmetric(adj)
n_components = int(sys.argv[2])
n_nodes = adj.shape[0]
create_lap = _laplacian_sparse
laplacian, dd = create_lap(adj, normed=True, axis=0)
laplacian = set_diag(laplacian, 1, True)
laplacian *= -1
v0 = np.random.uniform(-1, 1, laplacian.shape[0])
lambdas, diffusion_map = eigsh(laplacian, k=n_components,
                               sigma=1.0, which='LM',
                               v0=v0)
embedding = diffusion_map.T[n_components::-1]
embedding = embedding / dd
embedding = deterministic_vector_sign_flip(embedding)
labels = KMeans(n_clusters=n_components, random_state=0, n_init=300, n_jobs=3).fit(embedding[:n_components].T).labels_
cluster_dict = kMeansG(embedding[:n_components].T, n_components, 150)

# for index, label in enumerate(labels):
#    cluster_dict[label].append(images_list[index])
#
# for index, label in enumerate(cluster_dict.keys()):
#    visualize_images("Cluster " + str(index + 1), cluster_dict[label])
for index, cluster_id in enumerate(list(cluster_dict.keys())):
    visualize_images("Cluster " + str(cluster_id + 1), list([images_list[index] for index in cluster_dict[cluster_id]]))
# with open('../task1/pickles/pre-processed/total_diffs.pkl', 'rb') as f:
#    total_diffs = pickle.load(f)

# si = sklearn.metrics.silhouette_score(embedding[:n_components].T, labels, metric='euclidean')
# print("Silhouette:" + str(si))
