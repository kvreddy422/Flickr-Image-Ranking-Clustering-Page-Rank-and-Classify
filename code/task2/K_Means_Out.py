import pickle
import scipy as sp
import networkx as nx
from scipy.sparse.linalg import svds
from collections import defaultdict
from visualization.visualize import visualize_images
import numpy as np
import pandas as pd
from sklearn.cluster import SpectralClustering
from scipy import stats
import time

start = time.time()
N_Centroids = input('Enter the No of Centroids')
N_Centroids = int(N_Centroids)
input_data = pickle.load(open('pickle/graph-k-10-20181119-123614.pkl', "rb"))
input_data2 = pickle.load(open('pickle/graph-k-10-20181119-123614.pkl', "rb"))
K = len(list(nx.neighbors(input_data,list(input_data.nodes)[0])))
global adj_mat
# A=nx.adjacency_matrix(input_data)
adj_mat = nx.to_numpy_array(input_data)
adj_mat[adj_mat <= 0] = 1  # Making the values = 1 where there are no edges between the nodes
np.fill_diagonal(adj_mat, 0)  # We are filling the diagonal matrix with value 0
# adj_mat1 = pickle.load(open('pickles/total_diffs.pkl', "rb"))
# A2=nx.adjacency_matrix(input_data2)
adj_mat1 = nx.to_numpy_array(input_data2)
adj_mat1[adj_mat1 <= 0] = 1   # Making the values = 1 where there are no edges between the nodes
np.fill_diagonal(adj_mat1, 0) # We are filling the diagonal matrix with value 0

image_ids = None
with open('pickle/images_list.pkl', 'rb') as f:
    image_ids = pickle.load(f)

image_ids_copy = list(image_ids).copy()
# print(adj_mat)
cluster_dict = defaultdict(list)


def getMinMaxCentriods(adj_mat1,node_list,c=4):
    # adj_mat1[adj_mat1 > 0.001] = 0 # need to get the 0.006 dynamically;
    centroids = [] # Centroids are strored here
    centroids_index = [] # List strores the index of the centiords .. this makes it easy for us retrive the actual image
    factor = len(node_list)//c
    for i in range(c):
        centroids.append(node_list[i*factor+10])
        centroids_index.append(i*factor+10)
    visualize_images("Custer ", centroids)
    return centroids_index,centroids

def createDictCentroid(centroids,centroidIndex):
    centroids_dict = {}
    centroids_dict_val = {}
    for centroid_val in centroidIndex:
        image_ids_copy.remove(image_ids[centroid_val])
        centroids_dict[image_ids[centroid_val]]=[]
        centroids_dict_val[centroid_val] = []
        centroids_dict[image_ids[centroid_val]].append(image_ids[centroid_val])
        centroids_dict_val[centroid_val].append(0)
    return centroids_dict,centroids_dict_val

def assignKNearestToCentroid(centroids_dict_array):
    centroids_dict_index = centroids_dict_array[1]
    centroids_dict_images = centroids_dict_array[0]
    adj_mat_int = adj_mat.copy()
    for c in centoidValues[0]:
        K_Nearest_Neightbours = list(nx.neighbors(input_data, image_ids[c]))
        print('')
        for k in K_Nearest_Neightbours:
            if image_ids_copy.__contains__(k):
                image_ids_copy.remove(k)
                centroids_dict_images[image_ids[c]].append(k)
                centroids_dict_index[c].append(adj_mat[c][image_ids.index(k)])
            else:
                continue
    return centroids_dict_index,centroids_dict_images



def addNodesToCentroids(centoidValues):
    centroids = centoidValues[1]
    centroidIndex = centoidValues[0]
    centroids_dict_array = createDictCentroid(centroids,centroidIndex)
    centroids_dict = assignKNearestToCentroid(centroids_dict_array)
    i=0
    while i<6:
        for image in image_ids_copy:
            K_Nearest_Neightbours = list(nx.neighbors(input_data, image))
            flag = 0
            for k in range(K):
                min = 10
                if(flag==1):
                    break
                if(k==0):
                    continue
                centroids_attach = -1
                for c in centroids:
                    if(centroids_dict[1][c].__contains__(K_Nearest_Neightbours[k])):
                        try:
                            if(min>nx.dijkstra_path_length(input_data,image,c)):
                                min = nx.dijkstra_path_length(input_data,image,c)
                                centroids_attach = c
                        except nx.NetworkXNoPath:
                                continue
                    else:
                        continue
                if (centroids_attach > 0):
                    centroids_dict[0][list(image_ids).index(centroids_attach)].append(adj_mat[list(image_ids).index(centroids_attach)][list(image_ids).index(image)])
                    centroids_dict[1][centroids_attach].append(image)
                    image_ids_copy.remove(image)
                    flag =1
        i=i+1

    return centroids_dict

centoidValues = getMinMaxCentriods(adj_mat1,image_ids,N_Centroids)
centroids_dict = addNodesToCentroids(centoidValues)

for index, key in enumerate(list(centroids_dict[1].keys())):
    visualize_images("CusterNew " + str(index + 1), list(centroids_dict[1][key]))
print('Time Taken'+ str(time.time()-start))


def getMinMaxCentriods(adj_mat,node_list,c=5):
    centroids = []
    centroids.append(node_list)
    print('Code Here')