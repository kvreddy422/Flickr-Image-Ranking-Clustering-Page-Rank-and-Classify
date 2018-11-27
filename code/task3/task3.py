import numpy as np
import networkx as nx
import sys
import datetime
from collections import Counter
import pandas as pd
from visualization.visualize import visualise_images


# This method as the name suggests, will take E(init_mat) matrix and
# The Restart/Teleportation/seed matrix as an input.Then Assuming
# The initial P vector as all ones, it will continue executing till
# the difference between two successive runs is lower than the "converg_err"
def power_method(init_mat, teleport_mat, converg_err):
    result = np.ones(teleport_mat.shape)
    tmp = np.zeros(teleport_mat.shape)
    start_time = datetime.datetime.now()
    i = 0
    # i = 1
    while np.sum(np.abs(result - tmp)) > converg_err:
        i += 1
        tmp = result
        result = np.dot(init_mat, result) + teleport_mat
        # for i in range
        nrm = np.linalg.norm(result)
        result = result / nrm
    curr_time = datetime.datetime.now()
    print("Converged in ", i+1, " iterations and ", (curr_time - start_time).seconds, "seconds.")
    return result

# This method as the name suggests, will take E(init_mat) matrix and
# The Restart/Teleportation/seed matrix as an input.Then Assuming
# The initial P vector as all ones, it will continue executing till
# the difference between two successive runs is lower than the "eps"


def calculatePageRank(init_mat, eps=1.0e-8, df=0.85):
    N = init_mat.shape[1]
    p = np.random.rand(N, 1)
    p = p / np.linalg.norm(p, 1)
    last_p = np.ones((N, 1), dtype=np.float32) * 100
    init_mat_hat = (df * init_mat) + (((1 - df) / N) * np.ones((N, N), dtype=np.float32))
    
    while np.linalg.norm(p - last_p, 2) > eps:
        last_p = p
        p = np.matmul(init_mat_hat, p)
    return p

# Tries to find the index corresponding to the input image ID from the dictionary
def find_image(imgdict, img):
    for index, imgID in imgdict.items():
        if imgID == img:
            return index
    return -1


if __name__ == '__main__':
    # candidate_imgs = (sys.argv[1]).split(",")
    k = int(sys.argv[2])
    df = 0.85
    error_thresh = 0.001

    GS = nx.read_gpickle("../task1/pickles/cache/"+sys.argv[1])

    # Storing the Image ID v/s Index values in a dict for further reference
    node_list = list(GS.nodes)
    node_index_dict = {i: str(node_list[i]) for i in range(0, len(node_list))}
    idict=Counter(dict(GS.in_degree(node_list)))

    # Converting the graph into a column stochastic matrix  
    inter_matrix = nx.adjacency_matrix(GS, nodelist=node_list, weight=None)
    task1_k = inter_matrix.sum(axis=1)
    A1=np.zeros((len(node_list), len(node_list)), int) 
    np.fill_diagonal(A1, np.asarray(task1_k)[0][0])
    init_matrix=(np.linalg.inv(A1)*inter_matrix).transpose()

    img_list=[]
   
    # pagerank_mat = power_method(left_mat, right_mat, error_thresh)
    pagerank_mat = calculatePageRank(init_matrix,0.001,0.85)

    i=0
    nodeDict={}
    for row in np.array(pagerank_mat):
        nodeDict[node_list[i]]=row[0]
        i+=1

    # sort the dictionary with pagerank values
    s = [(kl, nodeDict[kl]) for kl in sorted(nodeDict, key=nodeDict.get, reverse=True)]

    print("List of ", k, " Most dominant Images in PR")
    img_list=[]
    i=0
    for key,v in s:
        img_list.append(key)
        # print(idict[key])
        i+=1
        if(i==k):
            break 
    print(img_list) 
    
    # visualize the dominant images
    visualise_images('Task3 Dominant images',img_list) 
   