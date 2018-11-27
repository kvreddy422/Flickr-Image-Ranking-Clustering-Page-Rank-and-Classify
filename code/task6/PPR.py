import numpy as np
import networkx as nx
import sys
import datetime


# This method as the name suggests, will take E(init_mat) matrix and
# The Restart/Teleportation/seed matrix as an input.Then Assuming
# The initial P vector as all ones, it will continue executing till
# the difference between two successive runs is lower than the "converg_err"
def power_method(init_mat, teleport_mat, converg_err):
    N = init_mat.shape[1]
    result = np.random.rand(N, 1)
    result = result / np.linalg.norm(result, 1)
    tmp = np.ones((N, 1), dtype=np.float32) * 100
    start_time = datetime.datetime.now()
    i = 0
    while np.sum(np.abs(result - tmp)) > converg_err:
        i += 1
        tmp = result
        result = np.matmul(init_mat, result) + teleport_mat
        nrm = np.linalg.norm(result)
        result = result / nrm

    curr_time = datetime.datetime.now()
    #print("Converged in ", i+1, " iterations and ", (curr_time - start_time).seconds, "seconds.")
    return result


def list_PPR(candidate_image_str, pickleFile):
    candidate_imgs = (candidate_image_str).split(",")
    df = 0.85
    error_thresh = 0.001

    GS = nx.read_gpickle(pickleFile)#"graph-k-10-20181124-155609.pkl")

    # Storing the Image ID v/s Index values in a dict for further reference
    node_list = list(GS.nodes)
    node_index_dict = {i: str(node_list[i]) for i in range(0, len(node_list))}

    # Converting the graph into a column stochastic matrix
    inter_matrix = nx.adjacency_matrix(GS, nodelist=node_list, weight=None).todense().transpose()
    init_matrix = inter_matrix / inter_matrix.sum(axis=0)

    # Creating the teleportation/seed vector based on the input candidate images
    tele_matrix = np.zeros((len(init_matrix), 1), dtype=np.float32)
    tele_matrix[[node_list.index(int(x)) for x in candidate_imgs]] = 1.0

    left_mat = df * init_matrix
    right_mat = (((1 - df) / len(candidate_imgs)) * tele_matrix)

    pagerank_mat = power_method(left_mat, right_mat, error_thresh).transpose()

    # This section simply sorts the page ranks and give top K imageID, Page Rank,
    # while preserving the indexes so that the original node (Image ID) can be derived
    pr_dict = {}
    for index in range(len(init_matrix)):
        pr_dict[node_index_dict[index]] = pagerank_mat[0, index]

    return pr_dict