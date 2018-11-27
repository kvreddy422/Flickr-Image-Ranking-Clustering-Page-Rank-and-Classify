import pandas as pd
import networkx as nx
import numpy as np
import scipy.spatial
import pickle
from utility import get_loc_id_mapping
import time
import collections
from sklearn.preprocessing import MinMaxScaler

start = time.time()
root_dir = ""
visual_d_path = "../devset/descvis/img"
path = visual_d_path
visual_models = ["CM", "CN", "CSD", "HOG", "LBP", "CM3x3", "CN3x3", "GLRLM", "LBP3x3", "GLRLM3x3"]
locations = get_loc_id_mapping().keys()
model_dict = {}
image_ids = []
model_diffs_dict = {}
for loc in locations:
    file_content = pd.read_csv(path + "/" + loc + " " + "CM" + ".csv", header=None)
    image_ids.extend(file_content[file_content.columns[0]])
image_ids_no_duplicates = list(set(image_ids))
duplicates_images = [item for item, count in collections.Counter(image_ids).items() if count > 1]
with open('pickles/pre-processed/images_list.pkl', 'wb') as f:
    pickle.dump(image_ids_no_duplicates, f)
img_img_graph = nx.DiGraph()
img_img_graph.add_nodes_from(list(image_ids_no_duplicates))
# try:
#    model_diffs_dict = pd.read_pickle('pickles/pre-processed/model_diffs.pkl')
# except FileNotFoundError:
for vm in visual_models:
    model_tmp_list = []
    added = []
    for loc in locations:
        data = pd.read_csv(path + "/" + loc + " " + vm + ".csv", header=None)
        # for image_id in image_ids_no_duplicates:
        #     row_df = data.loc[data[0] == image_id]
        #     model_tmp_list.append(row_df.iloc[:, 1:])
        model_tmp_list.append(data)
    tmp = np.vstack(list(model_tmp_list))
    data_in_order = []
    for image_id in image_ids_no_duplicates:
        data_in_order.append(tmp[list(tmp[:, 0]).index(image_id)][1:])
    model_dict[vm] = np.vstack(list(data_in_order))
    scaler = MinMaxScaler()
    scaler.fit(model_dict[vm])
    normalized = scaler.transform(model_dict[vm])
    model_diffs_dict[vm] = scipy.spatial.distance.cdist(normalized, normalized, metric='euclidean')
with open('pickles/pre-processed/model_diffs.pkl', 'wb') as pkl_file:
    pickle.dump(model_diffs_dict, pkl_file)
    # TODO: implement multiprocessing

total_diffs = np.zeros(model_diffs_dict[list(model_diffs_dict.keys())[0]].shape)
for vm in visual_models:
    # row_sums = model_diffs_dict[vm].sum(axis=0)
    # new_matrix = model_diffs_dict[vm] / row_sums[:, np.newaxis]
    # total_diffs += new_matrix
    total_diffs += model_diffs_dict[vm]
sorted_indexes_dict = {}
with open('pickles/pre-processed/total_diffs.pkl', 'wb') as f:
    pickle.dump(total_diffs, f)

# print(total_diffs.shape)
# try:
#     sorted_indexes_dict = pd.read_pickle('pickles/pre-processed/sorted_indexes.pkl')
# except FileNotFoundError:
for index, image_id in enumerate(image_ids_no_duplicates):
    sorted_indexes_dict[image_id] = np.argsort(total_diffs[index])
with open('pickles/pre-processed/sorted_indexes.pkl', 'wb') as pkl_file:
    pickle.dump(sorted_indexes_dict, pkl_file)
k = 10
for i, image_id in enumerate(image_ids_no_duplicates):
    for srtd in sorted_indexes_dict[image_id][:k+1]:
        if image_id == image_ids_no_duplicates[srtd]:
            continue
        img_img_graph.add_edge(image_id, image_ids_no_duplicates[srtd], weight=total_diffs[i][srtd])
time_str = time.strftime("%Y%m%d-%H%M%S")
with open('pickles/cache/graph-k-' + str(k) + '-' + time_str + '.pkl', 'wb') as f:
    pickle.dump(img_img_graph, f)
print("Total execution time is : " + str(time.time() - start) + " Seconds.")
