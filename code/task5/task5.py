from task5.lsh import LSH
import os.path as op
from os import listdir
import pandas as pd
from visualization import visualize as vis
import numpy as np
import os
import sys


def get_data():
    if os.path.exists('data.npy') and os.path.exists('images.npy'):
        data = np.load('data.npy')
        images = np.load('images.npy')
        return data, images

    models = ['CM', 'CM3x3', 'CN', 'CN3x3', 'CSD', 'GLRLM', 'GLRLM3x3', 'HOG', 'LBP', 'LBP3x3']
    path = 'devset/descvis/img/'
    model_dict = {}
    files = [f for f in sorted(listdir(path)) if op.isfile(op.join(path, f))]
    for f in files:
        m = f.split(' ')[1].replace('.csv', '')
        # im = pd.read_csv(path + '/' + f, dtype=int, header=None).values[:, 0]
        contents = pd.read_csv(path + '/' + f, header=None)
        if m in model_dict:
            model_dict[m]['images'] = np.append(model_dict[m]['images'], contents[contents.columns[0]].values)
            model_dict[m]['values'] = np.vstack((model_dict[m]['values'], contents[contents.columns[1:]].values))
        else:
            model_dict[m]= {}
            model_dict[m]['images'] = contents[contents.columns[0]].values
            model_dict[m]['values'] = contents[contents.columns[1:]].values
    data = None
    for model in models:
        vectors = model_dict[model]['values']
        minv = vectors.min()
        maxv = vectors.max()
        norm_vectors = (vectors-minv)/float(maxv-minv)
        model_dict[model]['values'] = norm_vectors
        if data is not None:
            data = np.hstack((data, norm_vectors))
        else:
            data = norm_vectors
    data = get_SVD(data)
    np.save('task5/data.npy', data)
    np.save('task5/images.npy', model_dict['CM']['images'])
    return data, model_dict['CM']['images']


def get_SVD(data):
    U, S, V = np.linalg.svd(data)
    return U[:, :300]


if __name__ == "__main__":
    arguments = sys.argv
    vectors, images = get_data()
    lsh = LSH(int(arguments[1]), int(arguments[2]))
    lsh.initialize_tool(vectors, images.tolist(), False)
    y = 'y'
    while y == 'y'or y == 'Y':
        query_point = int(input('\nEnter the query image:'))
        t = int(input('\nEnter the number of nearest neighbours:'))
        nn, nc, nu = lsh.query_nn(query_point, t)
        cnn = lsh.compare()
        indexes = []
        for i in range(len(cnn)):
            if cnn[i] in nn:
                indexes.append(cnn.index(cnn[i]))
        print("\n" + str(t)+" nearest neighbors of "+ str(lsh.query) + " are:")
        for i in range(len(nn)):
            print(str(i+1)+". " + str(nn[i]))

        # print(indexes)

        print("\nTotal number of images considered: " + str(nc))
        print("Number of unique images considered: " + str(nu))
        vis.visualize_query_results("Indexed nearest neighbors using LSH", "Query Image "+ str(lsh.query)+ ":", [lsh.query],
                                    str(t) + " Nearest neightbours:", nn)
        y = input("\n Do you want to query again? (Y/N)")
