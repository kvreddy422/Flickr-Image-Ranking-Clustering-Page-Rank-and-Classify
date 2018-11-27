import numpy as np
from scipy.spatial.distance import cdist
import os
import pickle


def normalize_vectors(v, distances):
    for i in range(len(v)):
        v[i] = v[i] / distances[i]
    return v


class LSH:
    def __init__(self, l, k):
        ''' Initialize the lsh tool with k and l '''
        self.index_path = 'task5/index-'+str(l)+'-'+str(k)+'.pkl'
        self.vector_path = 'task5/vector-' + str(l) + '-' + str(k) + '.pkl'
        self.bias_path = 'task5/bias-' + str(l) + '-' + str(k) + '.pkl'
        self.buckets_path = 'task5/buckets-' + str(l) + '-' + str(k) + '.pkl'
        self.w_path = 'w-'+str(l)+'-'+str(k)+'.pkl'
        self.k = k  # number of hashes in one layer
        self.l = l  # number of layers of hashes
        self.w = 0  # window length
        self.b = None  # bias values
        self.data = None  # data set
        self.ids = None  # ids corresponding to each row in data
        self.r_vectors = None  # random vectors
        self.buckets = None  # buckets
        self.indexed_data = None  # index structure using buckets
        self.query = None  # query id
        self.query_point = None  # query vector
        self.table = None

    def initialize_vectors(self):
        data_set = self.data
        n_dims = len(data_set[0])
        n = self.l * self.k
        vectors = np.random.random((n, n_dims))
        length = cdist(vectors, np.zeros((1, n_dims))).reshape(n)
        vectors = normalize_vectors(vectors, length)
        if os.path.exists('task5/distances.npy'):
            d = np.load('task5/distances.npy')
        else:
            d = cdist(data_set, data_set)
            np.save('task5/distances.npy', d)
        self.w = float(d.max())/75
        with open(self.w_path,'wb') as f:
            pickle.dump(self.w, f)
        self.b = np.random.uniform(0, self.w, (self.l * self.k))
        with open(self.bias_path, 'wb') as f:
            pickle.dump(self.b, f)
        self.r_vectors = vectors
        with open(self.vector_path, 'wb') as f:
            pickle.dump(self.r_vectors, f)

    def project_data(self, vector=None):
        if vector is None:
            data_set = self.data
        else:
            data_set = vector
        projection = []
        for i in range(len(self.r_vectors)):
            p = np.dot(data_set, self.r_vectors[i].T)
            projection.append((p + self.b[i]) / self.w)
        return projection

    def assign_buckets(self, projection):
        buckets = []
        for p in projection:
            buckets.append(np.floor(p))
        if self.buckets is None:
            self.buckets = np.array(buckets).T
            with open(self.buckets_path, 'wb') as f:
                pickle.dump(self.buckets, f)
        else:
            return np.array(buckets).T

    def index(self):
        buckets = self.buckets
        data_set = self.data
        ids = self.ids
        index = {}
        for i in range(len(data_set)):
            n = 0
            for j in range(self.l):
                bucket_name = ''
                for m in range(self.k):
                    b = buckets[i][n]
                    bucket_name = (bucket_name + '^' + str(int(b))).strip('^')
                    if bucket_name in index.keys():
                        if ids[i] not in index[bucket_name]:
                            index[bucket_name].append(ids[i])
                    else:
                        index[bucket_name] = [ids[i]]
                    n += 1
        self.indexed_data = index
        with open(self.index_path, 'wb') as f:
            pickle.dump(self.indexed_data, f)

    def get_bucket_names(self, bucket_list):
        bucket_names = []
        n = 0
        for i in range(self.l):
            b_name = ''
            for j in range(self.k):
                b = bucket_list[n]
                b_name = (b_name + '^' + str(int(b))).strip('^')
                n += 1
            if b_name not in bucket_names:
                bucket_names.append(b_name)
        return bucket_names

    def find_distances(self, vector, objects):
        d = []
        for o in objects:
            i = self.ids.index(o)
            v = self.data[i]
            if o != self.query:  # ignore itself
                dist = np.linalg.norm(vector - v)
                d.append((dist, o))
        return d

    def initialize_tool(self, data_set, ids, delete):
        ''' Configure the tool for the given data set. Expects data_set as np array and ids as string ids corresponding
        to the data_set rows'''
        self.ids = ids
        self.data = data_set
        if os.path.exists(self.index_path) and delete:
            with open(self.vector_path, 'rb') as f:
                self.r_vectors = pickle.load(f)
            with open(self.bias_path, 'rb') as f:
                self.b = pickle.load(f)
            with open(self.w_path, 'rb') as f:
                self.w = pickle.load(f)
            with open(self.buckets_path, 'rb') as f:
                self.buckets = pickle.load(f)
            with open(self.index_path, 'rb') as f:
                self.indexed_data = pickle.load(f)
        else:
            self.initialize_vectors()
            projection = self.project_data()
            self.assign_buckets(projection)
            self.index()
        print('Your LSH tool is now ready. Number of layers: ' + str(self.l) + ', and number of hashes per layer: '
              + str(self.k))

    def find_n(self, bucket_names, t):
        num_considered = 0
        unique = set()
        index = self.indexed_data
        for i in range(self.k):
            for bucket in bucket_names:
                b = bucket.rsplit('^',i)[0]
                if b in index.keys():
                    objects = index[b]
                    num_considered += len(objects)
                    unique.update(objects)
            if len(unique) > t:  # finding more than t neighbours because self will be removed later
                break

        distances = self.find_distances(self.query_point, unique)
        distances = sorted(distances, key=lambda x: x[0])
        nn = [item[1] for item in distances[0:t]]
        return nn, num_considered, len(unique)

    def query_nn(self, data_point, t):
        self.query = data_point
        i = self.ids.index(data_point)
        vector = self.data[i]
        self.query_point = vector
        projection = self.project_data(vector)
        bucket_list = self.assign_buckets(projection)
        bucket_names = self.get_bucket_names(bucket_list)
        return self.find_n(bucket_names, t)

    def compare(self):
        d = self.find_distances(self.query_point, self.ids)
        distances = sorted(d, key=lambda x: x[0])
        nn = [item[1] for item in distances[0:20]]
        return nn
