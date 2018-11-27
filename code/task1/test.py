import pickle
import networkx as nx
import matplotlib.pyplot as plt

x = None
# with open('pickles/cache/graph-k-10-20181119-123614.pkl', 'rb') as f:
#    x = pickle.load(f)
# with open('pickles/pre-processed/images_list.pkl', 'rb') as f:
#    y = pickle.load(f)


plt.plot([5, 6, 7, 8, 9, 10],
         [0.4851850097368347, 0.446025843001635, 0.415808252550941, 0.3674891811009707, 0.32722409950948483,
          0.3069819628310201])
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette value')
plt.show()
