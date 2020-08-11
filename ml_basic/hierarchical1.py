'''
Created on Aug. 11, 2020

@author: zollen
'''

from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
import seaborn as sb
import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering

X = np.array(
   [[7,8],[12,20],[17,19],[26,15],[32,37],[87,75],[73,85], [62,80],[73,60],[87,96],])
labels = range(1, 11)
plt.figure(figsize = (10, 7))
plt.subplots_adjust(bottom = 0.1)
plt.scatter(X[:,0],X[:,1], label = 'True Position')
for label, x, y in zip(labels, X[:, 0], X[:, 1]):
    plt.annotate(label, xy = (x, y), xytext = (-3, 3),textcoords = 'offset points', ha = 'right', va = 'bottom')


linked = linkage(X, 'single')
labelList = range(1, 11)
plt.figure(figsize = (10, 7))
dendrogram(linked, orientation = 'top',labels = labelList, 
   distance_sort ='descending',show_leaf_counts = True)

plt.figure()


cluster = AgglomerativeClustering(n_clusters = 2, affinity = 'euclidean', linkage = 'ward')
cluster.fit_predict(X)


plt.scatter(X[:,0],X[:,1], c = cluster.labels_, cmap = 'rainbow')

plt.figure()

df = pd.DataFrame(dict({'x': X[:, 0], 'y': X[:, 1], 'member': cluster.labels_}))

sb.scatterplot(x = "x", y = "y", hue="member", data=df)

plt.show()