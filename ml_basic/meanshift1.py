'''
Created on Aug. 10, 2020

@author: zollen
'''

import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.cluster import MeanShift
from sklearn.datasets import make_blobs


"""
## General Algorithm
## kernal_bandwidth directly influence how many clusters will be forms.
## Lower value of bandwidth encourages more peaks, more clustering
## higher value of bandwidth encourage less peaks, less clustering


def shift(p, original_points): 
    shift_x = float(0) 
    shift_y = float(0) 
    scale_factor = float(0) 
  
    for p_temp in original_points: 
        # numerator 
        dist = euclidean_dist(p, p_temp) 
        weight = kernel(dist, kernel_bandwidth) 
        shift_x += p_temp[0] * weight 
        shift_y += p_temp[1] * weight 
        # denominator 
        scale_factor += weight 
  
    shift_x = shift_x / scale_factor 
    shift_y = shift_y / scale_factor 
    return [shift_x, shift_y] 
    
for p in copied_points: 
    while not at_kde_peak: 
        p = shift(p, original_points)     
"""

sb.set_style('whitegrid')

X, y_true = make_blobs(n_samples = 400, centers = 4, cluster_std = 0.60, random_state = 0)

model = MeanShift(bandwidth=0.8)
model.fit(X)
y_kmeans = model.predict(X)


plt.scatter(X[:, 0], X[:, 1], c = y_kmeans, s = 20, cmap ='summer')
centers = model.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c = 'red', s = 20, alpha = 0.8);
plt.show()