'''
Created on Aug. 11, 2020

@author: zollen
'''

import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt


"""
Kernal density destimation function for meanshift clustering
It smooths out the original curve by estimating the underlying distribution
"""
sb.set_style("whitegrid")


def make_data(N, f=0.3, rseed=87):
    rand = np.random.RandomState(rseed)
    x = rand.randn(N)
    x[int(f * N):] += 5
    return x

x = make_data(1000)

print(x)


sb.distplot(x, kde = True, norm_hist = True, color='blue')


plt.show()