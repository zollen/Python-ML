'''
Created on Jul. 30, 2020

@author: zollen
'''
import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure()

ax = fig.add_axes([0.1, 0.1, 0.9, 0.9])

data = [[30, 25, 50, 20],
        [40, 23, 51, 17],
        [35, 22, 45, 19]]

X = np.arange(4)

ax.bar(X + 0.00, data[0], color = 'b', width = 0.25)
ax.bar(X + 0.20, data[1], color = 'g', width = 0.25)
ax.bar(X + 0.40, data[2], color = 'r', width = 0.25)

plt.show()


N = 5
menMeans = ( 20, 35, 30, 35, 27 )
womenMeans = ( 25, 32, 34, 20, 25 )
ind = np.arange(N)
width = 0.35

fig = plt.figure()
ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])

ax.bar(ind, menMeans, width, color='r')
ax.bar(ind, womenMeans, width, bottom=menMeans, color='b')
ax.set_ylabel('Scores')
ax.set_title('Scores by group and gender')
ax.set_xticks(ind, ('G1', 'G2', 'G3', 'G4', 'G5'))
ax.set_yticks(np.arange(0, 81, 10))
ax.legend(labels=['Men', 'Women'])



plt.show()