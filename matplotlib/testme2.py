'''
Created on Jul. 30, 2020

@author: zollen
'''
import math
import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0, math.pi * 2, 0.05)

fig = plt.figure()

ax = fig.add_axes([0, 0, 1, 1])

ax.set_xlabel("angel")
ax.set_ylabel("sine")
ax.set_title("sine wave")

ax.plot(x, np.sin(x))
ax.plot(x, np.cos(x), 'r--')
ax.legend(labels = ('sin', 'cos'), loc = 'lower right')

plt.show()