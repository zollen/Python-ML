'''
Created on Jul. 30, 2020

@author: zollen
'''

import matplotlib.pyplot as plt

plt.subplot(211)
plt.plot(range(12))

plt.subplot(212, facecolor='y')
plt.plot(range(12))

fig = plt.figure()

ax1 = fig.add_subplot(111)
ax1.plot([1, 2, 3])

ax2 = fig.add_subplot(221, facecolor='y')
ax2.plot([1, 2, 3])

plt.show()