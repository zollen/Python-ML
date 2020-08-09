'''
Created on Jul. 30, 2020

@author: zollen
'''
import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure()

x = np.arange(1, 10)
xticks = np.arange(1, 10, 2)
yticks = np.arange(0, 10000, 500)

a1 = fig.add_axes([ 0.1, 0.1, 0.8, 0.8 ])
a1.plot(x, np.exp(x), 'r')
a1.set_title('exp')
a1.set_ylim(0, 9000)
a1.set_xlim(0, 10)
a1.set_xticks(xticks)
a1.set_xticklabels(['one', 'three', 'five', 'seven', 'nine'])
a1.set_yticks(yticks)

a2 = a1.twinx()
a2.plot(x, np.log(x), 'ro-')
#a2.set_yscale('log')
a2.set_ylabel('log')

fig.legend(labels=('exp', 'log'), loc = 'upper left')

plt.show()