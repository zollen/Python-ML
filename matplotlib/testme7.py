'''
Created on Jul. 30, 2020

@author: zollen
'''
import matplotlib.pyplot as plt
import numpy as np

fig, a = plt.subplots(1, 3, figsize = (12, 4))

x = np.arange(1, 11)

a[0].plot(x, x**3, 'g', lw=2)
a[0].grid(True)
a[0].set_title('default grid')

a[1].plot(x, np.exp(x), 'r')
a[1].grid(color='b', ls='-.', lw = 0.25)
a[1].set_title('custom grid')

a[2].plot(x, x)
a[2].set_title('no grid')

fig.tight_layout()

plt.show()