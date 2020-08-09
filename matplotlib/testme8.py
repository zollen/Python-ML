'''
Created on Jul. 30, 2020

@author: zollen
'''
import matplotlib.pyplot as plt
import numpy as np


fig, a = plt.subplots(1, 2, figsize=(10, 4))

x = np.arange(1, 5)

a[0].plot(x, np.exp(x))
a[0].plot(x, x**2)
a[0].set_xlabel('x axis')
a[0].set_ylabel('y axis')
a[0].xaxis.labelpad = 10
a[0].yaxis.labelpad = 10
a[0].set_title('Normal scale')

a[1].plot(x, np.exp(x))
a[1].plot(x, x**2)
a[1].set_yscale('log')
a[1].set_xlabel('x axis')
a[1].set_ylabel('y axis')
a[1].set_title('Logarithmic scale(y)')

plt.show()