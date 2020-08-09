'''
Created on Jul. 30, 2020

@author: zollen
'''
import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np

sb.set_style("whitegrid")

fig, a = plt.subplots(2, 2)

x = np.arange(1, 5)

a[0][0].plot(x, x*x)
a[0][0].set_title('sqaure')

a[0][1].plot(x, np.sqrt(x))
a[0][1].set_title('square root')

a[1][0].plot(x, np.exp(x))
a[1][0].set_title('exp')

a[1][1].plot(x, np.log10(x))
a[1][1].set_title('log')


plt.show()