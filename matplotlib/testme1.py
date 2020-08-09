'''
Created on Jul. 30, 2020

@author: zollen
'''
import math
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt

x = np.arange(0, math.pi * 2, 0.05)

sb.set_style("whitegrid")

plt.xlabel("angel")
plt.ylabel("sine")
plt.title("sine wave")

plt.plot(x, np.sin(x))
plt.plot(x, np.cos(x), 'r--')
plt.plot(x, -np.sin(x), 'g+')




plt.show()
