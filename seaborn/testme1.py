'''
Created on Jul. 29, 2020

@author: zollen
'''

import numpy as np
import seaborn as sb
from matplotlib import pyplot as plt

df = sb.load_dataset('titanic')

print(type(df))
print(len(df))
print(df.head())

def sinplot(flip = 1):
    x = np.linspace(0, 14, 100)
    for i in range(1, 5):
        plt.plot(x, np.sin(x + i * 0.5) * (7 - i) * flip)

sb.set_style('whitegrid')
#sb.set_palette("husl")

sinplot()

print(sb.axes_style())

plt.show()




