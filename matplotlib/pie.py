'''
Created on Jul. 30, 2020

@author: zollen
'''
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sb

sb.set_style("whitegrid")

fig = plt.figure()

ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
ax.axis('equal')

langs = [ 'C', 'C++', 'Java', 'Python', 'PHP' ]
students = [ 23, 17, 35, 29, 12 ]

ax.pie(students, labels = langs, autopct="%1.2f%%")

plt.show()