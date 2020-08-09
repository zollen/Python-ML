'''
Created on Aug. 8, 2020

@author: zollen
'''
import numpy as np
import seaborn as sb
from matplotlib import pyplot as plt


sb.set_style("whitegrid")


x = np.array([ 1, 2, 3, 4, 5 ])
y = np.array([ 1, 2, 3, 4, 5 ])
z = np.array([ 
    [ 0.5,   0.5,   0.5,   1.0,   1.0 ],
    [ 0.5,   0.5,   0.5,   1.0,   1.0 ],
    [ 0.0,   0.0,   0.0,   1.0,   1.0 ],
    [ 0.0,   0.0,   0.0,   1.0,   1.0 ],
    [ 0.0,   0.0,   0.0,   1.0,   1.0 ]])

a = np.array([ 2.0, 3.0, 3.9 ])
b = np.array([ 1.2, 2.1, 4.2 ])
c = np.array([ 1.0, 1.0, 1.0 ])

print(x)

X, Y = np.meshgrid(x, y)

print(X)

Z = z

plt.figure(figsize=(8, 7))
plt.contourf(X, Y, Z, cmap=plt.cm.tab10, alpha=0.3, levels=[0, 0.49, 0.99]);

plt.scatter(a, b, c=c, cmap=plt.cm.Set1)

plt.show()