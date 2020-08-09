'''
Created on Jul. 31, 2020

@author: zollen
'''
import matplotlib.pyplot as plt
fig = plt.figure()

ax = fig.add_axes([0.1,0.1,0.8,0.8])

ax.set_title('axes title')
ax.set_xlabel('xlabel')
ax.set_ylabel('ylabel')
ax.text(3, 8, 'Hello World!!', style='italic', 
bbox = {'facecolor': 'red'})

ax.text(2, 6, r'an equation: $E = mc^2$', fontsize = 15)
ax.text(4, 0.05, 'colored text in axes coords', verticalalignment = 'bottom', color = 'green', fontsize = 15)
ax.plot([2], [1], 'o')
ax.annotate('here', xy = (2, 1), xytext = (3, 4),
arrowprops = dict(facecolor = 'black', shrink = 0.05))
ax.axis([0, 10, 0, 10])
plt.show()