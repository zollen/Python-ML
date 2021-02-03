'''
Created on Feb. 3, 2021

@author: zollen
@url: https://towardsdatascience.com/dirichlet-distribution-a82ab942a879
'''


import numpy as np
import matplotlib.pyplot as plt

'''
The multi-dimension array indicates the percent sigificants for each class
If we have three classes, therefore 3D array
10: The first class length
5: The second class length
3: The third class length
'''
s = np.random.dirichlet((10, 5, 3), 20).transpose()
print(s.shape)

plt.barh(range(20), s[0])
plt.barh(range(20), s[1], left=s[0], color='g')
plt.barh(range(20), s[2], left=s[0]+s[1], color='r')
plt.title("Lengths of Strings")

plt.show()