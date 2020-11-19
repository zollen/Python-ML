'''
Created on Jul. 31, 2020

@author: zollen
'''

import seaborn as sb
from matplotlib import pyplot as plt
df = sb.load_dataset('iris')
sb.distplot(df['petal_length'], hist = True, kde = False, color='red')
plt.show()