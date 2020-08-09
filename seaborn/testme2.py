'''
Created on Jul. 31, 2020

@author: zollen
'''
from matplotlib import pyplot as plt
import seaborn as sns, numpy as np
sns.set(); np.random.seed(0)
x = np.random.randn(100)
print(x)
ax = sns.distplot(x, rug=True, hist=True, kde=False)

plt.show()