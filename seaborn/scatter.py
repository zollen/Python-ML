'''
Created on Jul. 31, 2020

@author: zollen
'''

import seaborn as sb
from matplotlib import pyplot as plt


df = sb.load_dataset('iris')

print(df.head())


sb.jointplot(x = 'petal_length',y = 'petal_width',data = df, color='red')

sb.jointplot(x = 'petal_length',y = 'petal_width',data = df,kind = 'hex')

sb.jointplot(x = 'petal_length',y = 'petal_width',data = df,kind = 'kde')


plt.show()