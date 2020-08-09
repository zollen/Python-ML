'''
Created on Jul. 31, 2020

@author: zollen
'''
import seaborn as sb
import pandas as pd
from matplotlib import pyplot as plt

pd.set_option('max_columns', None)
pd.set_option('max_rows', None)
sb.set_style('whitegrid')

df = sb.load_dataset('iris')

print(df.head())

#sb.set_style("ticks")
sb.pairplot(df, hue = 'species', diag_kind = "hist", kind = "scatter", palette = "husl")
plt.show()