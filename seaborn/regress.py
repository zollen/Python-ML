'''
Created on Aug. 1, 2020

@author: zollen
'''

import pandas as pd
import seaborn as sb
from matplotlib import pyplot as plt

pd.set_option('max_columns', None)

df = sb.load_dataset('tips')
print(df.head())

df1 = sb.load_dataset('anscombe')
print(df1.head())

sb.regplot(x = "total_bill", y = "tip", data = df)

sb.lmplot(x = "total_bill", y = "tip", data = df)

sb.lmplot(x = "size", y = "tip", data = df)

sb.lmplot(x="x", y="y", data=df1.query("dataset == 'I'"))

sb.lmplot(x="x", y="y", data=df1.query("dataset == 'II'"), order = 2)

plt.show()