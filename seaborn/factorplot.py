'''
Created on Jul. 31, 2020

@author: zollen
'''
import pandas as pd
import seaborn as sb
from matplotlib import pyplot as plt

pd.set_option('max_columns', None)

df = sb.load_dataset('exercise')

print(df.head())

sb.factorplot(x = "time", y = "pulse", hue = "kind", data = df);

sb.factorplot(x = "time", y = "pulse", hue = "kind", kind = 'violin', data = df);

sb.factorplot(x = "time", y = "pulse", hue = "kind", kind = 'point', col = "diet", data = df);

df1 = sb.load_dataset('titanic')

print(df1.head())

sb.factorplot("alive", col = "deck", col_wrap = 3,data = df1[df1.deck.notnull()], kind = "count")

plt.show()