'''
Created on Jul. 31, 2020

@author: zollen
'''

import seaborn as sb
from matplotlib import pyplot as plt

sb.set_style("whitegrid")

df = sb.load_dataset('titanic')

print(df.head())

fig, (a1, a2, a3) = plt.subplots(1, 3)

fig.set_size_inches(10 , 8)

sb.barplot(x = "sex", y = "survived", hue = "class", data = df, ax = a1)
a1.set_title('barplot')

sb.countplot(x = "class", data = df, palette = "Reds", ax = a2);
a2.set_title('countplot')

sb.pointplot(x = "sex", y = "survived", hue = "class", data = df, ax = a3)
a3.set_title('pointplot')

plt.show()