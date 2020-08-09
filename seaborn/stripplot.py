'''
Created on Jul. 31, 2020

@author: zollen
'''
import seaborn as sb
from matplotlib import pyplot as plt

sb.set_style("whitegrid")

df = sb.load_dataset('iris')

print(df.head())

fig, (a1, a2, a3) = plt.subplots(1, 3)

fig.set_size_inches(10 , 8)

sb.stripplot(x = "species", y = "petal_length", data = df, ax = a1)
a1.set_title("with jitter")

sb.stripplot(x = "species", y = "petal_length", data = df, jitter = False, ax = a2)
a2.set_title("without jitter")

sb.swarmplot(x = "species", y = "petal_length", data = df, ax= a3)
a3.set_title('swampplot')


plt.show()