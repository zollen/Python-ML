'''
Created on Mar. 3, 2021

@author: zollen
@url: https://towardsdatascience.com/rip-correlation-introducing-the-predictive-power-score-3d90808b9598

Problem with Standard Correlation Matrix
----------------------------------------
Let’s take a moment to review the correlation. The score ranges from -1 to 1 and indicates if there is a 
strong linear relationship — either in a positive or negative direction. So far so good. However, t
here are many non-linear relationships that the score simply won’t detect. For example, a sinus wave, 
a quadratic curve or a mysterious step function. The score will just be 0, saying: “Nothing interesting 
here”. Also, correlation is only defined for numeric columns. So, let’s drop all the categoric columns. 

If you are a little bit too well educated you know that the correlation matrix is symmetric. So you 
basically can throw away one half of it. Great, we saved ourselves some work there! Or did we? Symmetry 
means that the correlation is the same whether you calculate the correlation of A and B or the 
correlation of B and A. However, relationships in the real world are rarely symmetric. More often, 
relationships are *asymmetric*. 

The last time I checked, my zip code of 60327 tells strangers quite reliably that I am living in 
Frankfurt, Germany. But when I only tell them my city, somehow they are never able to deduce the 
correct zip code.

Predictive Power Score (PPS) for addressing the shortcomings of Correlation Matrix
'''

import os
import random
from pathlib import Path
import pandas as pd
import numpy as np
import ppscore as pps
import seaborn as sb
from matplotlib import pyplot as plt
import warnings

warnings.filterwarnings('ignore')
pd.set_option('max_columns', None)
pd.set_option('max_rows', None)
pd.set_option('display.width', 1000)

PROJECT_DIR=str(Path(__file__).parent.parent)  
df = pd.read_csv(os.path.join(PROJECT_DIR, 'titanic_kaggle/data/train.csv'))
df.drop(columns = 'PassengerId', inplace = True)
corr = df.corr()

fig, (a1, a2) = plt.subplots(1, 2)
fig.set_size_inches(20, 7)

a1.set_title("Correlation Matrix")
sb.heatmap(corr, cmap='RdBu_r', annot=True, linewidths=0.5, fmt='0.2f', ax = a1)


for name in df.columns:
    c = pps.score(df, x = name, y = "Survived")
    print("PPS({}): {}".format(name, c['ppscore']))


dd = pps.matrix(df)[['x', 'y', 'ppscore']].pivot(columns='x', index='y', values='ppscore')

a2.set_title("Asymmetric PPS Matrix")
sb.heatmap(dd, cmap="Oranges", annot=True, linewidths = 0.5, fmt='0.2f', ax = a2)

plt.show()
