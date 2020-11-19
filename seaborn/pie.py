'''
Created on Nov. 19, 2020

@author: zollen
'''

import os
from pathlib import Path
import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')
pd.set_option('max_columns', None)
pd.set_option('max_rows', None)
pd.set_option('display.width', 1000)
sb.set_style("whitegrid")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (16, 7))

data = [35, 25, 25, 15]
labels = ['Apple', 'Banana', 'Cherry', 'Date']
ax1.pie(data, labels = labels, startangle = 90)
ax1.set_title('First')

myexplode = [ 0.2, 0, 0, 0 ]
ax2.pie(data, labels = labels, startangle = 90, explode = myexplode)
ax1.set_title('Second')

plt.show()