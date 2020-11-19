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

PROJECT_DIR=str(Path(__file__).parent.parent)  
train_df = pd.read_csv(os.path.join(PROJECT_DIR, 'houseprices_kaggle/data/train.csv'))

train_df = train_df[['YrSold', 'SaleCondition', 'SalePrice']]

fig, ax = plt.subplots(figsize = (16, 7))

sb.despine(left=True)
chart = sb.lineplot(x = 'YrSold', y = 'SalePrice', hue = 'SaleCondition', data = train_df, ax = ax)
chart.set_xlabel('Year Sold', weight = 'bold', fontsize = 13)
chart.set_ylabel('Prices', weight = 'bold', fontsize = 16)
chart.set_title("Demo Report")

plt.show()