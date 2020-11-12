'''
Created on Nov. 11, 2020

@author: zollen
'''
import os
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import norm, skew, boxcox_normmax
from scipy.special import boxcox1p
import matplotlib.pyplot as plt
import seaborn as sb
import warnings

warnings.filterwarnings('ignore')

sb.set_style("whitegrid")

PROJECT_DIR=str(Path(__file__).parent.parent)  
train_df = pd.read_csv(os.path.join(PROJECT_DIR, 'data/train.csv'))


train_df['LogSalePrice'] = train_df['SalePrice'].apply(lambda x : np.log1p(x))
train_df['BoxSalePrice'] = boxcox1p(train_df['SalePrice'], boxcox_normmax(train_df['SalePrice']))
print("Skew('SalePrice'): ", skew(train_df['SalePrice'].values))
print("Skew(Log1p('SalePrice')): ", skew(train_df['LogSalePrice'].values))
print("Skew(BoxCox('SalePrice')): ", skew(train_df['BoxSalePrice'].values))

fig, (a1, a2, a3) = plt.subplots(1, 3)

fig.set_size_inches(12 , 4)

a1.set_title("SalePrice")
sb.distplot(train_df['SalePrice'], fit = norm, ax = a1)
a2.set_title("Log1p(SalePrice)")
sb.distplot(train_df['LogSalePrice'], fit = norm, ax = a2)
a3.set_title("BoxCox(SalePrice)")
sb.distplot(train_df['BoxSalePrice'], fit = norm, ax = a3)

plt.show()