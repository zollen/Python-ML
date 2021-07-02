'''
Created on Jul. 2, 2021

@author: zollen
'''

import pandas as pd
import numpy as np
from datetime import datetime
from matplotlib import pyplot as plt
import seaborn as sb
import warnings

warnings.filterwarnings('ignore')

pd.set_option('max_columns', None)
pd.set_option('max_rows', None)

np.random.seed(0)

sales = pd.read_csv('../data/monthly_train.csv')

sales['date'] = pd.to_datetime(sales['date'], format='%Y-%m-%d')

print(sales.head(1000))
