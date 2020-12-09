'''
Created on Dec. 9, 2020

@author: zollen
'''

import os
from pathlib import Path
import pandas as pd
import numpy as np
import houseprices_kaggle.lib.house_lib as hb

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import AdaBoostClassifier
import warnings


SEED = 23

warnings.filterwarnings('ignore')
pd.set_option('max_columns', None)
pd.set_option('max_rows', None)
pd.set_option('display.width', 1000)
np.random.seed(SEED)

PROJECT_DIR=str(Path(__file__).parent.parent)  
train_df = pd.read_csv(os.path.join(PROJECT_DIR, 'houseprices_kaggle/data/train.csv'))

col_types = train_df.columns.to_series().groupby(train_df.dtypes)
numeric_columns = []
catgeroical_columns = []
for col in col_types:
    if col[0] == 'object':
        categorical_columns = col[1].unique().tolist()
    else:
        numeric_columns += col[1].unique().tolist()
        
numeric_columns.remove('Id')
numeric_columns.remove('SalePrice')

all_columns = numeric_columns + categorical_columns



hb.auto_encode(train_df)


print(train_df.isnull().sum())

imputer1 = IterativeImputer(random_state = 0,
                            estimator = AdaBoostClassifier(random_state = 17, n_estimators = 100))

result_df = imputer1.fit_transform(train_df[all_columns])

print("==================================")
print(result_df.isnull().sum())

