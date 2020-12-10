'''
Created on Dec. 8, 2020

@author: zollen
'''

import os
from pathlib import Path
import numpy as np
import pandas as pd
import houseprices_kaggle.lib.house_lib as hb
import warnings

SEED = 23

warnings.filterwarnings('ignore')
pd.set_option('max_columns', None)
pd.set_option('max_rows', None)
pd.set_option('display.width', 1000)
np.random.seed(SEED)

ID_FIELD = ['Id']
LABEL_FIELD = ['SalePrice'] 

PROJECT_DIR=str(Path(__file__).parent.parent)  
train_df = pd.read_csv(os.path.join(PROJECT_DIR, 'data/train.csv'))
test_df = pd.read_csv(os.path.join(PROJECT_DIR, 'data/test.csv'))


all_df = pd.concat([ train_df, test_df ], ignore_index = True)

imputer = hb.MultStageImputer(ID_FIELD + LABEL_FIELD)
all_df = imputer.fit_transform(all_df)


train_df = all_df[all_df['Id'] < 1461]
test_df = all_df[all_df['Id'] >= 1461]

train_df.to_csv(os.path.join(PROJECT_DIR, 'data/train_data.csv'), index = False)
test_df.to_csv(os.path.join(PROJECT_DIR, 'data/test_data.csv'), index = False)

print("DONE")