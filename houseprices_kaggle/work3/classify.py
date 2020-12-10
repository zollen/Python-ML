'''
Created on Dec. 8, 2020

@author: zollen
'''

import os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import ElasticNet
from catboost import CatBoostRegressor
import houseprices_kaggle.lib.house_lib as hb
import warnings

SEED = 23

warnings.filterwarnings('ignore')
pd.set_option('max_columns', None)
pd.set_option('max_rows', None)
pd.set_option('display.width', 1000)
pd.set_option('display.float_format', '{:,.2f}'.format)
np.random.seed(SEED)

PROJECT_DIR=str(Path(__file__).parent.parent)  
train_df = pd.read_csv(os.path.join(PROJECT_DIR, 'data/train_data.csv'))
test_df = pd.read_csv(os.path.join(PROJECT_DIR, 'data/test_data.csv'))


hb.feature_engineering2(train_df, test_df)

num_columns, cat_columns = hb.get_types(train_df)
num_columns.remove('Id')
num_columns.remove('SalePrice')


hb.deSkew(train_df, test_df, num_columns)




all_df = pd.concat([ train_df, test_df ], ignore_index = True)

encoder = hb.AutoEncoder()
all_df = encoder.fit_transform(all_df)


scaler = RobustScaler()
all_df[num_columns] = scaler.fit_transform(all_df[num_columns])


all_df = pd.get_dummies(all_df, columns = cat_columns)

all_columns = all_df.columns.tolist()
all_columns.remove('Id')
all_columns.remove('SalePrice')


train_df = all_df[all_df['Id'] < 1461]
test_df = all_df[all_df['Id'] >= 1461]


'''
RMSE   : 7266.3167
CV RMSE: 31564.1877
'''
#model = CatBoostRegressor(random_seed=SEED, loss_function='RMSE', verbose=False)
model = ElasticNet(alpha=0.005, l1_ratio=0.08, max_iter=10000)
model.fit(train_df[all_columns], train_df['SalePrice'])
train_df['Prediction'] = model.predict(train_df[all_columns])
test_df['SalePrice'] = model.predict(test_df[all_columns])

train_df['Prediction'] = train_df['Prediction'].apply(lambda x : np.expm1(x))  
train_df['SalePrice'] = train_df['SalePrice'].apply(lambda x : np.expm1(x))  
test_df['SalePrice'] = test_df['SalePrice'].apply(lambda x : np.expm1(x))
          

print("======================================================")
print("RMSE   : %0.4f" % np.sqrt(mean_squared_error(train_df['SalePrice'], train_df['Prediction'])))
print("CV RMSE: %0.4f" % hb.rmse_cv(ElasticNet(), train_df[all_columns], train_df['Prediction'], 5))



