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


train_df["BuiltAge"] = train_df["YrSold"] - train_df["YearBuilt"]
train_df["RemodAge"] = train_df["YrSold"] - train_df["YearRemodAdd"]
train_df["Remodeled"] = train_df["YearBuilt"] != train_df["YearRemodAdd"]
train_df["BuiltAge"] = train_df["BuiltAge"].apply(lambda x: 0 if x < 0 else x)
train_df["RemodAge"] = train_df["RemodAge"].apply(lambda x: 0 if x < 0 else x)

test_df["BuiltAge"] = test_df["YrSold"] - test_df["YearBuilt"]
test_df["RemodAge"] = test_df["YrSold"] - test_df["YearRemodAdd"]
test_df["Remodeled"] = test_df["YearBuilt"] != test_df["YearRemodAdd"]
test_df["BuiltAge"] = test_df["BuiltAge"].apply(lambda x: 0 if x < 0 else x)
test_df["RemodAge"] = test_df["RemodAge"].apply(lambda x: 0 if x < 0 else x)

train_df['TotalBathrooms'] = (train_df['FullBath'] + (0.5 * train_df['HalfBath']) + train_df['BsmtFullBath'] + (0.5 * train_df['BsmtHalfBath']))
test_df['TotalBathrooms'] = (test_df['FullBath'] + (0.5 * test_df['HalfBath']) + test_df['BsmtFullBath'] + (0.5 * test_df['BsmtHalfBath']))
train_df['TotalPorchSF'] = (train_df['OpenPorchSF'] + train_df['3SsnPorch'] + train_df['EnclosedPorch'] + train_df['ScreenPorch'] + train_df['WoodDeckSF'])
test_df['TotalPorchSF'] = (test_df['OpenPorchSF'] + test_df['3SsnPorch'] + test_df['EnclosedPorch'] + test_df['ScreenPorch'] + test_df['WoodDeckSF'])

train_df['TotalSF'] = train_df['TotalBsmtSF'] + train_df['1stFlrSF'] + train_df['2ndFlrSF'] 
test_df['TotalSF'] = test_df['TotalBsmtSF'] + test_df['1stFlrSF'] + test_df['2ndFlrSF'] 

train_df['TotalHomeQuality'] = train_df['OverallQual'] + train_df['OverallCond']
test_df['TotalHomeQuality'] = test_df['OverallQual'] + test_df['OverallCond']

train_df["SqFtPerRoom"] = train_df["GrLivArea"] / (
                            train_df["TotRmsAbvGrd"]
                            + train_df["FullBath"]
                            + train_df["HalfBath"]
                            + train_df["KitchenAbvGr"]
                            )

test_df["SqFtPerRoom"] = test_df["GrLivArea"] / (
                        test_df["TotRmsAbvGrd"]
                        + test_df["FullBath"]
                        + test_df["HalfBath"]
                        + test_df["KitchenAbvGr"]
                        )

train_df['HasPool'] = train_df['PoolArea'].apply(lambda x: 1 if x > 0 else 0)
test_df['HasPool'] = test_df['PoolArea'].apply(lambda x: 1 if x > 0 else 0)
train_df['Has2ndFlr'] = train_df['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)
test_df['Has2ndFlr'] = test_df['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)
train_df['HasGarage'] = train_df['GarageArea'].apply(lambda x: 1 if x > 0 else 0)
test_df['HasGarage'] = test_df['GarageArea'].apply(lambda x: 1 if x > 0 else 0)
train_df['HasBsmt'] = train_df['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)
test_df['HasBsmt'] = test_df['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)
train_df['HasFireplace'] = train_df['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)
test_df['HasFireplace'] = test_df['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)

train_df['OtherRoom'] = train_df["TotRmsAbvGrd"] - train_df['KitchenAbvGr'] - train_df['BedroomAbvGr']
test_df['OtherRoom'] = test_df["TotRmsAbvGrd"] - test_df['KitchenAbvGr'] - test_df['BedroomAbvGr']

num_columns, cat_columns = hb.get_types(train_df)
num_columns.remove('Id')
num_columns.remove('SalePrice')


hb.deSkew(train_df, test_df, num_columns)




all_df = pd.concat([ train_df, test_df ], ignore_index = True)

encoder = hb.AutoEncoder()
encoder.fit(train_df)
train_df = encoder.transform(train_df)
test_df = encoder.transform(test_df)

scaler = RobustScaler()
all_df[num_columns] = scaler.fit_transform(all_df[num_columns])


all_df = pd.get_dummies(all_df, columns = cat_columns)

all_columns = all_df.columns.tolist()
all_columns.remove('Id')
all_columns.remove('SalePrice')


train_df = all_df[all_df['Id'] < 1461]
test_df = all_df[all_df['Id'] >= 1461]


'''
RMSE   : 24177.4567
CV RMSE: 19382.9478
Sit    : 0.12314
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


test_df[['Id', 'SalePrice']].to_csv(os.path.join(PROJECT_DIR, 'data/results.csv'), index = False)

