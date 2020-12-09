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
from catboost import CatBoostRegressor
import houseprices_kaggle.lib.house_lib as hb
import warnings

SEED = 23

warnings.filterwarnings('ignore')
pd.set_option('max_columns', None)
pd.set_option('max_rows', None)
pd.set_option('display.width', 1000)
np.random.seed(SEED)


'''
sklearn.ensemble.StackingClassifier
Normally I would use this classifier, but I already got the regression results of 
each sub-classifiers.
'''
FILES = [ 
            'cat.csv', 'xgb.csv', 'lasso.csv', 
            'eleasticnet.csv', 'linear.csv', 'svm.csv',
            'pass_agg.csv', 'sgd.csv', 'tweedie.csv'
        ]

PROJECT_DIR=str(Path(__file__).parent.parent)  
orig_df = pd.read_csv(os.path.join(PROJECT_DIR, 'data/train.csv'))

train_df = pd.DataFrame()
test_df = pd.DataFrame()

first = True
for name in FILES:
    df = pd.read_csv(os.path.join(PROJECT_DIR, 'data/models/', name))
    df1 = df.loc[df['Id'] < 1461, ['Id', 'SalePrice']]
    df2 = df.loc[df['Id'] >= 1461, ['Id', 'SalePrice']]
    
    if first:
        train_df['Id'] = df1['Id']
        test_df['Id'] = df2['Id']
        
    train_df[name] = df1['SalePrice']
    test_df[name] = df2['SalePrice']
    
train_df['SalePrice'] = orig_df['SalePrice']

all_columns = FILES.copy()



scaler = RobustScaler()
train_df[all_columns] = scaler.fit_transform(train_df[all_columns])
test_df[all_columns] = scaler.transform(test_df[all_columns])  

'''
RMSE   : 5074.4396
CV RMSE: 8492.7761
Site   : 0.11935
'''    
model = CatBoostRegressor(random_seed=SEED, loss_function='RMSE', verbose=False)
model.fit(train_df[all_columns], train_df['SalePrice'])

train_df['Prediction'] = model.predict(train_df[all_columns])
test_df['SalePrice'] = model.predict(test_df[all_columns])


print("======================================================")
print("RMSE   : %0.4f" % np.sqrt(mean_squared_error(train_df['SalePrice'], train_df['Prediction'])))
print("CV RMSE: %0.4f" % hb.rmse_cv(CatBoostRegressor(random_seed=SEED, loss_function='RMSE', verbose=False),
                                    train_df[all_columns], train_df['Prediction'], 5))

test_df[['Id', 'SalePrice']].to_csv(os.path.join(PROJECT_DIR, 'data/results.csv'), index = False)

print("Done")