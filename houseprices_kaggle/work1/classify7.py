'''
Created on Oct. 31, 2020

@author: zollen
'''

import os
from pathlib import Path
import numpy as np
import pandas as pd
import pprint
from sklearn.preprocessing import MinMaxScaler 
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import make_scorer
from skopt import BayesSearchCV
import houseprices_kaggle.lib.house_lib as hb
import warnings

SEED = 87

warnings.filterwarnings('ignore')
pd.set_option('max_columns', None)
pd.set_option('max_rows', None)
pd.set_option('display.width', 1000)
np.random.seed(SEED)

pp = pprint.PrettyPrinter(indent=3) 


PROJECT_DIR=str(Path(__file__).parent.parent)  
train_df = pd.read_csv(os.path.join(PROJECT_DIR, 'data/train_data.csv'))
test_df = pd.read_csv(os.path.join(PROJECT_DIR, 'data/test_data.csv'))

'''
feature engineering
'''
hb.feature_engineering2(train_df, test_df)



'''
DeSkew numerical features
'''
col_types = train_df.columns.to_series().groupby(train_df.dtypes)
numeric_columns = []
       
for col in col_types:
    if col[0] != 'object':
        numeric_columns += col[1].unique().tolist()

numeric_columns.remove('Id')
numeric_columns.remove('SalePrice')
hb.deSkew(train_df, test_df, numeric_columns) 


col_types = train_df.columns.to_series().groupby(train_df.dtypes)
numeric_columns = []
for col in col_types:
    if col[0] == 'object':
        categorical_columns = col[1].unique().tolist()
    else:
        numeric_columns += col[1].unique().tolist()

numeric_columns.remove('Id')
numeric_columns.remove('SalePrice')

encoder = hb.AutoEncoder()
encoder.fit(train_df)
train_df = encoder.transform(train_df)
test_df = encoder.transform(test_df)

all_df = pd.concat([ train_df, test_df ]) 

all_df = pd.get_dummies(all_df, columns = categorical_columns)

categorical_columns = set(all_df.columns).symmetric_difference(numeric_columns + ['Id', 'SalePrice'])
categorical_columns = list(categorical_columns)
categorical_columns.sort(reverse = True)

train_df[categorical_columns] = all_df.loc[all_df['Id'].isin(train_df['Id']), categorical_columns]
test_df[categorical_columns] = all_df.loc[all_df['Id'].isin(test_df['Id']), categorical_columns]

all_columns = numeric_columns + categorical_columns

scaler = MinMaxScaler()
train_df[numeric_columns] = scaler.fit_transform(train_df[numeric_columns])
test_df[numeric_columns] = scaler.transform(test_df[numeric_columns])    

if False:
    params = {
                'max_iter': [ 900, 1400, 1450, 1480, 1500, 1520, 1550 ],
                'epsilon': [ 0.06, 0.08, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0 ],
                'random_state': [ 0, 1, 3, 5, 7, 9, 23, 27, 37, 43, 87 ],
                'penalty': [ 'l2' ],
                'alpha': [ 0.00003, 0.00004, 0.00005, 0.00006, 0.00007, 0.00008 ],
                'l1_ratio': [ 0.00, 0.15, 0.30, 0.45, 0.55, 0.60, 0.65, 0.75, 0.9, 1.0 ]
            }
    
    optimizer = BayesSearchCV(
                estimator = SGDRegressor(), 
                search_spaces = params,
                scoring = make_scorer(mean_squared_error, greater_is_better=False, needs_threshold=False),
                cv = KFold(n_splits=5, shuffle=True, random_state=0),
                n_jobs=20, 
                n_iter=200,
                return_train_score=False,
                refit=True,
                random_state = SEED)

    optimizer.fit(train_df[all_columns], train_df['SalePrice'])

    print("====================================================================================")
    print("Best Score: ", optimizer.best_score_)
    pp.pprint(optimizer.cv_results_)
    pp.pprint(optimizer.best_params_)

    exit()


'''
RMSE   : 32057.3395
CV RMSE: 18219.0892
Site   : 0.17630
'''
model = SGDRegressor()
model.fit(train_df[all_columns], train_df['SalePrice'])




train_df['Prediction'] = model.predict(train_df[all_columns])
test_df['SalePrice'] = model.predict(test_df[all_columns])

train_df['Prediction'] = train_df['Prediction'].apply(lambda x : np.expm1(x))  
train_df['SalePrice'] = train_df['SalePrice'].apply(lambda x : np.expm1(x))  
test_df['SalePrice'] = test_df['SalePrice'].apply(lambda x : np.expm1(x))


print("======================================================")
print("RMSE   : %0.4f" % np.sqrt(mean_squared_error(train_df['SalePrice'], train_df['Prediction'])))
print("CV RMSE: %0.4f" % hb.rmse_cv(SGDRegressor(), train_df[all_columns], train_df['Prediction'], 5))


if False:
    test_df[['Id', 'SalePrice']].to_csv(os.path.join(PROJECT_DIR, 'data/results.csv'), index = False)
else:
    hb.write_result(os.path.join(PROJECT_DIR, 'data/models/sgd.csv'), train_df, test_df)

