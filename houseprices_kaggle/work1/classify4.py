'''
Created on Oct. 31, 2020

@author: zollen
'''

import os
from pathlib import Path
import numpy as np
import pandas as pd
import pprint
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import HuberRegressor
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

all_columns = numeric_columns + categorical_columns

all_df = pd.concat([ train_df, test_df ])   

encoder = hb.AutoEncoder()
all_df = encoder.fit_transform(all_df)  

all_df = pd.get_dummies(all_df, columns = categorical_columns)

categorical_columns = set(all_df.columns).symmetric_difference(numeric_columns + ['Id', 'SalePrice'])
categorical_columns = list(categorical_columns)
categorical_columns.sort(reverse = True)

train_df[categorical_columns] = all_df.loc[all_df['Id'].isin(train_df['Id']), categorical_columns]
test_df[categorical_columns] = all_df.loc[all_df['Id'].isin(test_df['Id']), categorical_columns]

all_columns = numeric_columns + categorical_columns

scaler = RobustScaler()
train_df[numeric_columns] = scaler.fit_transform(train_df[numeric_columns])
test_df[numeric_columns] = scaler.transform(test_df[numeric_columns])    

if False:
    params = {
                'alpha': [ 0.005, 0.006, 0.007, 0.008 ],
                'epsilon': [ 1.34, 1.35, 1.36, 1.37, 1.38 ],
                'max_iter': [ 17950, 17960, 17970, 17980, 17990, 18000, 18010, 18020 ]
            }
    
    optimizer = BayesSearchCV(
                estimator = HuberRegressor(), 
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
RMSE   : 25656.4708
CV RMSE: 21259.2641
Site   : 0.12914
'''
model = HuberRegressor(alpha = 0.006, epsilon = 1.35, max_iter = 17960)
model.fit(train_df[all_columns], train_df['SalePrice'])



train_df['Prediction'] = model.predict(train_df[all_columns])
test_df['SalePrice'] = model.predict(test_df[all_columns])

train_df['Prediction'] = train_df['Prediction'].apply(lambda x : np.expm1(x))  
train_df['SalePrice'] = train_df['SalePrice'].apply(lambda x : np.expm1(x))  
test_df['SalePrice'] = test_df['SalePrice'].apply(lambda x : np.expm1(x))


print("======================================================")
print("RMSE   : %0.4f" % np.sqrt(mean_squared_error(train_df['SalePrice'], train_df['Prediction'])))
print("CV RMSE: %0.4f" % hb.rmse_cv(HuberRegressor(alpha = 0.006, epsilon = 1.35, max_iter = 17960),
                                    train_df[all_columns], train_df['Prediction'], 5))


if False:
    test_df[['Id', 'SalePrice']].to_csv(os.path.join(PROJECT_DIR, 'data/results.csv'), index = False)
else:
    hb.write_result(os.path.join(PROJECT_DIR, 'data/models/huber.csv'), train_df, test_df)

