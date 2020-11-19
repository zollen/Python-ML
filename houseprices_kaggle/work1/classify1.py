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
from sklearn.model_selection import cross_val_score
from catboost import CatBoostRegressor
from sklearn.metrics import make_scorer
from skopt import BayesSearchCV
from skopt.space.space import Integer
import warnings

SEED = 87

warnings.filterwarnings('ignore')
pd.set_option('max_columns', None)
pd.set_option('max_rows', None)
pd.set_option('display.width', 1000)
np.random.seed(SEED)

pp = pprint.PrettyPrinter(indent=3) 

def rmse_cv(data, label, n_folds):
    kf = KFold(n_folds, shuffle=True, random_state=SEED).get_n_splits(data.values)
    rmse = np.sqrt(-1 * cross_val_score(CatBoostRegressor(random_seed=SEED, loss_function='RMSE', verbose = False), 
                                  data.values, label, scoring="neg_mean_squared_error", cv = kf))
    return np.mean(rmse)



PROJECT_DIR=str(Path(__file__).parent.parent)  
train_df = pd.read_csv(os.path.join(PROJECT_DIR, 'data/train_data.csv'))
test_df = pd.read_csv(os.path.join(PROJECT_DIR, 'data/test_data.csv'))





col_types = train_df.columns.to_series().groupby(train_df.dtypes)
numeric_columns = []
for col in col_types:
    if col[0] == 'object':
        categorical_columns = col[1].unique().tolist()
    else:
        numeric_columns += col[1].unique().tolist()


for name in categorical_columns:   
    keys = train_df[name].unique().tolist()
        
    if np.nan in keys:
        keys.remove(np.nan)
    
    vals = [ i  for i in range(0, len(keys))]
    labs = dict(zip(keys, vals))
    train_df[name] = train_df[name].map(labs)
    test_df[name] = test_df[name].map(labs)

numeric_columns.remove('Id')
numeric_columns.remove('SalePrice')

all_columns = numeric_columns + categorical_columns

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

'''
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Huber
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import ExtraTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor
from catboost import CatBoostRegressor
from sklearn.decomposition import PCA
'''

#pca = PCA(n_components = 5)
#ttrain_df = pd.DataFrame(pca.fit_transform(train_df[all_columns]))
#ttest_df = pd.DataFrame(pca.transform(test_df[all_columns]))

if False:
    params = {
                'iterations': [ 150, 200, 250, 300, 350, 400, 450, 
                               500, 550, 600, 650 ],
                'depth': Integer(1, 16),
                'learning_rate': [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 
                                  0.01, 0.02, 0.03, 0.04, 0.05, 0.06 ],
                'bagging_temperature': [0.0, 0.1, 0.2, 0.03, 0.4, 0.5, 
                                        0.6, 0.7, 0.8, 0.9, 1.0],
                'border_count': Integer(1, 128),             
                'l2_leaf_reg': Integer(2, 30)
            }
    
    optimizer = BayesSearchCV(
                estimator = CatBoostRegressor(random_seed=SEED, 
                                              loss_function='RMSE', 
                                              verbose=False), 
                search_spaces = params,
                scoring = make_scorer(mean_squared_error, greater_is_better=False, needs_threshold=False),
                cv = KFold(n_splits=5, shuffle=True, random_state=0),
                n_jobs=20, 
                n_iter=100,
                return_train_score=False,
                refit=True,
                random_state = SEED)

    optimizer.fit(train_df[all_columns], train_df['SalePrice'])

    print("====================================================================================")
    print("Best Score: ", optimizer.best_score_)
    pp.pprint(optimizer.cv_results_)
    pp.pprint(optimizer.best_params_)

    exit()
    
#model = CatBoostRegressor(random_seed=SEED, 
#                          loss_function='RMSE', 
#                          verbose=False,
#                          bagging_temperature = 1.0,
#                          depth = 7,
#                          iterations = 550,
#                          learning_rate = 0.04,
#                          border_count = 113,
#                          l2_leaf_reg = 2)
    
model = CatBoostRegressor(random_seed=SEED, loss_function='RMSE', verbose=False)
model.fit(train_df[all_columns], train_df['SalePrice'])



train_df['Prediction'] = model.predict(train_df[all_columns])
test_df['SalePrice'] = model.predict(test_df[all_columns])

train_df['Prediction'] = train_df['Prediction'].apply(lambda x : np.expm1(x))  
train_df['SalePrice'] = train_df['SalePrice'].apply(lambda x : np.expm1(x))  
test_df['SalePrice'] = test_df['SalePrice'].apply(lambda x : np.expm1(x))


print("======================================================")
print("RMSE   : %0.4f" % np.sqrt(mean_squared_error(train_df['SalePrice'], train_df['Prediction'])))
if False:
    print("CV RMSE: %0.4f" % rmse_cv(train_df[all_columns], train_df['Prediction'], 5))


test_df[['Id', 'SalePrice']].to_csv(os.path.join(PROJECT_DIR, 'data/results.csv'), index = False)


