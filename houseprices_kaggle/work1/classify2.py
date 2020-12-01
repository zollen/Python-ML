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
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Lasso
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

def rmse_cv(data, label, n_folds):
    kf = KFold(n_folds, shuffle=True, random_state=SEED).get_n_splits(data.values)
    rmse = np.sqrt(-1 * cross_val_score(Lasso(alpha=0.0005,max_iter=10000), 
                                  data.values, label, scoring="neg_mean_squared_error", cv = kf))
    return np.mean(rmse)



PROJECT_DIR=str(Path(__file__).parent.parent)  
train_df = pd.read_csv(os.path.join(PROJECT_DIR, 'data/train_data.csv'))
test_df = pd.read_csv(os.path.join(PROJECT_DIR, 'data/test_data.csv'))

'''
Adding BuiltAge, RemodAge and Remodeled
RMSE   : 22924.8292
CV RMSE: 12419.2189
Site   : 0.12107
'''
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

scaler = RobustScaler()
train_df[numeric_columns] = scaler.fit_transform(train_df[numeric_columns])
test_df[numeric_columns] = scaler.transform(test_df[numeric_columns])    


if False:
    params = {
                'alpha': [  0.0005, 0.0006, 0.0007, 0.0008, 0.001, 0.002, 0.003 ],
                'max_iter': [ 10070, 10080, 10090, 10100, 10110, 10120 ],
                'random_state': [ 13, 17, 41, 87 ]
            }
    
    optimizer = BayesSearchCV(
                estimator = Lasso(), 
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
    

# model = Lasso(alpha=0.0006,max_iter=10080, random_state = 41)
model = Lasso(alpha=0.0006,max_iter=10100, random_state = 17)
model.fit(train_df[all_columns], train_df['SalePrice'])



train_df['Prediction'] = model.predict(train_df[all_columns])
test_df['SalePrice'] = model.predict(test_df[all_columns])

train_df['Prediction'] = train_df['Prediction'].apply(lambda x : np.expm1(x))  
train_df['SalePrice'] = train_df['SalePrice'].apply(lambda x : np.expm1(x))  
test_df['SalePrice'] = test_df['SalePrice'].apply(lambda x : np.expm1(x))


print("======================================================")
print("RMSE   : %0.4f" % np.sqrt(mean_squared_error(train_df['SalePrice'], train_df['Prediction'])))
print("CV RMSE: %0.4f" % rmse_cv(train_df[all_columns], train_df['Prediction'], 5))


test_df[['Id', 'SalePrice']].to_csv(os.path.join(PROJECT_DIR, 'data/results.csv'), index = False)


