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
from sklearn.linear_model import TweedieRegressor
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
Add TotalSF
RMSE   : 22911.4003
CV RMSE: 12273.8919
Site   : 0.12092
'''
train_df['TotalSF'] = train_df['TotalBsmtSF'] + train_df['1stFlrSF'] + train_df['2ndFlrSF']
test_df['TotalSF'] = test_df['TotalBsmtSF'] + test_df['1stFlrSF'] + test_df['2ndFlrSF']


'''
Add SqFtPerRoom
RMSE   : 22881.0458
CV RMSE: 12150.3098
Site   : 0.12091
'''
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


'''
Add SqFtPerRoom
RMSE   : 22874.7657
CV RMSE: 11951.6143
Site   : 0.12089
'''
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


'''
Add OtherRoom
RMSE   : 22814.8221
CV RMSE: 12090.8070
Site   : 0.12073
'''
train_df['OtherRoom'] = train_df["TotRmsAbvGrd"] - train_df['KitchenAbvGr'] - train_df['BedroomAbvGr']
test_df['OtherRoom'] = test_df["TotRmsAbvGrd"] - test_df['KitchenAbvGr'] - test_df['BedroomAbvGr']



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

#pca = PCA(n_components = 100)
#ttrain_df = pd.DataFrame(pca.fit_transform(train_df[all_columns]))
#ttest_df = pd.DataFrame(pca.transform(test_df[all_columns]))

if False:
    params = {
                'max_iter': [ 90, 100, 120, 140, 160, 180, 200, 250, 500, 1000, 9000 ],
                'link': [ 'auto'],
                'alpha': [ 50, 80, 100, 120, 150, 160, 170, 180, 190, 200],
                'power': [ 0 ]
            }
    
    optimizer = BayesSearchCV(
                estimator = TweedieRegressor(), 
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
RMSE   : 37241.0210
CV RMSE: 14070.1177
Site   : 0.12504
'''
model = TweedieRegressor(power = 0, alpha = 1, link = 'auto')
model.fit(train_df[all_columns], train_df['SalePrice'])




train_df['Prediction'] = model.predict(train_df[all_columns])
test_df['SalePrice'] = model.predict(test_df[all_columns])

train_df['Prediction'] = train_df['Prediction'].apply(lambda x : np.expm1(x))  
train_df['SalePrice'] = train_df['SalePrice'].apply(lambda x : np.expm1(x))  
test_df['SalePrice'] = test_df['SalePrice'].apply(lambda x : np.expm1(x))


print("======================================================")
print("RMSE   : %0.4f" % np.sqrt(mean_squared_error(train_df['SalePrice'], train_df['Prediction'])))
print("CV RMSE: %0.4f" % hb.rmse_cv(TweedieRegressor(power = 0, alpha = 1, link = 'auto'),
                                    train_df[all_columns], train_df['Prediction'], 5))


test_df[['Id', 'SalePrice']].to_csv(os.path.join(PROJECT_DIR, 'data/results.csv'), index = False)


