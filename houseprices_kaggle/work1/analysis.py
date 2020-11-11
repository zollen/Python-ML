'''
Created on Nov. 1, 2020

@author: zollen
@url: https://www.kaggle.com/toyox2020/house-prices-comprehensive-eda-visualization
'''

import os
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import skew
import pprint
from sklearn.preprocessing import LabelEncoder
from scipy.special import boxcox1p
from lightgbm import LGBMRegressor
import seaborn as sb
from matplotlib import pyplot as plt
import warnings

warnings.filterwarnings('ignore')
pd.set_option('max_columns', None)
pd.set_option('max_rows', None)
pd.set_option('display.width', 1000)

pp = pprint.PrettyPrinter(indent=3) 

PROJECT_DIR=str(Path(__file__).parent.parent)  
train_df = pd.read_csv(os.path.join(PROJECT_DIR, 'data/train.csv'))
test_df = pd.read_csv(os.path.join(PROJECT_DIR, 'data/test.csv'))

print(train_df.info())
print(train_df.describe())
print(train_df.head())
print(train_df.isnull().sum().sort_values(ascending=False))

tmp_df = train_df.drop(columns = ['SalePrice', 'Id'])

col_types = tmp_df.columns.to_series().groupby(tmp_df.dtypes)
numeric_columns = []
for col in col_types:
    if col[0] == 'object':
        categorical_columns = col[1].unique()
    else:
        numeric_columns += col[1].unique().tolist()



tmp_df[categorical_columns] = tmp_df[categorical_columns].fillna('NULL')

for col in categorical_columns:
    encoder = LabelEncoder()
    encoder.fit(tmp_df[col])
    tmp_df[col] = encoder.transform(tmp_df[col])


model = LGBMRegressor()
model.fit(tmp_df, train_df['SalePrice'])
feats = pd.DataFrame({'importance': model.feature_importances_}, index=tmp_df.columns).sort_values('importance', ascending=False)
feats['dtype'] = ['numeric' if feat in numeric_columns else 'categorical' for feat in tmp_df.columns]
num_train = train_df[numeric_columns + ['SalePrice']]

if False:
    fig, ax = plt.subplots(figsize=(20,10))
    bars = ax.bar(feats.index, feats['importance'])
    leg1 = ax.legend(['categorical'], loc=(0.9, 0.9))
    fig.gca().add_artist(leg1)

    # set labels ans title
    ax.set_xlabel('features')
    ax.set_ylabel('importance')
    ax.set_xticklabels(feats.index, rotation=90)
    ax.set_title('Feature importances', fontsize=16)
    # change color of numeric columns
    for i in np.where(feats['dtype'].values=='numeric')[0]:
        bars[i].set_color('red')
    leg2 = ax.legend(['numeric'], loc=(0.9, 0.85));


if False:
    num_feat_importances = feats.loc[numeric_columns].sort_values('importance', ascending=False)
    cols = num_feat_importances.index
    fig, ax = plt.subplots(-(-len(cols)//4), 4, figsize=(14, len(cols)/1.2))
 
    for idx,col in enumerate(cols):
        # show regplot with Sale Price
        row = idx // 4
        col = idx % 4
        sb.regplot(data=num_train, x=cols[idx], y='SalePrice', ax=ax[row][col])
    
        # show correlation coefficient and feature importance
        corr = num_train['SalePrice'].corr(num_train[cols[idx]])
        feat_imp = num_feat_importances.loc[cols[idx], 'importance']
        ax[row][col].set_title(f'corr: {corr:.3f}, importance: {feat_imp}')

        # display scale and label only on the left edge
        if col != 0:
            ax[row][col].set_ylabel('')
            ax[row][col].set_yticklabels('')

    fig.tight_layout()

if True:
    '''
    skew_features = features[numerics2].apply(lambda x: skew(x)).sort_values(ascending=False)

    high_skew = skew_features[skew_features > 0.5]
    skew_index = high_skew.index

    for i in skew_index:
        features[i] = boxcox1p(features[i], boxcox_normmax(features[i] + 1))
    '''
    log_num = np.log1p(num_train)
    box_num = boxcox1p(num_train, 0.1)
    # compare skewnesses original with after of logarithm
    skewness = pd.concat([
                        num_train.apply(lambda x: skew(x.dropna())),
                        log_num.apply(lambda x: skew(x.dropna())),
                        box_num.apply(lambda x: skew(x.dropna()))      
                        ],
                     axis=1).rename(columns={0:'original', 1:'logarithmization', 2:'boxcox'}).sort_values('original')
 
    ax = skewness.plot.barh(figsize=(15,12), title='Comparison of skewness of original and logarithmized', width=0.8)
    ax.set_xlabel('skewness');

if False:
    def spearman(frame, features, target = 'SalePrice'):
        spr = pd.DataFrame()
        spr['feature'] = features
        spr['spearman'] = [frame[f].corr(frame[target], 'spearman') for f in features]
        spr = spr.sort_values('spearman', ascending = False)
        plt.figure(figsize=(10, 0.25*len(features)))
        sb.barplot(data=spr, y='feature', x='spearman', orient='h')


    spearman(num_train, numeric_columns)
    



plt.show()
