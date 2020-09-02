'''
Created on Aug. 1, 2020

@author: zollen
'''

import os
from pathlib import Path
import numpy as np
import pandas as pd
import re
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
import seaborn as sb
from matplotlib import pyplot as plt

import warnings


warnings.filterwarnings('ignore')

pd.set_option('max_columns', None)
pd.set_option('max_rows', None)
np.random.seed(0)
sb.set_style('whitegrid')

label_column = [ 'Survived']
identity_columns = [ 'PassengerId', 'Ticket' ]
numeric_columns = [ 'Age', 'SibSp', 'Parch', 'Fare' ]
categorical_columns = [ 'Sex', 'Title', 'Pclass', 'Embarked', 'Cabin' ]
all_features_columns = numeric_columns + categorical_columns 

title_category = {
        "Capt.": "Army",
        "Col.": "Army",
        "Major.": "Army",
        "Jonkheer.": "Baronness",
        "Don.": "Baron",
        "Sir.": "Baron",
        "Dr.": "Doctor",
        "Rev.": "Clergy",
        "Countess.": "Baronness",
        "Dona.": "Baronness",
        "Mme.": "Mrs",
        "Mlle.": "Miss",
        "Ms.": "Mrs",
        "Mr.": "Mr",
        "Mrs.": "Mrs",
        "Miss.": "Miss",
        "Master.": "Master",
        "Lady.": "Baronness",
        "Girl.": "Girl",
        "Boy.": "Boy",
        "Nurse.": "Nurse",
        'GramPa.':'GramPa',
        'GramMa.': 'GramMa'
    }
    
def map_title(rec):
    title = title_category[rec['Title']]
    sex = rec['Sex']
    
    if title == 'Doctor' and sex == 'male':
        return 'Doctor'
    else:
        if title == 'Doctor' and sex == 'female':
            return 'Nurse'

    if str(rec['Age']) != 'nan' and (title == 'Miss' or title == 'Mrs' or title == 'Ms') and rec['Age'] < 16:
        return 'Girl'
    
    if str(rec['Age']) == 'nan' and (title == 'Miss' or title == 'Mrs' or title == 'Ms') and rec['Parch'] > 0:
        return 'Girl'
    
    if str(rec['Age']) == 'nan' and (title == 'Miss' or title == 'Mrs' or title == 'Ms') and rec['Age'] < 16:
        return 'Girl'
    
    if str(rec['Age']) != 'nan' and title == 'Mr' and rec['Age'] < 16:
        return 'Boy'
    
    if str(rec['Age']) == 'nan' and title == 'Mr' and rec['Parch'] > 0:
        return 'Boy'
    
    if str(rec['Age']) == 'nan' and title == 'Mr' and rec['Age'] < 16:
        return 'Boy'
        
    if str(rec['Age']) != 'nan' and title == 'Mr' and rec['Age'] >= 50:
        return 'GramPa'
    
    if str(rec['Age']) != 'nan' and (title == 'Miss' or title == 'Mrs' or title == 'Ms') and rec['Age'] >= 50:
        return 'GramMa' 
    
    
    return title

def captureCabin(val):
    
    if str(val) != 'nan':
        x = re.findall("[a-zA-Z]+[0-9]{1}", val)
        if len(x) == 0:
            x = re.findall("[a-zA-Z]{1}", val)
            
        return x[0][0]
        
    return val

def captureSize(val):
    
    if str(val) != 'nan':
        if val == 1:
            return 1

        if val == 2:
            return 2

        if val == 3 or val == 4:
            return 3
        else:
            return 5
    
def normalize(df, columns):
    pdf = df.copy()
        
    for name in columns:
        encoder = preprocessing.LabelEncoder()   
        keys = pdf[name].unique()

        if len(keys) == 2:
            encoder = preprocessing.LabelBinarizer()

        encoder.fit(keys)
        pdf[name] = encoder.transform(pdf[name].values)
            
    return pdf

def fill_by_regression(df_src, df_dest, name, columns):
 
    input_columns = columns
    predicted_columns = [ name ]

    withVal = df_src[df_src[name].isna() == False]
    withoutVal = df_src[df_src[name].isna() == True]
    
    cat_columns = set(input_columns).intersection(categorical_columns)
    
    df1 = normalize(withVal, cat_columns)
    df2 = normalize(withoutVal, cat_columns)
    
    model = LogisticRegression()
    model.fit(df1[input_columns], withVal[predicted_columns])

    preds = model.predict(df2[input_columns])
    preds = [ round(i, 0) for i in preds ]
    print("Predicted %s values: " % name)
    print(np.stack((withoutVal['PassengerId'], preds), axis=1))

    df_dest.loc[df_dest[name].isna() == True, name] = preds

def fill_by_classification(df_src, df_dest, name, columns):

    input_columns = columns
    predicted_columns = [ name ]

    withVal = df_src[df_src[name].isna() == False]
    withoutVal = df_dest[df_dest[name].isna() == True]
    
    cat_columns = set(input_columns).intersection(categorical_columns)

    df1 = normalize(withVal, cat_columns)
    df2 = normalize(withoutVal, cat_columns)
     
    model = LogisticRegression()
    model.fit(df1[input_columns], withVal[predicted_columns])
    preds = model.predict(df2[input_columns])
    
    df_dest.loc[df_dest[name].isna() == True, name] = preds
    
def enginneering(src_df, dest_df):
    
    src_df['Title'] = src_df['Name'].apply(lambda x : re.search('[a-zA-Z]+\\.', x).group(0))
    src_df['Title'] = src_df.apply(map_title, axis = 1)
    dest_df['Title'] = dest_df['Name'].apply(lambda x : re.search('[a-zA-Z]+\\.', x).group(0))
    dest_df['Title'] = dest_df.apply(map_title, axis = 1)

    src_df.loc[src_df['Embarked'].isna() == True, 'Embarked'] = 'S'
    src_df.loc[src_df['Fare'].isna() == True, 'Fare'] = 7.25
    dest_df.loc[dest_df['Embarked'].isna() == True, 'Embarked'] = 'S'
    dest_df.loc[dest_df['Fare'].isna() == True, 'Fare'] = 7.25
    
    ## 1. the medum should be calculated based on Sex, Pclass and Title
    ## All three features have the highest correlation with Age in the heatmap
    medians = src_df.groupby(['Title', 'Pclass', 'Sex'])['Age'].median()
     
    if True:  
        for index, value in medians.items():
            for df in [src_df, dest_df]:
                df.loc[(df['Age'].isna() == True) & (df['Title'] == index[0]) & 
                       (df['Pclass'] == index[1]) & (df['Sex'] == index[2]), 'Age'] = value  
    else:  
        src_df['Age'] = src_df.groupby(['Title', 'Sex', 'Pclass'])['Age'].apply(lambda x : x.fillna(x.median()))        
        dest_df['Age'] = dest_df.groupby(['Title', 'Sex', 'Pclass'])['Age'].apply(lambda x : x.fillna(x.median()))
                       
    src_df.loc[src_df['Age'] < 1, 'Age'] = 1
    src_df['Age'] = src_df['Age'].astype('int32')              
    dest_df.loc[dest_df['Age'] < 1, 'Age'] = 1
    
    ## 2. Binning the Age
    dest_df['Age'] = pd.qcut(dest_df['Age'], q = 9,
                             labels = [ 1, 2, 3, 4, 5, 6, 7, 8, 9 ])  
    dest_df['Age'] = dest_df['Age'].astype('int32')
    
      
    src_df['Cabin'] = src_df['Cabin'].apply(captureCabin) 
    dest_df['Cabin'] = dest_df['Cabin'].apply(captureCabin) 
    
    ## 3. Try maximun counts with Pclass for calculating Cabin
    counts = src_df.groupby(['Title', 'Pclass', 'Sex', 'Cabin'])['Cabin'].count()
    
    for index, value in counts.items():
        for df in [ src_df, dest_df ]:
            df.loc[(df['Cabin'].isna() == True) & (df['Title'] == index[0]) & 
                       (df['Pclass'] == index[1]) & (df['Sex'] == index[2]), 'Cabin'] = counts[index[0], index[1], index[2]].idxmax()      
    
    fill_by_classification(dest_df, dest_df, 'Cabin', [ 'Title', 'Pclass', 'Sex', 'Embarked', 'Age', 'Fare' ])
    
    dest_df.loc[dest_df['Cabin'] == 'T', 'Cabin'] = 'C'
    dest_df.loc[(dest_df['Cabin'] == 'B') | (dest_df['Cabin'] == 'C'), 'Cabin'] = 'A'
    dest_df.loc[(dest_df['Cabin'] == 'D') | (dest_df['Cabin'] == 'E'), 'Cabin'] = 'D'
    dest_df.loc[(dest_df['Cabin'] == 'F') | (dest_df['Cabin'] == 'G'), 'Cabin'] = 'G'
      
    ## 3. Family Size (SibSp + Parch + 1)column 'Alone', 'Small', 'Medium', 'Big'
    
    dest_df['Size'] = dest_df['SibSp'] + dest_df['Parch'] + 1
    dest_df['Size'] = dest_df['Size'].apply(captureSize)
    
    
    ## 4. Binning the Fare
#    dest_df.loc[dest_df['Fare'] < 7.854, 'Fare'] = 0
#    dest_df.loc[(dest_df['Fare'] >= 7.854) & (dest_df['Fare'] < 10.5), 'Fare'] = 7.854 
#    dest_df.loc[(dest_df['Fare'] >= 10.5) & (dest_df['Fare'] < 21.679), 'Fare'] = 10.5
#    dest_df.loc[(dest_df['Fare'] >= 21.679) & (dest_df['Fare'] < 39.688), 'Fare'] = 21.679
#    dest_df.loc[(dest_df['Fare'] >= 39.688) & (dest_df['Fare'] < 200.00), 'Fare'] = 39.688
#    dest_df.loc[(dest_df['Fare'] >= 200.00), 'Fare'] = 50

    
    ## 5. Try add a survivibility percentage column
   
    
    
    dest_df.drop(columns = [ 'Name', 'Ticket', 'Sex', 'SibSp', 'Parch' ], inplace = True)

     
    return dest_df


PROJECT_DIR=str(Path(__file__).parent.parent)  
train_df = pd.read_csv(os.path.join(PROJECT_DIR, 'data/train.csv'))
test_df = pd.read_csv(os.path.join(PROJECT_DIR, 'data/test.csv'))

all_df = pd.concat([ train_df, test_df ])
all_df1 = all_df.copy()
all_df2 = all_df.copy()

all_df1.set_index('PassengerId', inplace=True)
all_df2.set_index('PassengerId', inplace=True)


train_df = enginneering(all_df1, train_df)
test_df = enginneering(all_df2, test_df)


if False:
    dd = train_df[train_df['Cabin'].isna() == False]  
    sb.catplot(x = "Pclass", y = "Title", hue = "Cabin", kind = "swarm", data = dd)
    plt.show()
    exit()
    
if False:    
    dd = train_df[train_df['Cabin'].isna() == True]    
    sb.catplot(x = "Pclass", y = "Fare", hue = "Cabin", kind = "swarm", data = dd)
    plt.show()
    exit()
    
if False:
    for name in categorical_columns:
        if name == 'Cabin':
            continue
        keys = np.union1d(train_df[name].unique(), test_df[name].unique())            
        for key in keys:
            train_df[name + "." + str(key)] = train_df[name].apply(lambda x : 1 if x == key else 0)
          
    train_df.drop(columns = categorical_columns, inplace = True)
  
    dd = train_df.copy()
    dd.drop(columns=['PassengerId'], inplace=True)
    corr = dd.corr() 
  
    mask = np.triu(np.ones_like(corr, dtype=np.bool))    
    plt.figure(figsize=(14, 10))   
    sb.heatmap(corr, mask=mask, cmap='RdBu_r', annot=True, linewidths=0.5, fmt='0.2f')
    plt.show()
    exit()
    
if False:
    dd = train_df[train_df['Cabin'].isna() == False]
    for name in categorical_columns:
        keys = dd[name].unique()
        for key in keys:
            dd[name + "." + str(key)] = dd[name].apply(lambda x : 1 if x == key else 0)
    dd.drop(columns = categorical_columns, inplace = True)
    dd.drop(columns=['PassengerId'], inplace=True)
    corr = dd.corr()
     
    mask = np.triu(np.ones_like(corr, dtype=np.bool))    
    plt.figure(figsize=(16, 12))   
    sb.heatmap(corr, mask=mask, cmap='RdBu_r', annot=True, linewidths=0.5, fmt='0.2f')
    plt.show()
    exit()
    
    
if False:
    dd = train_df[train_df['Cabin'].isna() == True]
    sb.catplot(x = "Pclass", y = "Sex", hue = "Survived", kind = "swarm", data = dd)
    plt.show()
    exit()

train_df.to_csv(os.path.join(PROJECT_DIR, 'data/train_processed.csv'), index=False)
test_df.to_csv(os.path.join(PROJECT_DIR, 'data/test_processed.csv'), index=False)
