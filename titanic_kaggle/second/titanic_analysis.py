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
categorical_columns = [ 'Title', 'Pclass', 'Embarked' ]
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
    withoutVal = df_src[df_src[name].isna() == True]
    
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


    dest_df.loc[dest_df['Embarked'].isna() == True, 'Embarked'] = 'S'
    dest_df.loc[dest_df['Fare'].isna() == True, 'Fare'] = 7.25
    
    ## the medum should be calculated based on Sex, Pclass and Title
    ## All three features have the highest correlation in the heatmap
    medians = src_df.groupby(['Title', 'Pclass', 'Sex'])['Age'].median()
       
    for index, value in medians.items():
        dest_df.loc[(dest_df['Age'].isna() == True) & (dest_df['Title'] == index[0]) & 
               (dest_df['Pclass'] == index[1]) & (dest_df['Sex'] == index[2]), 'Age'] = value
               
    dest_df.loc[dest_df['Age'] < 1, 'Age'] = 1
    
    dest_df['Age'] = dest_df['Age'].astype('int32')
      
    
    dest_df['Cabin'] = dest_df['Cabin'].apply(captureCabin) 
     
    
    dest_df.drop(columns = [ 'Name', 'Ticket', 'Sex' ], inplace = True)

     
    return dest_df


PROJECT_DIR=str(Path(__file__).parent.parent)  
train_df = pd.read_csv(os.path.join(PROJECT_DIR, 'data/train.csv'))
test_df = pd.read_csv(os.path.join(PROJECT_DIR, 'data/test.csv'))

all_df = pd.concat([ train_df, test_df ])
train_df = enginneering(all_df, train_df)
test_df = enginneering(test_df, test_df)



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
        keys = np.union1d(train_df[name].unique(), test_df[name].unique())               
        for key in keys:
            train_df[name + "." + str(key)] = train_df[name].apply(lambda x : 1 if x == key else 0)
            test_df[name + "." + str(key)] = test_df[name].apply(lambda x : 1 if x == key else 0)
        
    train_df.drop(columns = categorical_columns, inplace = True)
    test_df.drop(columns = categorical_columns, inplace = True)
    
    dd = train_df.copy()
    dd.drop(columns=['PassengerId'], inplace=True)
    corr = dd.corr() 
  
    mask = np.triu(np.ones_like(corr, dtype=np.bool))    
    plt.figure(figsize=(14, 10))   
    sb.heatmap(corr, mask=mask, cmap='RdBu_r', annot=True, linewidths=0.5, fmt='0.2f')
    plt.show()
    exit()
    
    
    

print(train_df.head())

train_df.to_csv(os.path.join(PROJECT_DIR, 'data/train_processed.csv'), index=False)
test_df.to_csv(os.path.join(PROJECT_DIR, 'data/test_processed.csv'), index=False)
