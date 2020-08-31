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
        y = re.findall("[0-9]+", val)
        if len(y) == 0:
            y = [ 0 ]

        return x[0][0]
#        rec['Room'] = int(str(y[0]))
        
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
    
def enginneering(df):
    
    df['Title'] = df['Name'].apply(lambda x : re.search('[a-zA-Z]+\\.', x).group(0))
    df['Title'] = df.apply(map_title, axis = 1)

    df.drop(columns=[ 'Name', 'Ticket' ], inplace=True)

    df.loc[df['Embarked'].isna() == True, 'Embarked'] = 'C'
    df.loc[df['Fare'].isna() == True, 'Fare'] = 7.25
    
    
    for title in set(title_category.values()):
        df.loc[((df['Age'].isna() == True) & (df['Title'] == title) & (df['Pclass'] == 1)), 'Age'] = df.loc[((df['Age'].isna() == False) & (df['Title'] == title) & (df['Pclass'] == 1)), 'Age'].median()
        df.loc[((df['Age'].isna() == True) & (df['Title'] == title) & (df['Pclass'] == 2)), 'Age'] = df.loc[((df['Age'].isna() == False) & (df['Title'] == title) & (df['Pclass'] == 2)), 'Age'].median()
        df.loc[((df['Age'].isna() == True) & (df['Title'] == title) & (df['Pclass'] == 3)), 'Age'] = df.loc[((df['Age'].isna() == False) & (df['Title'] == title) & (df['Pclass'] == 3)), 'Age'].median()
    df['Age'] = df['Age'].astype('int32')
    
    df['Cabin'] = df['Cabin'].apply(captureCabin) 
    
     
    
#    df.drop(columns = [ 'Sex' ], inplace = True)

     
    return df


PROJECT_DIR=str(Path(__file__).parent.parent)  
train_df = pd.read_csv(os.path.join(PROJECT_DIR, 'data/train.csv'))
test_df = pd.read_csv(os.path.join(PROJECT_DIR, 'data/test.csv'))

train_df = enginneering(train_df)
test_df = enginneering(test_df)


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
        encoder = preprocessing.LabelEncoder()
        keys = np.union1d(train_df[name].unique(), test_df[name].unique())
    
        if len(keys) == 2:
            encoder = preprocessing.LabelBinarizer()
        
        encoder.fit(keys)
        train_df[name] = encoder.transform(train_df[name].values)
        test_df[name] = encoder.transform(test_df[name].values)

    train_df = pd.get_dummies(train_df, columns=categorical_columns)
    test_df = pd.get_dummies(test_df, columns=categorical_columns)
    
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
