'''
Created on Aug. 1, 2020

@author: zollen
'''

import os
import re
from pathlib import Path
import numpy as np
import pandas as pd
import pprint
import seaborn as sb
import warnings
import titanic_kaggle.lib.titanic_lib as tb
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesRegressor, ExtraTreesClassifier

warnings.filterwarnings('ignore')

label_column = [ 'Survived']
identity_columns = [ 'PassengerId', 'Ticket' ]
numeric_columns = [ 'Age', 'Fare' ]
categorical_columns = [ 'Sex', 'Title', 'Pclass', 'Embarked', 'Cabin' ]
all_features_columns = numeric_columns + categorical_columns 



def fill_by_regression(df_src, df_dest, name, columns):
 
    input_columns = columns
    predicted_columns = [ name ]

    cat_columns = set(input_columns).intersection(categorical_columns)
    
    df1 = tb.normalize({}, df_src, cat_columns)
    df2 = tb.normalize({}, df_dest, cat_columns)
       
    model = ExtraTreesRegressor(random_state = 0)
    model.fit(df1[input_columns], df_src[predicted_columns])

    preds = model.predict(df2[input_columns])
    preds = [ round(i, 0) for i in preds ]

    return preds

def fill_by_classification(df_src, df_dest, name, columns):

    input_columns = columns
    predicted_columns = [ name ]

    cat_columns = set(input_columns).intersection(categorical_columns)

    df1 = tb.normalize({}, df_src, cat_columns)
    df2 = tb.normalize({}, df_dest, cat_columns)
     
    model = ExtraTreesClassifier(random_state = 0)
    model.fit(df1[input_columns], df_src[predicted_columns])
    preds = model.predict(df2[input_columns])
    
    return preds
            
def capturePreTitle(rec):
    name = rec['Name']
    sex = rec['Sex']
    name = re.search('[a-zA-Z]+\\.', name).group(0)
    name = name.replace(".", "")
    if name == 'Col' or name == 'Capt' or name  == 'Major':
        name = 'Army'
    elif name == 'Rev' and sex == 'male':
        return 'Clergy'
    elif name == 'Dr' and sex == 'female':
        name = 'Mrs'
    elif name == 'Dr' and sex == 'male':
        return 'Doctor'
    elif name == 'Sir' or name == 'Don':
        name = 'Baron'
    elif name == 'Lady' or name == 'Countess' or name == 'Dona':
        return 'Baronness'
    elif name == 'Mme':
        return 'Mrs'
    elif name == 'Miss' or name == 'Ms' or name == 'Mlle' or name == 'Jonkheer':
        return 'Miss'
    
    return name
      
def capturePostTitle(rec):
    title = rec['Title']
    sex = rec['Sex']
    age = rec['Age']
    
    if sex == 'male' and age < 16:
        return 'Master'
    elif sex == 'female' and age < 16:
        return 'Girl'
    elif age >= 55 and sex == 'male':
        return 'GramPa'
    elif age >= 55 and sex == 'female':
        return 'GramMa'
    
    
    return title

fareBinned = {}
def calFare(df):
         
    last = 0
    for fare in [ 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170 ]:
        alivesBoy = len(df[(df['Fare'] >= last) & 
                             (df['Fare'] < fare) & 
                             (df['Sex'] == 0) &
                             (df['Survived'] == 1)])
        deadsBoy = len(df[(df['Fare'] >= last) & 
                            (df['Fare'] < fare) &
                            (df['Sex'] == 0) & 
                            (df['Survived'] == 0)])
        alivesGirl = len(df[(df['Fare'] >= last) & 
                             (df['Fare'] < fare) & 
                             (df['Sex'] == 1) &
                             (df['Survived'] == 1)])
        deadsGirl = len(df[(df['Fare'] >= last) & 
                            (df['Fare'] < fare) &
                            (df['Sex'] == 1) & 
                            (df['Survived'] == 0)])
    
        ratioBoy = 0 if alivesBoy + deadsBoy == 0 else alivesBoy / (alivesBoy + deadsBoy)
        ratioGirl = 0 if alivesGirl + deadsGirl == 0 else alivesGirl / (alivesGirl + deadsGirl)
#        print("[%2d, %2d]: Male {%2d, %2d} ==> [%0.4f], Female {%2d, %2d} ==> [%0.4f]" % 
#              (last, fare, alivesBoy, deadsBoy, ratioBoy, alivesGirl, deadsGirl, ratioGirl))
        
        fareBinned[str(last) + ":" + str(fare) + ":male"] = round(ratioBoy, 4)
        fareBinned[str(last) + ":" + str(fare) + ":female"] = round(ratioGirl, 4)        
        last = fare
        
def binFare(df):
    
    last = 0
    for fare in [ 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170]:
        male = str(last) + ":" + str(fare) + ":male"
        female = str(last) + ":" + str(fare) + ":female"
        df.loc[(df['Fare'] >= last) & (df['Fare'] < fare) & (df['Sex'] == 0), 'FareP'] = fareBinned[male]
        df.loc[(df['Fare'] >= last) & (df['Fare'] < fare) & (df['Sex'] == 1), 'FareP'] = fareBinned[female]
        last = fare
    
    df.loc[(df['Fare'] >= 170) & (df['Sex'] == 0), 'FareP'] = fareBinned[male]
    df.loc[(df['Fare'] >= 170) & (df['Sex'] == 1), 'FareP'] = fareBinned[female]
    
    df.loc[(df['Fare'] >= 200) & (df['Sex'] == 0), 'FareP'] = 0.0
    df.loc[(df['Fare'] >= 200) & (df['Sex'] == 1), 'FareP'] = 1.0
    
    df.loc[(df['Fare'] >= 500) & (df['Sex'] == 0), 'FareP'] = 1.0
    df.loc[(df['Fare'] >= 500) & (df['Sex'] == 1), 'FareP'] = 1.0
    
    df['Fare'] = df['FareP']
    df.drop(columns = ['FareP'], inplace = True)   
    
ageBinned = {}
def calAge(df):
         
    last = 0
    for age in [ 5, 10, 15, 20, 25, 30, 35, 40, 50, 55, 60, 65, 70, 75, 80 ]:
        alivesBoy = len(df[(df['Age'] >= last) & 
                             (df['Age'] < age) & 
                             (df['Sex'] == 0) &
                             (df['Survived'] == 1)])
        deadsBoy = len(df[(df['Age'] >= last) & 
                            (df['Age'] < age) &
                            (df['Sex'] == 0) & 
                            (df['Survived'] == 0)])
        alivesGirl = len(df[(df['Age'] >= last) & 
                             (df['Age'] < age) & 
                             (df['Sex'] == 1) &
                             (df['Survived'] == 1)])
        deadsGirl = len(df[(df['Age'] >= last) & 
                            (df['Age'] < age) &
                            (df['Sex'] == 1) & 
                            (df['Survived'] == 0)])
    
        ratioBoy = 0 if alivesBoy + deadsBoy == 0 else alivesBoy / (alivesBoy + deadsBoy)
        ratioGirl = 0 if alivesGirl + deadsGirl == 0 else alivesGirl / (alivesGirl + deadsGirl)
#        print("[%2d, %2d]: Male {%2d, %2d} ==> [%0.4f], Female {%2d, %2d} ==> [%0.4f]" % 
#              (last, age, alivesBoy, deadsBoy, ratioBoy, alivesGirl, deadsGirl, ratioGirl))
        
        ageBinned[str(last) + ":" + str(age) + ":male"] = round(ratioBoy, 4)
        ageBinned[str(last) + ":" + str(age) + ":female"] = round(ratioGirl, 4)
        last = age
        
def binAge(df):
    
    last = 0
    for age in [ 5, 10, 15, 20, 25, 30, 35, 40, 50, 55, 60, 65, 70, 75, 80 ]:
        male = str(last) + ":" + str(age) + ":male"
        female = str(last) + ":" + str(age) + ":female"
        df.loc[(df['Age'] >= last) & (df['Age'] < age) & (df['Sex'] == 0), 'AgeP'] = ageBinned[male]
        df.loc[(df['Age'] >= last) & (df['Age'] < age) & (df['Sex'] == 1), 'AgeP'] = ageBinned[female]
        last = age
    
    df.loc[(df['Age'] >= 80) & (df['Sex'] == 0), 'AgeP'] = ageBinned[male]
    df.loc[(df['Age'] >= 80) & (df['Sex'] == 1), 'AgeP'] = ageBinned[female]
    
    df['Age'] = df['AgeP']
    df.drop(columns = ['AgeP'], inplace = True)   


def prepare(src_df, dest_df):
    
    global categorical_columns
    
    src_df = pd.get_dummies(src_df, columns = categorical_columns)
    dest_df = pd.get_dummies(dest_df, columns = categorical_columns)

    train_uni = set(dest_df.columns).symmetric_difference(numeric_columns + ['Name', 'PassengerId'] + label_column)

    cat_columns = list(train_uni)
    
    all__columns = numeric_columns + cat_columns 
     
    scaler = MinMaxScaler()
    src_df[numeric_columns] = scaler.fit_transform(src_df[numeric_columns])
    dest_df[numeric_columns] = scaler.transform(dest_df[numeric_columns])
    
    return src_df, dest_df, all__columns
    
def reenigneeringRegressor(model, name, src_df, dest_df, pred = False):
    
    df1 = src_df.copy()
    df2 = dest_df.copy()
     
    df1, df2, all_columns = prepare(df1, df2)
    
    if 'Survived' in all_columns:
        all_columns.remove('Survived')
    
    model.fit(df1[all_columns], df1['Survived'].squeeze())
    if pred == False:
        dest_df[name] = model.predict_proba(df2[all_columns])[:, 0]
    else:
        dest_df[name] = model.predict(df2[all_columns])
        
    dest_df[name] = dest_df[name].round(4)
    


            
pd.set_option('max_columns', None)
pd.set_option('max_rows', None)
pd.set_option('display.width', 1000)
np.random.seed(0)
sb.set_style('whitegrid')
pp = pprint.PrettyPrinter(indent=3) 



PROJECT_DIR=str(Path(__file__).parent.parent)  
train_df = pd.read_csv(os.path.join(PROJECT_DIR, 'data/train.csv'))
test_df = pd.read_csv(os.path.join(PROJECT_DIR, 'data/test.csv'))

train_df.loc[train_df['Cabin'] == 'T', 'Cabin'] = 'A'
train_df.loc[train_df['Embarked'].isna() == True, 'Embarked'] = 'S'
train_df.loc[train_df['Fare'].isna() == True, 'Fare'] = 7.25
test_df.loc[test_df['Embarked'].isna() == True, 'Embarked'] = 'S' 
test_df.loc[test_df['Fare'].isna() == True, 'Fare'] = 7.25

## Extract Title from Name
train_df['Title'] = train_df.apply(capturePreTitle, axis = 1)
test_df['Title'] = test_df.apply(capturePreTitle, axis = 1)

## Fill missing Age
res = fill_by_regression(train_df[train_df['Age'].isna() == False], 
                   train_df[train_df['Age'].isna() == True], 'Age', 
                   [ 'Title', 'Survived', 'SibSp', 'Parch', 'Fare', 'Sex', 'Pclass', 'Embarked' ])

train_df.loc[train_df['Age'].isna() == True, 'Age'] = res

res = fill_by_regression(tb.combine(train_df, test_df[test_df['Age'].isna() == False]), 
                   test_df[test_df['Age'].isna() == True], 'Age', 
                   [ 'Title', 'SibSp', 'Parch', 'Fare', 'Sex', 'Pclass', 'Embarked' ])

test_df.loc[test_df['Age'].isna() == True, 'Age'] = res

## Readjust Title
train_df['Title'] = train_df.apply(capturePostTitle, axis = 1)
test_df['Title'] = test_df.apply(capturePostTitle, axis = 1)

train_df['Sex'] = train_df['Sex'].map(tb.sexes)
test_df['Sex'] = test_df['Sex'].map(tb.sexes)

train_df['Embarked'] = train_df['Embarked'].map(tb.embarkeds)
test_df['Embarked'] = test_df['Embarked'].map(tb.embarkeds)

## Fill missing Cabin
train_df['Cabin'] = train_df['Cabin'].apply(tb.captureCabin) 
test_df['Cabin'] = test_df['Cabin'].apply(tb.captureCabin) 

res = fill_by_classification(train_df[train_df['Cabin'].isna() == False], 
                       train_df[train_df['Cabin'].isna() == True], 'Cabin', 
                       [ 'Title', 'Survived', 'SibSp', 'Parch', 'Fare', 'Sex', 'Age', 'Pclass', 'Embarked' ])

train_df.loc[train_df['Cabin'].isna() == True, 'Cabin'] = res

res = fill_by_classification(tb.combine(train_df, test_df[test_df['Cabin'].isna() == False]),
                        test_df[test_df['Cabin'].isna() == True], 'Cabin', 
                        [ 'Title', 'SibSp', 'Parch', 'Fare', 'Sex', 'Age', 'Pclass', 'Embarked' ])

test_df.loc[test_df['Cabin'].isna() == True, 'Cabin'] = res

train_df['Cabin'] = train_df['Cabin'].map(tb.cabins)
test_df['Cabin'] = test_df['Cabin'].map(tb.cabins)

train_df['Ticket'] = train_df['Ticket'].apply(tb.captureTicketId)
test_df['Ticket'] = test_df['Ticket'].apply(tb.captureTicketId)

train_df['Ticket'] = np.log(train_df['Ticket'])
test_df['Ticket'] = np.log(test_df['Ticket'])

train_df['Ticket'] = train_df['Ticket'].round(4)
test_df['Ticket'] = test_df['Ticket'].round(4)

train_df['Fare'] = train_df['Fare'].round(2)
test_df['Fare'] = test_df['Fare'].round(2)

train_df['Title'] = train_df['Title'].map(tb.titles)
test_df['Title'] = test_df['Title'].map(tb.titles)

train_df['Size'] = train_df['Parch'] + train_df['SibSp'] + 1
test_df['Size'] = test_df['Parch'] + test_df['SibSp'] + 1


calAge(train_df)
binAge(train_df)
binAge(test_df)

calFare(train_df)
binFare(train_df)
binFare(test_df)




reenigneeringRegressor(LogisticRegression(max_iter=500, solver='lbfgs'), 
                       "Logistic", train_df, train_df)
reenigneeringRegressor(LogisticRegression(max_iter=500, solver='lbfgs'), 
                       "Logistic", train_df, test_df)






train_df.drop(columns = ['Name', 'Parch', 'SibSp'], inplace = True)
test_df.drop(columns = ['Name', 'Parch', 'SibSp'], inplace = True)


train_df.to_csv(os.path.join(PROJECT_DIR, 'data/train_processed.csv'), index=False)
test_df.to_csv(os.path.join(PROJECT_DIR, 'data/test_processed.csv'), index=False)

print("Done")
