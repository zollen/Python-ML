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
from sklearn.ensemble import ExtraTreesRegressor

warnings.filterwarnings('ignore')

label_column = [ 'Survived']
identity_columns = [ 'PassengerId', 'Ticket' ]
numeric_columns = [ 'Age', 'Fare', 'Cabin', 'Size' ]
categorical_columns = [ 'Sex', 'Title', 'Pclass', 'Embarked' ]
all_features_columns = numeric_columns + categorical_columns 



def fillValues(name, *args):
    key, columns, label = ['PassengerId'], ['Title', 'Sex', 'Fare', 'SibSp', 'Parch', 'Pclass', 'Embarked' ], [name]
    
    alldf = []
    for df in args:
        alldf.append(df)
        
    all_df = pd.concat(alldf)
    all_df = all_df[key + columns + label]
    
    all_df = pd.get_dummies(all_df, columns = ['Title', 'Sex', 'Pclass', 'Embarked'])
    
    cols = set(all_df.columns)
    cols.remove(name)
    

    all_df_in = all_df.loc[all_df[name].isna() == False, cols]
    all_df_lb = all_df.loc[all_df[name].isna() == False, label]

    model = ExtraTreesRegressor(random_state = 0)
    model.fit(all_df_in, all_df_lb)
    
    
    all_df_im = all_df.loc[all_df[name].isna() == True, cols]
       
    preds = model.predict(all_df_im)
    all_df_im[name] = preds
    
    
    for df in args:
        df.loc[df[name].isna() == True, name] = all_df_im.loc[all_df_im['PassengerId'].isin(df['PassengerId']), name]
        df[name] = df[name].astype('int64')
    
   
    
            
def preprocessingTitle(rec):
    name = rec['Name']
    sex = rec['Sex']

    name = re.search('[a-zA-Z]+\\.', name).group(0)
    name = name.replace(".", "")
    if name == 'Col' or name == 'Capt' or name  == 'Major':
        name = 0
    elif name == 'Rev' and sex == 'male':
        return 1
    elif name == 'Dr' and sex == 'female':
        name = 2
    elif name == 'Dr' and sex == 'male':
        return 3
    elif name == 'Sir' or name == 'Don':
        name = 4
    elif name == 'Mme' or name == 'Mrs':
        name = 5
    elif name == 'Lady' or name == 'Countess' or name == 'Dona':
        return 6
    elif name == 'Miss' or name == 'Ms' or name == 'Mlle' or name == 'Jonkheer':
        return 7
    elif name == 'Master':
        return 8
    elif name == 'Mr':
        return 9
        
    
    return name
      
def postrocessingTitle(rec):
    title = rec['Title']
    sex = rec['Sex']
    age = rec['Age']
    
    if sex == 'male' and age < 16:
        return 10
    elif sex == 'female' and age < 16:
        return 11
    elif age >= 55 and sex == 'male':
        return 12
    elif age >= 55 and sex == 'female':
        return 13
    
    
    return title

fareBinned = {}
def calFare(df):
         
    last = 0
    for fare in [ 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 200, 500, 1000 ]:
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
        
def binFare(*args):
    
    for df in args:
        last = 0
        for fare in [ 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 200, 500, 1000]:
            male = str(last) + ":" + str(fare) + ":male"
            female = str(last) + ":" + str(fare) + ":female"
            df.loc[(df['Fare'] >= last) & (df['Fare'] < fare) & (df['Sex'] == 0), 'FareP'] = fareBinned[male]
            df.loc[(df['Fare'] >= last) & (df['Fare'] < fare) & (df['Sex'] == 1), 'FareP'] = fareBinned[female]
            last = fare
    
        df['Fare'] = df['FareP'].round(4)
        df.drop(columns = ['FareP'], inplace = True)   
    
ageBinned = {}
def calAge(df):
         
    last = 0
    for age in [ 5, 10, 15, 20, 25, 30, 35, 40, 50, 55, 60, 65, 70, 75, 100 ]:
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
        
def binAge(*args):
    
    for df in args:
        last = 0
        for age in [ 5, 10, 15, 20, 25, 30, 35, 40, 50, 55, 60, 65, 70, 75, 100 ]:
            male = str(last) + ":" + str(age) + ":male"
            female = str(last) + ":" + str(age) + ":female"
            df.loc[(df['Age'] >= last) & (df['Age'] < age) & (df['Sex'] == 0), 'AgeP'] = ageBinned[male]
            df.loc[(df['Age'] >= last) & (df['Age'] < age) & (df['Sex'] == 1), 'AgeP'] = ageBinned[female]
            last = age
    
        df['Age'] = df['AgeP'].round(4)
        df.drop(columns = ['AgeP'], inplace = True)   

titleBinned = {}
def calTitle(df):
    
    for title in [ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13 ]:
        alives = len(df[(df['Title'] == title) & (df['Survived'] == 1)])
        deads = len(df[(df['Title'] == title) & (df['Survived'] == 0)])
        ratio = 0 if alives + deads == 0 else alives / (alives + deads)
#        print("[%2d]: Alives {%3d} Dead: {%3d} ==> [%0.4f]" % 
#              (title, alives, deads, ratio))
        titleBinned[str(title)] = ratio
    
def binTitle(*args):
    
    for df in args:
        for title in [ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13 ]:
            df.loc[df['Title'] == title, 'TitleP'] = titleBinned[str(title)]
            
        df['Title'] = df['TitleP'].round(4)
        df.drop(columns = ['TitleP'], inplace = True)   
    
            
def prepare(*args):
    
    global categorical_columns
    
    alldf = []
    for df in args:
        alldf.append(df)
        
    all_df = pd.concat(alldf)    
    all_df = pd.get_dummies(all_df, columns = categorical_columns)

    train_uni = set(all_df.columns).symmetric_difference(numeric_columns + ['Name', 'PassengerId'] + label_column)

    cat_columns = list(train_uni)
    
    all__columns = numeric_columns + cat_columns 
      
    scaler = MinMaxScaler()
    all_df[numeric_columns] = scaler.fit_transform(all_df[numeric_columns])
    
    return all_df, all__columns
    
def computeRegression(name, df1, df2):
    
    all_df, all_columns = prepare(df1, df2)
     
    if 'Survived' in all_columns:
        all_columns.remove('Survived')

    all_df_in = all_df.loc[all_df['Survived'].isna() == False, ['PassengerId'] + all_columns]
    all_df_lb = all_df.loc[all_df['Survived'].isna() == False, 'Survived']
    
    model = LogisticRegression(max_iter=500, solver='lbfgs')
    model.fit(all_df_in[all_columns], all_df_lb)
        
    for df in [ df1, df2 ]:
        df[all_columns] = all_df.loc[all_df['PassengerId'].isin(df['PassengerId']), all_columns]
        df[name] = model.predict_proba(df[all_columns])[:, 0]
        df[name] = df[name].round(4)

def calCabin(rec):
    prefix = rec['CabinPrefix']
    room = rec['CabinRoom']
    
    if str(prefix) != 'nan' and str(room) != 'nan':
        return int(prefix) + int(room)
    
    return np.nan    


            
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
train_df['Fare'] = train_df['Fare'].round(2)
test_df['Fare'] = test_df['Fare'].round(2)


## extract Title from Name
train_df['Title'] = train_df.apply(preprocessingTitle, axis = 1)
test_df['Title'] = test_df.apply(preprocessingTitle, axis = 1)


## encoding sex
train_df['Sex'] = train_df['Sex'].map(tb.sexes)
test_df['Sex'] = test_df['Sex'].map(tb.sexes)

## encoding Embarked
train_df['Embarked'] = train_df['Embarked'].map(tb.embarkeds)
test_df['Embarked'] = test_df['Embarked'].map(tb.embarkeds)


## filling ages
fillValues('Age', train_df, test_df)


## adjust title
train_df['Title'] = train_df.apply(postrocessingTitle, axis = 1)
test_df['Title'] = test_df.apply(postrocessingTitle, axis = 1)


## encoding cabin
train_df['CabinPrefix'] = train_df['Cabin'].apply(tb.captureCabin) 
test_df['CabinPrefix'] = test_df['Cabin'].apply(tb.captureCabin) 

train_df['CabinRoom'] = train_df['Cabin'].apply(tb.captureRoom)
test_df['CabinRoom'] = test_df['Cabin'].apply(tb.captureRoom) 

train_df['CabinPrefix'] = train_df['CabinPrefix'].map({ 'A': 0, 'B': 800, 'C': 400, 
                                           'D': 1200, 'E': 1000, 'F': 600, 'G': 200 })
test_df['CabinPrefix'] = test_df['CabinPrefix'].map({ 'A': 0, 'B': 800, 'C': 400, 
                                           'D': 1200, 'E': 1000, 'F': 600, 'G': 200 })
train_df['Cabin'] = train_df.apply(calCabin, axis = 1)
test_df['Cabin'] = test_df.apply(calCabin, axis = 1)


## Fill Cabin
fillValues('Cabin', train_df, test_df)


train_df.drop(columns = ['CabinPrefix', 'CabinRoom'], inplace = True)
test_df.drop(columns = ['CabinPrefix', 'CabinRoom'], inplace = True)


## engineering ticket number
train_df['Ticket'] = train_df['Ticket'].apply(tb.captureTicketId)
test_df['Ticket'] = test_df['Ticket'].apply(tb.captureTicketId)

train_df['Ticket'] = np.log(train_df['Ticket'])
test_df['Ticket'] = np.log(test_df['Ticket'])

train_df['Ticket'] = train_df['Ticket'].round(4)
test_df['Ticket'] = test_df['Ticket'].round(4)


## engineering family size
train_df['Size'] = train_df['Parch'] + train_df['SibSp'] + 1
test_df['Size'] = test_df['Parch'] + test_df['SibSp'] + 1



calAge(train_df)
binAge(train_df, test_df)

calFare(train_df)
binFare(train_df, test_df)

calTitle(train_df)
binTitle(train_df, test_df)


computeRegression("Logistic", train_df, test_df)






train_df.drop(columns = ['Name', 'Parch', 'SibSp'], inplace = True)
test_df.drop(columns = ['Name', 'Parch', 'SibSp'], inplace = True)


train_df.to_csv(os.path.join(PROJECT_DIR, 'data/train_processed.csv'), index=False)
test_df.to_csv(os.path.join(PROJECT_DIR, 'data/test_processed.csv'), index=False)

print("Done")
