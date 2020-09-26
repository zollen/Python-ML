'''
Created on Sep. 7, 2020

@author: zollen
'''


import numpy as np
import pandas as pd
import re
from sklearn import preprocessing
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.impute import KNNImputer
from xgboost import XGBRegressor

bayes = {}


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

titles = {
    "Army": 0,
    "Doctor": 1,
    "Nurse": 2,
    "Clergy": 3,
    "Baronness": 4,
    "Baron": 5,
    "Mr": 6,
    "Mrs": 7,
    "Miss": 8,
    "Master": 9,
    "Girl": 10,
    "Boy": 11,
    "GramPa": 12,
    "GramMa": 13
    }

cabins = {
    'A': 7,
    'B': 6,
    'C': 5,
    'D': 4,
    'E': 3,
    'F': 2,
    'G': 1,
    'X': 0
    }

embarkeds = {
    'S': 0,
    'Q': 1,
    'C': 2
    }
 
sexes = {
    'male': 0,
    'female': 1
    }  

def captures(df):
    df['Room']  = df['Cabin'].apply(captureRoom)
    df['Cabin'] = df['Cabin'].apply(captureCabin) 
    df['Title'] = df['Title'].map(titles)
    df['Sex'] = df['Sex'].map(sexes)
    df['Embarked'] = df['Embarked'].map(embarkeds)
    df['Cabin'] = df['Cabin'].map(cabins)
    
def typecast(df):
    df['Room'] = df['Room'].astype('int64')
    df['Age'] = df['Age'].astype('int64')
    df['Sex'] = df['Sex'].astype('int64')
    df['Embarked'] = df['Embarked'].astype('int64')
    df['Cabin'] = df['Cabin'].astype('int64')
    df['Title'] = df['Title'].astype('int64')
    df['Fare'] = df['Fare'].astype('int64')

 
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

def navieBayes(df, columns_lists):
    
    ## P(A and B)  = P(A|B) * P(B)
    ## P(A|B) = P(A and B) / P(B)
    ## P(A|B) = P(B|A) * P(A) / P(B)
    ## P(A and B) / P(B) = P(B|A) * P(A) / P(B)
    ## P(A and B) = P(B|A) * P(A)
    ## P(B|A) = P(A and B) / P(A)
    ADJUSTMENT = 0.0001
    
    total = len(df)
    
    bayes['A'] = len(df[df['Survived'] == 0]) / total
    bayes['D'] = len(df[df['Survived'] == 1]) / total
    
    for name in columns_lists:
        for val in columns_lists[name]:
            bayes[name + "=" + str(val)] = (len(df[df[name] == val]) / total) + ADJUSTMENT
            bayes["A|" + name + "=" + str(val)] = (len(
                df[(df[name] == val) & (df['Survived'] == 1)]) / total / bayes[name + "=" + str(val)]) + ADJUSTMENT
            bayes["D|" + name + "=" + str(val)] = (len(
                df[(df[name] == val) & (df['Survived'] == 0)]) / total / bayes[name + "=" + str(val)]) + ADJUSTMENT

def captureSurname(name):
    
    names = name.split(sep='(')
    
    surnames = ''    
    for name in names:
         
        if len(surnames) > 0:
            surnames += ','
            
        lst = name.split()
        lastName = lst[-1]
         
        if lastName == 'Jr' or lastName == 'II' or lastName == 'Mrs.':
            lastName = lst[-2]
    
        if len(lastName) <= 1:
            lastName = lst[-2] + "." + lastName
            
        lastName = lastName.replace(')', '')
        lastName = lastName.replace('"', '')
        lastName = lastName.replace(',', '')
            
        surnames += lastName
        
    return surnames
    
def calculateFamilyMembers(df):
    
    def word_count(msg):
        counts = dict()
        words = msg.split()

        for word in words:
            if word in counts:
                counts[word] += 1
            else:
                counts[word] = 1

        return counts
    
    df['Surname'] = 'None'
    df.loc[(df['Parch'] > 0) | (df['SibSp'] > 0), 'Surname'] = df.loc[(df['Parch'] > 0) | (df['SibSp'] > 0), 'Name'].apply(captureSurname)
    
    allLiveNames = ''
    allDeadNames = ''
    for rec, life in zip(df['Surname'], df['Survived']):
        if rec == 'None':
            continue
    
        lst = rec.split(',')
    
        for name in lst:
            if life == 1:
                allLiveNames += name + ' '
            else:
                allDeadNames += name + ' '

    namesLiveCount = word_count(allLiveNames)
    namesDeadCount = word_count(allDeadNames)
    
    df['Surname'].drop(columns = ['Surname'], inplace = True)
    
    return namesLiveCount, namesDeadCount

def captureFamilyMembersRatio(alives, deads): 
    
    def func(lastName):
        names = lastName.split(',')

        friends = 0
        for name in names:
            if name == 'None':
                continue
        
            if name in alives:
                friends += alives[name]
            
            if name in deads:
                friends -= deads[name]

        return friends
    
    return func
                    
def captureRoom(val):

    if str(val) != 'nan':
        x = re.findall("[0-9]+", val)
        if len(x) == 0:
            x = [ 0 ]

        return x[0]
        
    return 0
    
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
    return val

def oneHotEncoder(df1, df2, columns):
    
    ddf1 = df1.copy()
    ddf2 = df2.copy()
    
    cat_columns = []

    for name in columns:
    
        keys = np.union1d(ddf1[name].unique(), ddf2[name].unique())

        for key in keys:
            func = lambda x : 1 if x == key else 0
            ddf1[name + '.' + str(key)] = ddf1[name].apply(func)
            ddf2[name + '.' + str(key)] = ddf2[name].apply(func)
            cat_columns.append(name + '.' + str(key))
        
        ddf1.drop(columns = [name], inplace = True)
        ddf2.drop(columns = [name], inplace = True)
    
    return ddf1, ddf2, cat_columns

def reenigneeringXgBoost(src_df, dest_df, columns):
     
    model = XGBRegressor(objective="reg:linear")
    model.fit(src_df[columns], src_df['Survived'].squeeze())
    dest_df['XGBoost'] = model.predict(dest_df[columns])
    dest_df['XGBoost'] = dest_df['XGBoost'].round(4)
    
    return dest_df['XGBoost']
    
    
def reenigneeringFamilyMembers(df, alives, deads):
    
    df['Surname'] = 'None'  
    df.loc[(df['Parch'] > 0) | (df['SibSp'] > 0), 'Surname'] = df.loc[(df['Parch'] > 0) | (df['SibSp'] > 0), 'Name'].apply(captureSurname)  
    func = captureFamilyMembersRatio(alives, deads) 
    df['Family'] = df['Surname'].apply(func) 
    df.drop(columns = ['Surname'], inplace = True)
    
    
def reeigneeringTitle(dest_df):
    dest_df['Title'] = dest_df['Name'].apply(lambda x : re.search('[a-zA-Z]+\\.', x).group(0))
    dest_df['Title'] = dest_df.apply(map_title, axis = 1)

def reeigneeringAge(src_df, dest_df, columns):
    
    ddff = normalize({}, src_df, ['Title', 'Sex', 'Embarked', 'Pclass' ])
    tdff = normalize({}, dest_df, ['Title', 'Sex', 'Embarked', 'Pclass' ])
    
    imputer = KNNImputer(n_neighbors=13)
        
    imputer.fit(ddff[columns])
    
    ages = imputer.transform(tdff[columns])
         
    dest_df['Age'] = ages[:, 0]
    dest_df.loc[dest_df['Age'] < 1, 'Age'] = 1
 
    ## 3. Binning the Age
    dest_df['Age'] = pd.qcut(dest_df['Age'], q = 9,
                            labels = [ 0, 15, 20, 24, 26, 28, 32, 36, 46 ])  
    dest_df['Age'] = dest_df['Age'].astype('int32')

def reeigneeringFare(dest_df):
    dest_df['Fare'] = pd.qcut(dest_df['Fare'], 6, labels=[0, 10, 20, 30, 40, 80 ])

def reeigneeringFamilySize(dest_df):
    dest_df['Size'] = dest_df['SibSp'] + dest_df['Parch'] + 1
    dest_df['Size'] = dest_df['Size'].apply(captureSize)

def reeigneeringSurvProb(dest_df, coeffs, columns):
    func = survivability(False, coeffs, columns)
    dest_df['Chance'] = dest_df.apply(func, axis = 1)
                    
def reeigneeringCabin(src_df, dest_df):
    src_df['Cabin'] = src_df['Cabin'].apply(captureCabin) 
    dest_df['Cabin'] = dest_df['Cabin'].apply(captureCabin) 

    counts = src_df.groupby(['Title', 'Pclass', 'Sex', 'Cabin'])['Cabin'].count()
    
    for index, _ in counts.items():
        for df in [ src_df, dest_df ]:
            df.loc[(df['Cabin'].isna() == True) & (df['Title'] == index[0]) & 
                    (df['Pclass'] == index[1]) & (df['Sex'] == index[2]), 'Cabin'] = counts[index[0], index[1], index[2]].idxmax()      
    
    counts = src_df.groupby(['Pclass', 'Sex', 'Cabin'])['Cabin'].count()
    
    for index, _ in counts.items():
        for df in [ src_df, dest_df ]:
            df.loc[(df['Cabin'].isna() == True) & (df['Pclass'] == index[0]) & 
                    (df['Sex'] == index[1]), 'Cabin'] = counts[index[0], index[1]].idxmax()      
    
    dest_df.loc[dest_df['Cabin'] == 'T', 'Cabin'] = 'A'
    dest_df.loc[(dest_df['Cabin'] == 'B') | (dest_df['Cabin'] == 'C'), 'Cabin'] = 'A'
    dest_df.loc[(dest_df['Cabin'] == 'D') | (dest_df['Cabin'] == 'E'), 'Cabin'] = 'D'
    dest_df.loc[(dest_df['Cabin'] == 'F') | (dest_df['Cabin'] == 'G'), 'Cabin'] = 'G'
        
def normalize(encoders, df, columns):
    pdf = df.copy()
           
    for name in columns:
        encoders[name] = preprocessing.LabelEncoder()   
        
        keys = pdf.loc[pdf[name].isna() == False, name].unique()

        if len(keys) == 2:
            encoders[name] = preprocessing.LabelBinarizer()

        encoders[name].fit(keys)
        pdf.loc[pdf[name].isna() == False, name] = encoders[name].transform(
            pdf.loc[pdf[name].isna() == False, name].values)
            
    return pdf

def fill_by_regression(df_src, df_dest, name, columns):
 
    input_columns = columns
    predicted_columns = [ name ]

    withVal = df_src[df_src[name].isna() == False]
    withoutVal = df_src[df_src[name].isna() == True]
    
    cat_columns = input_columns
    
    df1 = normalize({}, withVal, cat_columns)
    df2 = normalize({}, withoutVal, cat_columns)
    
    model = ExtraTreesRegressor(random_state = 0)
    model.fit(df1[input_columns], withVal[predicted_columns])

    preds = model.predict(df2[input_columns])
    preds = [ round(i, 0) for i in preds ]

    df_dest.loc[df_dest[name].isna() == True, name] = preds

def fill_by_classification(df_src, df_dest, name, columns):

    input_columns = columns
    predicted_columns = [ name ]

    withVal = df_src[df_src[name].isna() == False]
    withoutVal = df_dest[df_dest[name].isna() == True]
    
    cat_columns = input_columns

    df1 = normalize({}, withVal, cat_columns)
    df2 = normalize({}, withoutVal, cat_columns)
     
    model = ExtraTreesClassifier(random_state = 0)
    model.fit(df1[input_columns], withVal[predicted_columns])
    preds = model.predict(df2[input_columns])
    
    df_dest.loc[df_dest[name].isna() == True, name] = preds

def survivability(classification, coeffs, columns):
    
    def func(rec):
        live = np.log(bayes['A'])
        die = np.log(bayes['D'])
        for name in columns:
            live += np.log(bayes['A|' + name + '=' + str(rec[name])]) * coeffs[name]
            die += np.log(bayes['D|' + name + '=' + str(rec[name])]) * coeffs[name]
    
        if classification:
            return 1 if live > die else 0
        
        ratio = np.abs((live - die) / np.max([ np.abs(live), np.abs(die) ])) * 0.5
           
        if live < die:
            ratio *= -1
        
        return  np.round(0.5 + ratio, 4)    

    return func

def combine(df1, df2):
    df = pd.concat([ df1, df2 ])
    df.set_index('PassengerId', inplace=True)
    return df