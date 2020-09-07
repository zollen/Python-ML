'''
Created on Sep. 7, 2020

@author: zollen
'''


import numpy as np
import pandas as pd
import re
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.impute import KNNImputer

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

def navieBayes(train_df, columns_lists):
    
    ## P(A and B)  = P(A|B) * P(B)
    ## P(A|B) = P(A and B) / P(B)
    ## P(A|B) = P(B|A) * P(A) / P(B)
    ## P(A and B) / P(B) = P(B|A) * P(A) / P(B)
    ## P(A and B) = P(B|A) * P(A)
    ## P(B|A) = P(A and B) / P(A)
    ADJUSTMENT = 0.0001
    
    total = len(train_df)
    bayes['A'] = len(train_df[train_df['Survived'] == 0]) / total
    bayes['D'] = len(train_df[train_df['Survived'] == 1]) / total
    
    for name in columns_lists:
        for val in columns_lists[name]:
            bayes[name + "=" + str(val)] = (len(train_df[train_df[name] == val]) / total) + ADJUSTMENT
            bayes["A|" + name + "=" + str(val)] = (len(
                train_df[(train_df[name] == val) & (train_df['Survived'] == 1)]) / total / bayes[name + "=" + str(val)]) + ADJUSTMENT
            bayes["D|" + name + "=" + str(val)] = (len(
                train_df[(train_df[name] == val) & (train_df['Survived'] == 0)]) / total / bayes[name + "=" + str(val)]) + ADJUSTMENT
            
    
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

def reeigneeringTitle(dest_df):
    dest_df['Title'] = dest_df['Name'].apply(lambda x : re.search('[a-zA-Z]+\\.', x).group(0))
    dest_df['Title'] = dest_df.apply(map_title, axis = 1)

def reeigneeringAge(src_df, dest_df, columns):
    encoders = {}
    
    ddff = normalize(encoders, src_df, ['Title', 'Sex', 'Embarked', 'Pclass' ])
    tdff = normalize(encoders, dest_df, ['Title', 'Sex', 'Embarked', 'Pclass' ])
    
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

def reeigneeringSurvProb(dest_df, columns):
    func = survivability(False, columns)
    dest_df['Chance'] = dest_df.apply(func, axis = 1)
                    
def reeigneeringCabin(src_df, dest_df):
    src_df['Cabin'] = src_df['Cabin'].apply(captureCabin) 
    dest_df['Cabin'] = dest_df['Cabin'].apply(captureCabin) 

    counts = src_df.groupby(['Title', 'Pclass', 'Sex', 'Cabin'])['Cabin'].count()
    
    for index, value in counts.items():
        for df in [ src_df, dest_df ]:
            df.loc[(df['Cabin'].isna() == True) & (df['Title'] == index[0]) & 
                    (df['Pclass'] == index[1]) & (df['Sex'] == index[2]), 'Cabin'] = counts[index[0], index[1], index[2]].idxmax()      
    
    counts = src_df.groupby(['Pclass', 'Sex', 'Cabin'])['Cabin'].count()
    
    for index, value in counts.items():
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
    
    encoders = {}
    df1 = normalize(encoders, withVal, cat_columns)
    df2 = normalize(encoders, withoutVal, cat_columns)
    
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
    
    cat_columns = input_columns

    encoders = {}
    df1 = normalize(encoders, withVal, cat_columns)
    df2 = normalize(encoders, withoutVal, cat_columns)
     
    model = LogisticRegression()
    model.fit(df1[input_columns], withVal[predicted_columns])
    preds = model.predict(df2[input_columns])
    
    df_dest.loc[df_dest[name].isna() == True, name] = preds

def survivability(classification, columns):
    
    def func(rec):
        live = np.log(bayes['A'])
        die = np.log(bayes['D'])
        for name in columns:
            live += np.log(bayes['A|' + name + '=' + str(rec[name])])
            die += np.log(bayes['D|' + name + '=' + str(rec[name])])
    
        if classification:
            return 1 if live > die else 0
        
        ratio = np.abs((live - die) / np.max([ np.abs(live), np.abs(die) ])) * 0.5
           
        if live < die:
            ratio *= -1
        
        return  np.round(0.5 + ratio, 4)    

    return func
