'''
Created on Aug. 1, 2020

@author: zollen
'''

import os
from pathlib import Path
import numpy as np
import pandas as pd
import re
from sklearn.impute import KNNImputer
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
numeric_columns = [ 'Age', 'Fare' ]
categorical_columns = [ 'Sex', 'Title', 'Pclass', 'Embarked', 'Cabin' ]
all_features_columns = numeric_columns + categorical_columns 

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

def navieBayes():
    bayes['A'] = 0.6162
    bayes['D'] = 0.3838
    
    bayes['A|Sex=male'] = 0.1890
    bayes['A|Sex=female'] = 0.7421
    bayes['D|Sex=male'] = 0.8112
    bayes['D|Sex=female'] = 0.2581

    bayes['A|Pclass=1'] = 0.6297
    bayes['A|Pclass=2'] = 0.4729
    bayes['A|Pclass=3'] = 0.2425
    bayes['D|Pclass=1'] = 0.6297
    bayes['D|Pclass=2'] = 0.5273
    bayes['D|Pclass=3'] = 0.7577

    bayes['A|Embarked=S'] = 0.3391
    bayes['A|Embarked=C'] = 0.5537
    bayes['A|Embarked=Q'] = 0.3897
    bayes['D|Embarked=S'] = 0.6611
    bayes['D|Embarked=C'] = 0.4465
    bayes['D|Embarked=Q'] = 0.6105

    bayes['A|Cabin=A'] = 0.6116
    bayes['A|Cabin=D'] = 0.4959
    bayes['A|Cabin=G'] = 0.2582
    bayes['D|Cabin=A'] = 0.3886
    bayes['D|Cabin=D'] = 0.5043
    bayes['D|Cabin=G'] = 0.7420

    bayes['A|Size=1'] = 0.3036
    bayes['A|Size=2'] = 0.5529
    bayes['A|Size=3'] = 0.6108
    bayes['A|Size=5'] = 0.1614
    bayes['D|Size=1'] = 0.6966
    bayes['D|Size=2'] = 0.4473
    bayes['D|Size=3'] = 0.3805
    bayes['D|Size=5'] = 0.8388

    bayes['A|Title=Mr'] = 0.1653
    bayes['A|Title=Mrs'] = 0.7758
    bayes['A|Title=Miss'] = 0.7560
    bayes['A|Title=GramPa'] = 0.0931
    bayes['A|Title=Master'] = 0.5751
    bayes['A|Title=Girl'] = 0.5637
    bayes['A|Title=GramMa'] = 0.9092
    bayes['A|Title=Baron'] = 0.5001
    bayes['A|Title=Clergy'] = 0.0001
    bayes['A|Title=Boy'] = 0.0001
    bayes['A|Title=Doctor'] = 0.3334
    bayes['A|Title=Army'] = 0.4001
    bayes['A|Title=Baronness'] = 0.6668
    bayes['A|Title=Nurse'] = 1.0000

    bayes['D|Title=Mr'] = 0.8349
    bayes['D|Title=Mrs'] = 0.2244
    bayes['D|Title=Miss'] = 0.2442
    bayes['D|Title=GramPa'] = 0.9071
    bayes['D|Title=Master'] = 0.4251
    bayes['D|Title=Girl'] = 0.4365
    bayes['D|Title=GramMa'] = 0.0910
    bayes['D|Title=Baron'] = 0.5001
    bayes['D|Title=Clergy'] = 1.0000
    bayes['D|Title=Boy'] = 1.0000
    bayes['D|Title=Doctor'] = 0.6668  
    bayes['D|Title=Army'] = 0.6001
    bayes['D|Title=Baronness'] = 0.3334
    bayes['D|Title=Nurse'] = 0.0001

    bayes['A|Age=0'] = 0.5589
    bayes['A|Age=15'] = 0.2937
    bayes['A|Age=20'] = 0.5159
    bayes['A|Age=24'] = 0.2377
    bayes['A|Age=26'] = 0.2341
    bayes['A|Age=28'] = 0.4302
    bayes['A|Age=32'] = 0.4601
    bayes['A|Age=36'] = 0.3435
    bayes['A|Age=46'] = 0.3879
    bayes['D|Age=0'] = 0.4413
    bayes['D|Age=15'] = 0.7065
    bayes['D|Age=20'] = 0.4843
    bayes['D|Age=24'] = 0.7625
    bayes['D|Age=26'] = 0.7661
    bayes['D|Age=28'] = 0.5700
    bayes['D|Age=32'] = 0.5401
    bayes['D|Age=36'] = 0.6567
    bayes['D|Age=46'] = 0.6123

    bayes['A|Fare=0'] = 0.2052
    bayes['A|Fare=5'] = 0.1909
    bayes['A|Fare=10'] = 0.3670
    bayes['A|Fare=20'] = 0.4363
    bayes['A|Fare=40'] = 0.4179
    bayes['A|Fare=80'] = 0.6981
    bayes['D|Fare=0'] = 0.7950
    bayes['D|Fare=5'] = 0.8093
    bayes['D|Fare=10'] = 0.6332
    bayes['D|Fare=20'] = 0.7950
    bayes['D|Fare=40'] = 0.5823
    bayes['D|Fare=80'] = 0.3021
    
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
    
    cat_columns = set(input_columns).intersection(categorical_columns)
    
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
    
    cat_columns = set(input_columns).intersection(categorical_columns)

    encoders = {}
    df1 = normalize(encoders, withVal, cat_columns)
    df2 = normalize(encoders, withoutVal, cat_columns)
     
    model = LogisticRegression()
    model.fit(df1[input_columns], withVal[predicted_columns])
    preds = model.predict(df2[input_columns])
    
    df_dest.loc[df_dest[name].isna() == True, name] = preds

def survivability(rec):
    live = np.log(bayes['A'])
    die = np.log(bayes['D'])
    for name in  [ 'Title', 'Sex', 'Pclass', 'Embarked', 'Size' ]:
        live += np.log(bayes['A|' + name + '=' + str(rec[name])])
        die += np.log(bayes['D|' + name + '=' + str(rec[name])])
        
    ratio = np.abs((live - die) / (live + die)) * 0.8
        
    if live < die:
        ratio *= -1
        
    return  np.round(0.5 + ratio, 4)
    
    
#    return 1 if live > die else 0
    
def enginneering(src_df, dest_df, columns):
    
    src_df['Title'] = src_df['Name'].apply(lambda x : re.search('[a-zA-Z]+\\.', x).group(0))
    src_df['Title'] = src_df.apply(map_title, axis = 1)
    dest_df['Title'] = dest_df['Name'].apply(lambda x : re.search('[a-zA-Z]+\\.', x).group(0))
    dest_df['Title'] = dest_df.apply(map_title, axis = 1)

    src_df.loc[src_df['Embarked'].isna() == True, 'Embarked'] = 'S'
    src_df.loc[src_df['Fare'].isna() == True, 'Fare'] = 7.25
    dest_df.loc[dest_df['Embarked'].isna() == True, 'Embarked'] = 'S'
    dest_df.loc[dest_df['Fare'].isna() == True, 'Fare'] = 7.25
    
    src_df['Cabin'] = src_df['Cabin'].apply(captureCabin) 
    dest_df['Cabin'] = dest_df['Cabin'].apply(captureCabin) 
    
    ## 1. Binning the Fare
    dest_df['Fare'] = pd.qcut(dest_df['Fare'], 6, labels=[0, 10, 20, 30, 40, 80 ])
    

    ## 2. Impute the missing Age
    encoders = {}
    ddff = normalize(encoders, dest_df, ['Title', 'Sex', 'Embarked', 'Pclass' ])
    
    imputer = KNNImputer(n_neighbors=13)
        
    ages = imputer.fit_transform(ddff[columns])
         
    dest_df['Age'] = ages[:, 0]
 
    src_df.loc[src_df['Age'] < 1, 'Age'] = 1           
    dest_df.loc[dest_df['Age'] < 1, 'Age'] = 1

    
    ## 3. Binning the Age
    dest_df['Age'] = pd.qcut(dest_df['Age'], q = 9,
                            labels = [ 0, 15, 20, 24, 26, 28, 32, 36, 46 ])  
    dest_df['Age'] = dest_df['Age'].astype('int32')
    
      
    
    ## 4. Try approximating Cabin
    if False:
        encoders = {}
        
        ddff = normalize(encoders, dest_df, ['Title', 'Sex', 'Embarked', 'Pclass', 'Cabin' ])

        cabins = KNNImputer(n_neighbors=13).fit_transform(ddff[['Cabin'] + columns])
      
        dest_df['Cabin'] = encoders['Cabin'].inverse_transform(
                        [ np.round(x, 0).astype('int32') for x in cabins[:, 0] ])
    else:  

        counts = src_df.groupby(['Title', 'Pclass', 'Sex', 'Cabin'])['Cabin'].count()
    
        for index, value in counts.items():
            for df in [ src_df, dest_df ]:
                df.loc[(df['Cabin'].isna() == True) & (df['Title'] == index[0]) & 
                       (df['Pclass'] == index[1]) & (df['Sex'] == index[2]), 'Cabin'] = counts[index[0], index[1], index[2]].idxmax()      
    
        fill_by_classification(dest_df, dest_df, 'Cabin', columns)
    
    dest_df.loc[dest_df['Cabin'] == 'T', 'Cabin'] = 'A'
    dest_df.loc[(dest_df['Cabin'] == 'B') | (dest_df['Cabin'] == 'C'), 'Cabin'] = 'A'
    dest_df.loc[(dest_df['Cabin'] == 'D') | (dest_df['Cabin'] == 'E'), 'Cabin'] = 'D'
    dest_df.loc[(dest_df['Cabin'] == 'F') | (dest_df['Cabin'] == 'G'), 'Cabin'] = 'G'
      
    ## 5. Family Size (SibSp + Parch + 1)column 'Alone', 'Small', 'Medium', 'Big'
    
    dest_df['Size'] = dest_df['SibSp'] + dest_df['Parch'] + 1
    dest_df['Size'] = dest_df['Size'].apply(captureSize)
    
    
    
    ## 6. Try add a survivibility percentage column
    dest_df['Chance'] = dest_df.apply(survivability, axis = 1)
    
    
#   Testing the accuracy of features

#    if 'Survived' in columns:
#        alls = len(dest_df)
#        correct = dest_df[((dest_df['Survived'] == 0) & (dest_df['Chance'] == 0)) | 
#            ((dest_df['Survived'] == 1) & (dest_df['Chance'] == 1))]['PassengerId'].count()
               
#        print("Accuracy %0.2f" % (correct / alls))
    
    
     
    return dest_df


PROJECT_DIR=str(Path(__file__).parent.parent)  
train_df = pd.read_csv(os.path.join(PROJECT_DIR, 'data/train.csv'))
test_df = pd.read_csv(os.path.join(PROJECT_DIR, 'data/test.csv'))

all_df = pd.concat([ train_df, test_df ])
all_df.set_index('PassengerId', inplace=True)


navieBayes()


train_df = enginneering(all_df, train_df, 
                        ['Age', 'Title', 'Sex', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Pclass', 'Survived' ])

all_df = pd.concat([ train_df, test_df ])
all_df.set_index('PassengerId', inplace=True)

test_df = enginneering(all_df, test_df,
                        ['Age', 'Title', 'Sex', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Pclass' ])



train_df.drop(columns = [ 'Name', 'Ticket', 'Sex', 'SibSp', 'Parch'], inplace = True)
test_df.drop(columns = [ 'Name', 'Ticket', 'Sex', 'SibSp', 'Parch'], inplace = True)


if False:
    g = sb.FacetGrid(train_df, col = "Survived", row = "Pclass", size = 2)
    g.map(sb.distplot, "Age", bins = 25)
    plt.show()
    exit()
    
if False:
    dd = train_df[train_df['Cabin'].isna() == False]  
    sb.catplot(x = "Pclass", y = "Title", hue = "Cabin", kind = "swarm", data = dd)
    plt.show()
    exit()
    
if False:    
#    dd = train_df[train_df['Cabin'].isna() == False]    
    sb.factorplot(x = 'Pclass' ,y = 'Fare', hue = 'Survived', kind = 'violin', data = train_df)
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
