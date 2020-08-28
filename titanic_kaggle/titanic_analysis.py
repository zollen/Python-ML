'''
Created on Aug. 1, 2020

@author: zollen
'''

import numpy as np
import pandas as pd
import re
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
import statsmodels.api as sm
from sklearn import preprocessing
import seaborn as sb
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

pd.set_option('max_columns', None)
pd.set_option('max_rows', None)
np.random.seed(0)
sb.set_style('whitegrid')

label_column = [ 'Survived']
identity_columns = [ 'PassengerId', 'Ticket', 'Cabin' ]
numeric_columns = [ 'Age', 'SibSp', 'Parch', 'Fare' ]
categorical_columns = [ 'Title', 'Sex', 'Embarked', 'Pclass' ]
all_features_columns = numeric_columns + categorical_columns 


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


train_df = pd.read_csv('data/train.csv')
test_df = pd.read_csv('data/test.csv')

titles = lambda x : re.search('[a-zA-Z]+\\.', x).group(0)
train_df['Title'] = train_df['Name'].apply(titles)
test_df['Title'] = test_df['Name'].apply(titles)


print(train_df.info())
print("=============== STATS ===================")
print(train_df.describe())
print("============== Training Total NULL ==============")
print(train_df.isnull().sum())
print("============== SKEW =====================")
print(train_df.skew())

if False:
    df = normalize(train_df[train_df['Age'].isna() == False], numeric_columns + label_column)
    corr = train_df.corr() 
    surv = corr.copy()
    surv.loc[:,:] = False
    surv['Survived'] = True
  
    sb.heatmap(corr[((corr >= 0.3) | (corr <= -0.3)) & (corr != 1) | surv], annot=True, linewidths=0.5, fmt='0.2f')
    plt.show()
    exit()


print("ALIVE: ", len(train_df[train_df['Survived'] == 1]))
print("DEAD: ", len(train_df[train_df['Survived'] == 0]))

if True:
    subsetdf = train_df[(train_df['Age'].isna() == False) & (train_df['Embarked'].isna() == False)]
    df = normalize(subsetdf, all_features_columns + label_column)

    print("======= SelectKBest =======")
    model = SelectKBest(score_func=chi2, k=5)
    kBest = model.fit(df[all_features_columns], df[label_column])
    func = lambda x : np.round(x, 2)
    print(np.stack((all_features_columns, func(kBest.scores_)), axis=1))

    print("======= ExtermeDecisionTree =======")
    model = ExtraTreesClassifier(random_state = 0)
    model.fit(df[all_features_columns], df[label_column])
    print(np.stack((all_features_columns, func(model.feature_importances_)), axis=1))

    print("======= Logit Maximum Likelihood Analysis ===========")
    model=sm.Logit(df[label_column], df[all_features_columns])
    result=model.fit()
    print(result.summary2())


"""
Importants of each feature
==========================
SelectKBest: Fare > Sex > Age > Pclass > Parch > Embarked > SibSp
DecisionTree: Age, Sex > Fare > Pclass > Parch, SibSp > Embarked
Logit: Fare, Sex has the highest confident of not seeing invalid variants
"""


if False:
    sb.catplot(x = "Survived", y = "Age", hue = "Sex", kind = "swarm", data = subsetdf)
    sb.catplot(x = "Survived", y = "Age", hue = "Pclass", kind = "swarm", data = subsetdf)
    plt.show()
    exit()
    
"""
    The above plot shows that Age does play a role of the outcome of survivaility.
    Children who are below 10 years old have a decent higher chance to surivive (specially you 
    are a female).
    People between sixteen and thirty-five have much lower chance to surive (unless you are a 
    female)
    People who are above thirty five (both alive and dead group appears to converge to a similar
    points distribution with the dead group slightly bigger than the alive group)
    
    The number of dots between each column at each Age group appears to be vary enough
    to warrent a consideration
"""

if False:
    fig, (a1, a2, a3, a4) = plt.subplots(1, 4)
    fig.set_size_inches(14 , 10)
    sb.countplot(x = "Embarked", hue = "Survived", data = subsetdf, ax = a1);
    a1.set_title('Embarked - Survived')
    sb.countplot(x = "Pclass", hue = "Survived", data = subsetdf, ax = a2);
    a2.set_title('Pclass - Survived')
    sb.countplot(x = "SibSp", hue = "Survived", data = subsetdf, ax = a3);
    a3.set_title('SibSp - Survived')
    sb.countplot(x = "Parch", hue = "Survived", data = subsetdf, ax = a4);
    a4.set_title('Parch - Survived')
    plt.show()
    exit()
    
"""
Both Sibsp, Parch, Embarked and Pclass do show uneven distribution in term of survivability.
"""


""" 
Let's process the Cabin column to maximize the information available to us
"""
def reformat(val):
    if str(val) == 'nan':
        return np.nan
    else:
        x = re.findall("[a-zA-Z]+[0-9]{1}", val)
        if len(x) == 0:
            x = re.findall("[a-zA-Z]{1}", val)
        
        return x[0][0]
    
train_df['Cabin'] = train_df['Cabin'].apply(reformat)
test_df['Cabin'] = test_df['Cabin'].apply(reformat)




"""
Let's fill the missing values in both training and testing samples
"""

def fill_by_regression(df_src, df_dest, name, columns):
 
    input_columns = columns
    predicted_columns = [ name ]

    withVal = df_src[df_src[name].isna() == False]
    withoutVal = df_src[df_src[name].isna() == True]
    
    cat_columns = set(input_columns).intersection(categorical_columns)

    df1 = normalize(withVal, cat_columns)
    df2 = normalize(withoutVal, cat_columns)
    
    model = ExtraTreesRegressor(random_state = 0)
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
    
    model = ExtraTreesClassifier(random_state = 0)
    model.fit(df1[input_columns], withVal[predicted_columns])
    preds = model.predict(df1[input_columns])
    print("Accuracy: %0.2f" % accuracy_score(withVal[predicted_columns], preds))
    print(confusion_matrix(withVal[predicted_columns], preds))
    

    preds = model.predict(df2[input_columns])

    print("Predicted %s values: " % name)
    print(np.stack((withoutVal['PassengerId'], preds), axis=1))

    df_dest.loc[df_dest[name].isna() == True, name] = preds

    
"""
         PassengerId  Survived  Pclass                                      Name      Sex   Age  SibSp  Parch  Ticket  Fare Cabin Embarked  
61                62        1       1                        Icard, Miss. Amelie   female  38.0      0      0  113572  80.0     B      NaN  
829              830        1       1  Stone, Mrs. George Nelson (Martha Evelyn)   female  62.0      0      0  113572  80.0     B      NaN 
"""
    



"""
let analysis the two samples with missing Embarked value
"""
if False:
    df = train_df.copy()
    df.loc[train_df['Embarked'].isna() == True, 'Embarked'] = "A"
    plt.figure(figsize = (28, 10))
    sb.swarmplot(x = "Age", y = "Fare", hue = "Embarked", alpha = 0.7, data = df)
    plt.show()
    exit()

"""
At 32 yrs, there are 2 Pclass(S) dots and 2 Pclass(C) dots neighbours
At 80 yrs, there are 1 Pclass(S) dots and 0 Pclass(C) dots neighbour
-- inconclusive
"""



"""
let analysis the one *test* sample with missing Fare value
    PassengerId  Pclass                Name   Sex    Age  SibSp  Parch Ticket   Fare   Cabin  Embarked  
152        1044       3  Storey, Mr. Thomas  male  60.5      0      0   3701    NaN     NaN         S
"""

if False:
    plt.figure(figsize = (28, 10))
    sb.swarmplot(x = "Age", y = "Fare", hue = "Pclass", alpha = 0.7, data = train_df)
    plt.show()

print(train_df[train_df['Embarked'].isna() == True])

fill_by_classification(train_df, train_df, 'Embarked', [ 'Title', 'Survived', 'SibSp', 'Parch', 'Fare', 'Sex', 'Pclass' ])    
fill_by_regression(train_df, train_df, 'Age', [ 'Title', 'Survived', 'SibSp', 'Parch', 'Fare', 'Sex', 'Pclass', 'Embarked' ])
fill_by_classification(train_df, train_df, 'Cabin', [ 'Title', 'Survived', 'SibSp', 'Parch', 'Fare', 'Sex', 'Age', 'Pclass', 'Embarked' ])


allsamples = pd.concat([ train_df, test_df ])
fill_by_regression(allsamples[allsamples['Age'].isna() == False], test_df, 'Fare', [ 'Title', 'Age', 'SibSp', 'Parch', 'Embarked', 'Sex', 'Pclass' ])
fill_by_regression(pd.concat([ train_df, test_df ]), test_df, 'Age', [ 'Title', 'SibSp', 'Parch', 'Fare', 'Sex', 'Pclass', 'Embarked' ])
fill_by_classification(pd.concat([ train_df, test_df ]), test_df, 'Cabin', [ 'Title', 'Age', 'SibSp', 'Parch', 'Embarked', 'Sex', 'Fare', 'Pclass' ])


print(train_df.head())

outputs = ['PassengerId', 'Name', 'Ticket', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex', 'Embarked',  'Pclass', 'Cabin', 'Survived' ]
train_df[outputs].to_csv('data/train_processed.csv', index=False)
outputs = ['PassengerId', 'Name', 'Ticket', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex', 'Embarked',  'Pclass', 'Cabin' ]
test_df[outputs].to_csv('data/test_processed.csv', index=False)

print("Done")
