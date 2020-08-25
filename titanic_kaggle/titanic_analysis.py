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
sb.set_style('whitegrid')

label_column = [ 'Survived']
identity_columns = [ 'PassengerId', 'Ticket', 'Cabin' ]
numeric_columns = [ 'Age', 'SibSp', 'Parch', 'Fare' ]
categorical_columns = [ 'Sex', 'Embarked', 'Pclass' ]
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

print(train_df.info())
print("=============== STATS ===================")
print(train_df.describe())
print("============== Training Total NULL ==============")
print(train_df.isnull().sum())
print("============== SKEW =====================")
print(train_df.skew())

print("ALIVE: ", len(train_df[train_df['Survived'] == 1]))
print("DEAD: ", len(train_df[train_df['Survived'] == 0]))


subsetdf = train_df[(train_df['Age'].isna() == False) & (train_df['Embarked'].isna() == False)]

df = normalize(subsetdf, all_features_columns + label_column)


print("======= SelectKBest =======")
model = SelectKBest(score_func=chi2, k=5)
kBest = model.fit(df[all_features_columns], df[label_column])
func = lambda x : np.round(x, 2)
print(np.stack((all_features_columns, func(kBest.scores_)), axis=1))

print("======= ExtermeDecisionTree =======")
model = ExtraTreesClassifier()
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

def fillinFile(df, with_target, fileName):
    
    """
    Let's 'guess' the samples with missing 'Age' values
    """
    if with_target == True:
        input1_columns = [ 'Survived', 'SibSp', 'Parch', 'Fare', 'Sex', 'Embarked', 'Pclass' ]
    else:
        input1_columns = [ 'SibSp', 'Parch', 'Fare', 'Sex', 'Embarked', 'Pclass' ]
        
    predicted1_columns = [ 'Age' ]

    withAge = df[df['Age'].isna() == False]
    withoutAge = df[df['Age'].isna() == True]

    df1 = normalize(withAge, input1_columns)
    df2 = normalize(withoutAge, input1_columns)

    model1 = ExtraTreesRegressor()
    model1.fit(df1[input1_columns], df1[predicted1_columns])

    preds1 = model1.predict(df2[input1_columns])
    preds1 = [ round(i, 0) for i in preds1 ]
    print("Predicted Age values: ")
    print(np.stack((withoutAge['PassengerId'], preds1), axis=1))

    df.loc[df['Age'].isna() == True, 'Age'] = preds1


    """
    Let's 'guess' the samples with missing 'Cabin' values
    """

    if with_target == True:
        input2_columns = [ 'Survived', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex', 'Embarked', 'Pclass' ]
    else:
        input2_columns = [ 'Age', 'SibSp', 'Parch', 'Fare', 'Sex', 'Embarked', 'Pclass' ]
        
    predicted2_columns = [ 'Cabin' ]


    withCabin = df[df['Cabin'].isna() == False]
    withoutCabin = df[df['Cabin'].isna() == True]

    df3 = normalize(withCabin, input2_columns)
    df4 = normalize(withoutCabin, input2_columns)

    model2 = ExtraTreesClassifier()
    model2.fit(df3[input2_columns], df3[predicted2_columns])
    preds = model2.predict(df3[input2_columns])
    print("Accuracy: %0.2f" % accuracy_score(df3[predicted2_columns], preds))
    print(confusion_matrix(df3[predicted2_columns], preds))

    preds2 = model2.predict(df4[input2_columns])

    print("Predicted Cabin values: ")
    print(np.stack((withoutCabin['PassengerId'], preds2), axis=1))

    df.loc[df['Cabin'].isna() == True, 'Cabin'] = preds2
    
    df.to_csv(fileName)
    



"""
let analysis the two samples with missing Embarked value
"""
if False:
    df = train_df.copy()
    df.loc[train_df['Embarked'].isna() == True, 'Embarked'] = "A"
    plt.figure(figsize = (28, 10))
    sb.swarmplot(x = "Age", y = "Fare", hue = "Embarked", alpha = 0.7, data = df)
    plt.show()

"""
At 32 yrs, there are 2 Pclass(S) dots and 2 Pclass(C) dots neighbours
At 80 yrs, there are 1 Pclass(S) dots and 0 Pclass(C) dots neighbour
-- inconclusive
"""

"""
Let's 'guess' the two missing Embarked samples
      Age  SibSp  Parch  Fare     Sex Embarked  Pclass  Survived  Ticket Cabin
61   38.0      0      0  80.0  female      NaN       1         1  113572   B28
829  62.0      0      0  80.0  female      NaN       1         1  113572   B28
"""
withoutEmbarked = train_df[train_df['Embarked'].isna() == True]
print("The two samples with missing Embarked value")
print(withoutEmbarked[identity_columns + all_features_columns + label_column])

withEmbarked = train_df[(train_df['Embarked'].isna() == False) & (train_df['Age'].isna() == False)]

input_columns_for_embarked = [ 'Survived', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex', 'Pclass' ]
predicted_embarked_columns = [ 'Embarked' ]

df1 = normalize(withEmbarked, input_columns_for_embarked)
df2 = normalize(withoutEmbarked, input_columns_for_embarked)


model = ExtraTreesClassifier()
model.fit(df1[input_columns_for_embarked], df1[predicted_embarked_columns])
preds = model.predict(df1[input_columns_for_embarked])
print("Accuracy: %0.2f" % accuracy_score(df1[predicted_embarked_columns], preds))
print(confusion_matrix(df1[predicted_embarked_columns], preds))
        
preds = model.predict(df2[input_columns_for_embarked])
print("Predicted Embarked values: ", preds)

"""
The classifier has determined both missing Embarked samples should be 'S'
"""
train_df.loc[train_df['Embarked'].isna() == True, 'Embarked'] = 'S'



"""
let analysis the one *test* sample with missing Fare value
    PassengerId  Pclass                Name   Sex    Age  SibSp  Parch Ticket   Fare   Cabin  Embarked  
152        1044       3  Storey, Mr. Thomas  male  60.5      0      0   3701    NaN     NaN         S
"""

if False:
    plt.figure(figsize = (28, 10))
    sb.swarmplot(x = "Age", y = "Fare", hue = "Pclass", alpha = 0.7, data = train_df)
    plt.show()
    
fillinFile(train_df, True, "data/train_processed.csv")

input_columns_for_fare = [ 'Age', 'SibSp', 'Parch', 'Embarked', 'Sex', 'Pclass' ]
predicted_fare_columns = [ 'Fare' ]

df3 = normalize(train_df, input_columns_for_fare)
df4 = normalize(test_df[test_df['Fare'].isna() == True], input_columns_for_fare)

model = ExtraTreesRegressor()
model.fit(df3[input_columns_for_fare], df3[predicted_fare_columns])
        
preds = model.predict(df4[input_columns_for_fare])
test_df.loc[test_df['Fare'].isna() == True, 'Fare'] = round(preds[0], 2)

fillinFile(test_df, False, "data/test_processed.csv")

print("Missing Fare Value: %0.2f" % preds[0])
print("============== Test Total NULL ==============")
print(test_df[test_df['PassengerId'] == 1000])