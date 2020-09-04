'''
Created on Sep. 2, 2020

@author: zollen
'''

import os
from pathlib import Path
import numpy as np
import re
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
import pandas as pd
import seaborn as sb
import warnings


warnings.filterwarnings('ignore')

pd.set_option('max_columns', None)
pd.set_option('max_rows', None)
np.random.seed(0)
sb.set_style('whitegrid')

ADJUSTMENT = 0.0001

categorical_columns = [ 'Sex', 'Title', 'Pclass', 'Embarked', 'Cabin' ]

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
    
def captureCabin(val):
    
    if str(val) != 'nan':
        x = re.findall("[a-zA-Z]+[0-9]{1}", val)
        if len(x) == 0:
            x = re.findall("[a-zA-Z]{1}", val)
            
        return x[0][0]
        
    return val
        
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

PROJECT_DIR=str(Path(__file__).parent.parent)  
train_df = pd.read_csv(os.path.join(PROJECT_DIR, 'data/train.csv'))

train_df['Title'] = train_df['Name'].apply(lambda x : re.search('[a-zA-Z]+\\.', x).group(0))
train_df['Title'] = train_df.apply(map_title, axis = 1)
train_df['Age'] = train_df.groupby(['Title', 'Sex', 'Pclass'])['Age'].apply(lambda x : x.fillna(x.median()))
train_df.loc[train_df['Age'] < 1, 'Age'] = 1
train_df['Age'] = pd.qcut(train_df['Age'], q = 9, labels = [ 0, 15, 20, 24, 26, 28.5, 32, 36, 46 ])    
train_df['Age'] = train_df['Age'].astype('int32')              
train_df['Embarked'] = train_df['Embarked'].fillna('S')
train_df['Fare'] = train_df['Fare'].fillna(7.25)

train_df['Cabin'] = train_df['Cabin'].apply(captureCabin) 
counts = train_df.groupby(['Title', 'Pclass', 'Sex', 'Cabin'])['Cabin'].count()
    
for index, value in counts.items():
    train_df.loc[(train_df['Cabin'].isna() == True) & (train_df['Title'] == index[0]) & 
                   (train_df['Pclass'] == index[1]) & (train_df['Sex'] == index[2]), 'Cabin'] = counts[index[0], index[1], index[2]].idxmax()      

fill_by_classification(train_df, train_df, 'Cabin', [ 'Title', 'Pclass', 'Sex', 'Embarked', 'Age', 'Fare' ])

train_df.loc[train_df['Cabin'] == 'T', 'Cabin'] = 'A'
train_df.loc[(train_df['Cabin'] == 'B') | (train_df['Cabin'] == 'C'), 'Cabin'] = 'A'
train_df.loc[(train_df['Cabin'] == 'D') | (train_df['Cabin'] == 'E'), 'Cabin'] = 'D'
train_df.loc[(train_df['Cabin'] == 'F') | (train_df['Cabin'] == 'G'), 'Cabin'] = 'G'
  
## 3. Family Size (SibSp + Parch + 1)column 'Alone', 'Small', 'Medium', 'Big'

train_df['Size'] = train_df['SibSp'] + train_df['Parch'] + 1
train_df['Size'] = train_df['Size'].apply(captureSize)

train_df['Fare'] = pd.qcut(train_df['Fare'], 6, labels=[0, 5, 10, 20, 40, 80 ])

train_df.drop(columns=['Name', 'Ticket', 'SibSp', 'Parch' ], inplace = True)



  

total = len(train_df)
alives = len(train_df[train_df['Survived'] == 0]) 
dead = len(train_df[train_df['Survived'] == 1]) 

males = len(train_df[train_df['Sex'] == 'male'])
females = len(train_df[train_df['Sex'] == 'female'])
pclass1 = len(train_df[train_df['Pclass'] == 1])
pclass2 = len(train_df[train_df['Pclass'] == 2])
pclass3 = len(train_df[train_df['Pclass'] == 3])
embarkedS = len(train_df[train_df['Embarked'] == 'S'])
embarkedC = len(train_df[train_df['Embarked'] == 'C'])
embarkedQ = len(train_df[train_df['Embarked'] == 'Q'])
cabinA = len(train_df[train_df['Cabin'] == 'A'])
cabinD = len(train_df[train_df['Cabin'] == 'D'])
cabinG = len(train_df[train_df['Cabin'] == 'G'])
size1 = len(train_df[train_df['Size'] == 1])
size2 = len(train_df[train_df['Size'] == 2])
size3 = len(train_df[train_df['Size'] == 3])
size5 = len(train_df[train_df['Size'] == 5])
titleMr = len(train_df[train_df['Title'] == 'Mr'])
titleMrs = len(train_df[train_df['Title'] == 'Mrs'])
titleMiss = len(train_df[train_df['Title'] == 'Miss'])
titleGramPa = len(train_df[train_df['Title'] == 'GramPa'])
titleMaster = len(train_df[train_df['Title'] == 'Master'])
titleGirl = len(train_df[train_df['Title'] == 'Girl'])
titleGramMa = len(train_df[train_df['Title'] == 'GramMa'])
titleBaron = len(train_df[train_df['Title'] == 'Baron'])
titleClergy = len(train_df[train_df['Title'] == 'Clergy'])
titleBoy = len(train_df[train_df['Title'] == 'Boy'])
titleDoctor = len(train_df[train_df['Title'] == 'Doctor'])
titleArmy = len(train_df[train_df['Title'] == 'Army'])
titleBaronness = len(train_df[train_df['Title'] == 'Baronness'])
titleNurse = len(train_df[train_df['Title'] == 'Nurse'])




## Prior(ALIVE) = Total(ALIVE) / TOTAL(ALL)
palives = alives / total
## Prior(DEAD) = Total(DEAD) / TOTAL(ALL)
pdead = dead / total
## Prior(Pclass=1) = Total(Pclass=1) / Total(Pclass=1,2,3)
ppclass1 = pclass1 / total
## Prior(Pclass=2) = Total(Pclass=2) / Total(Pclass=1,2,3)
ppclass2 = pclass2 / total
## Prior(Pclass=3) = Total(Pclass=3) / Total(Pclass=1,2,3)
ppclass3 = pclass3 / total
## Prior(Embarked=S) = Total(Embarked=S) / Total(Embarked=S,C,Q)
pembarkedS = embarkedS / total
## Prior(Embarked=C) = Total(Embarked=C) / Total(Embarked=S,C,Q)
pembarkedC = embarkedC / total
## Prior(Embarked=Q) = Total(Embarked=Q) / Total(Embarked=S,C,Q)
pembarkedQ = embarkedQ / total
## Prior(Cabin=A) = Total(Cabin=A) / Total(Cabin=A,D,G)
pcabinA = cabinA / total
## Prior(Cabin=D) = Total(Cabin=D) / Total(Cabin=A,D,G)
pcabinD = cabinD / total
## Prior(Cabin=G) = Total(Cabin=G) / Total(Cabin=A,D,G)
pcabinG = cabinG / total
## Prior(Size=1) = Total(Size=1) / Total(Size=1,2,3,5)
psize1 = size1 / total
## Prior(Size=2) = Total(Size=2) / Total(Size=1,2,3,5)
psize2 = size2 / total
## Prior(Size=3) = Total(Size=3) / Total(Size=1,2,3,5)
psize3 = size3 / total
## Prior(Size=5) = Total(Size=5) / Total(Size=1,2,3,5)
psize5 = size5 / total
## Prior(Title=Mr) = Total(Title=Mr) / Total(All)
ptitleMr = titleMr / total
## Prior(Title=Mrs) = Total(Title=Mrs) / Total(All)
ptitleMrs = titleMrs / total
## Prior(Title=Miss) = Total(Title=Miss) / Total(All)
ptitleMiss = titleMiss / total
## Prior(Title=GramPa) = Total(Title=GramPa) / Total(All)
ptitleGramPa = titleGramPa / total
## Prior(Title=Master) = Total(Title=Master) / Total(All)
ptitleMaster = titleMaster / total
## Prior(Title=Girl) = Total(Title=Girl) / Total(All)
ptitleGirl = titleGirl / total
## Prior(Title=GramMa) = Total(Title=GramMa) / Total(All)
ptitleGramMa = titleGramMa / total
## Prior(Title=Baron) = Total(Title=Baron) / Total(All)
ptitleBaron = titleBaron / total
## Prior(Title=Clergy) = Total(Title=Clergy) / Total(All)
ptitleClergy = titleClergy / total
## Prior(Title=Boy) = Total(Title=Boy) / Total(All)
ptitleBoy = titleBoy / total
## Prior(Title=Doctor) = Total(Title=Doctor) / Total(All)
ptitleDoctor = titleDoctor / total
## Prior(Title=Army) = Total(Title=Army) / Total(All)
ptitleArmy = titleArmy / total
## Prior(Title=Baronness) = Total(Title=Baronness) / Total(All)
ptitleBaronness = titleBaronness / total
## Prior(Title=Nurse) = Total(Title=Nurse) / Total(All)
ptitleNurse = titleNurse / total

print("Prob(ALIVE): %0.4f" % palives)
print("Prob(DEAD): %0.4f" % pdead)
print("Prob(Pclass=1): %0.4f" % ppclass1)
print("Prob(Pclass=2): %0.4f" % ppclass2)
print("Prob(Pclass=3): %0.4f" % ppclass3)
print("Prob(Embarked=S): %0.4f" % pembarkedS)
print("Prob(Embarked=C): %0.4f" % pembarkedC)
print("Prob(Embarked=Q): %0.4f" % pembarkedQ)
print("Prob(Cabin=A): %0.4f" % pcabinA)
print("Prob(Cabin=D): %0.4f" % pcabinD)
print("Prob(Cabin=G): %0.4f" % pcabinG)
print("Prob(Size=1): %0.4f" % psize1)
print("Prob(Size=2): %0.4f" % psize2)
print("Prob(Size=3): %0.4f" % psize3)
print("Prob(Size=5): %0.4f" % psize5)
print("Prob(Title=Mr): %0.4f" % ptitleMr)
print("Prob(Title=Mrs): %0.4f" % ptitleMrs)
print("Prob(Title=Miss): %0.4f" % ptitleMiss)
print("Prob(Title=GramPa): %0.4f" % ptitleGramPa)
print("Prob(Title=Master): %0.4f" % ptitleMaster)
print("Prob(Title=Girl): %0.4f" % ptitleGirl)
print("Prob(Title=GramMa): %0.4f" % ptitleGramMa)
print("Prob(Title=Baron): %0.4f" % ptitleBaron)
print("Prob(Title=Clergy): %0.4f" % ptitleClergy)
print("Prob(Title=Boy): %0.4f" % ptitleBoy)
print("Prob(Title=Doctor): %0.4f" % ptitleDoctor)
print("Prob(Title=Army): %0.4f" % ptitleArmy)
print("Prob(Title=Baronness): %0.4f" % ptitleBaronness)
print("Prob(Title=Nurse): %0.4f" % ptitleNurse)


print("=================================================")

## P(Pclass=1|Alive) = Total(Pclass=1 & Alive)/Total(Alive) 
## P(Pclass=1|Alive) = P(Alive|Pclass=1)*P(Pclass=1) / P(Alive)
## P(Alive|Pclass=1) = P(Pclass=1|Alive) * P(Alive) / P(Pclass=1)
pclass1Alive = len(train_df[(train_df['Pclass'] == 1) & (train_df['Survived'] == 1)]) / alives
palivepclass1 = pclass1Alive * palives / ppclass1 + ADJUSTMENT
print("Prob(Alive|Pclass=1): %0.4f" % palivepclass1)

## P(Pclass=2|Alive) = Total(Pclass=2 & Alive)/Total(Alive) 
## P(Pclass=2|Alive) = P(Alive|Pclass=1)*P(Pclass=1) / P(Alive)
## P(Alive|Pclass=2) = P(Pclass=2|Alive) * P(Alive) / P(Pclass=2)
pclass2Alive = len(train_df[(train_df['Pclass'] == 2) & (train_df['Survived'] == 1)]) / alives
palivepclass2 = pclass2Alive * palives / ppclass2 + ADJUSTMENT
print("Prob(Alive|Pclass=2): %0.4f" % palivepclass2)

## P(Pclass=3|Alive) = Total(Pclass=3 & Alive)/Total(Alive) 
## P(Pclass=3|Alive) = P(Alive|Pclass=3)*P(Pclass=1) / P(Alive)
## P(Alive|Pclass=3) = P(Pclass=3|Alive) * P(Alive) / P(Pclass=3)
pclass3Alive = len(train_df[(train_df['Pclass'] == 3) & (train_df['Survived'] == 1)]) / alives
palivepclass3 = pclass3Alive * palives / ppclass3 + ADJUSTMENT
print("Prob(Alive|Pclass=3): %0.4f" % palivepclass3)


## P(Pclass=1|Dead) = Total(Pclass=1 & Dead)/Total(Dead) 
## P(Pclass=1|Dead) = P(Dead|Pclass=1)*P(Pclass=1) / P(Dead)
## P(Dead|Pclass=1) = P(Pclass=1|Dead) * P(Dead) / P(Pclass=1)
pclass1Dead = len(train_df[(train_df['Pclass'] == 1) & (train_df['Survived'] == 0)]) / dead
pdeadpclass1 = pclass1Dead * pdead / ppclass1 + ADJUSTMENT
print("Prob(Dead|Pclass=1): %0.4f" % palivepclass1)

## P(Pclass=2|Dead) = Total(Pclass=2 & Dead)/Total(Dead) 
## P(Pclass=2|Dead) = P(Dead|Pclass=1)*P(Pclass=2) / P(Dead)
## P(Dead|Pclass=2) = P(Pclass=2|Dead) * P(Dead) / P(Pclass=2)
pclass2Dead = len(train_df[(train_df['Pclass'] == 2) & (train_df['Survived'] == 0)]) / dead
pdeadpclass2 = pclass2Dead * pdead / ppclass2 + ADJUSTMENT
print("Prob(Dead|Pclass=2): %0.4f" % pdeadpclass2)

## P(Pclass=3|Dead) = Total(Pclass=3 & Dead)/Total(Dead) 
## P(Pclass=3|Dead) = P(Dead|Pclass=3)*P(Pclass=1) / P(Dead)
## P(Dead|Pclass=3) = P(Pclass=3|Dead) * P(Dead) / P(Pclass=3)
pclass3Dead = len(train_df[(train_df['Pclass'] == 3) & (train_df['Survived'] == 0)]) / dead
pdeadpclass3 = pclass3Dead * pdead / ppclass3 + ADJUSTMENT
print("Prob(Dead|Pclass=3): %0.4f" % pdeadpclass3)

## P(Embarked=S|Alive) = Total(Embarked=S & Alive) / Total(Alive)
## P(Embarked=S|Alive) = P(Alive|Embarked=S) * P(Embarked=S) / P(Alive)
## P(Alive|Embarked=S) = P(Embarked=S|Alive) * P(Alive) / P(Embarked=S)
pembarkedSAlive = len(train_df[(train_df['Embarked'] == 'S') & (train_df['Survived'] == 1)]) / alives
paliveEmbarkedS = pembarkedSAlive * palives / pembarkedS + ADJUSTMENT
print("Prob(Alive|Embarked=S): %0.4f" % paliveEmbarkedS)

## P(Embarked=C|Alive) = Total(Embarked=C & Alive) / Total(Alive)
## P(Embarked=C|Alive) = P(Alive|Embarked=C) * P(Embarked=C) / P(Alive)
## P(Alive|Embarked=C) = P(Embarked=C|Alive) * P(Alive) / P(Embarked=C)
pembarkedCAlive = len(train_df[(train_df['Embarked'] == 'C') & (train_df['Survived'] == 1)]) / alives
paliveEmbarkedC = pembarkedCAlive * palives / pembarkedC + ADJUSTMENT
print("Prob(Alive|Embarked=C): %0.4f" % paliveEmbarkedC)

## P(Embarked=Q|Alive) = Total(Embarked=Q & Alive) / Total(Alive)
## P(Embarked=Q|Alive) = P(Alive|Embarked=Q) * P(Embarked=Q) / P(Alive)
## P(Alive|Embarked=Q) = P(Embarked=Q|Alive) * P(Alive) / P(Embarked=Q)
pembarkedQAlive = len(train_df[(train_df['Embarked'] == 'Q') & (train_df['Survived'] == 1)]) / alives
paliveEmbarkedQ = pembarkedQAlive * palives / pembarkedQ + ADJUSTMENT
print("Prob(Alive|Embarked=Q): %0.4f" % paliveEmbarkedQ)

## P(Embarked=S|Dead) = Total(Embarked=S & Dead) / Total(Dead)
## P(Embarked=S|Dead) = P(Alive|Embarked=S) * P(Embarked=S) / P(Dead)
## P(Dead|Embarked=S) = P(Embarked=S|Dead) * P(Dead) / P(Embarked=S)
pembarkedSDead = len(train_df[(train_df['Embarked'] == 'S') & (train_df['Survived'] == 0)]) / dead
pdeadEmbarkedS = pembarkedSDead * pdead / pembarkedS + ADJUSTMENT
print("Prob(Dead|Embarked=S): %0.4f" % pdeadEmbarkedS)

## P(Embarked=C|Dead) = Total(Embarked=C & Dead) / Total(Dead)
## P(Embarked=C|Dead) = P(Dead|Embarked=C) * P(Embarked=C) / P(Dead)
## P(Dead|Embarked=C) = P(Embarked=C|Dead) * P(Dead) / P(Embarked=C)
pembarkedCDead = len(train_df[(train_df['Embarked'] == 'C') & (train_df['Survived'] == 0)]) / dead
pdeadEmbarkedC = pembarkedCDead * pdead / pembarkedC + ADJUSTMENT
print("Prob(Dead|Embarked=C): %0.4f" % pdeadEmbarkedC)

## P(Embarked=Q|Dead) = Total(Embarked=Q & Dead) / Total(Dead)
## P(Embarked=Q|Dead) = P(Dead|Embarked=Q) * P(Embarked=Q) / P(Dead)
## P(Dead|Embarked=Q) = P(Embarked=Q|Dead) * P(Dead) / P(Embarked=Q)
pembarkedQDead = len(train_df[(train_df['Embarked'] == 'Q') & (train_df['Survived'] == 0)]) / dead
pdeadEmbarkedQ = pembarkedQDead * pdead / pembarkedQ + ADJUSTMENT
print("Prob(Dead|Embarked=Q): %0.4f" % pdeadEmbarkedQ)

## P(Cabin=A|Alive) = Total(Cabin=A & Alive) / Total(Alive)
## P(Cabin=A|Alive) = P(Alive|Cabin=A) * P(Cabin=A) / P(Alive)
## P(Alive|Cabin=A) = P(Cabin=A|Alive) * P(Alive) / P(Cabin=A)
pcabinAAlive = len(train_df[(train_df['Cabin'] == 'A') & (train_df['Survived'] == 1)]) / alives
paliveCabinA = pcabinAAlive * palives / pcabinA + ADJUSTMENT
print("Prob(Alive|Cabin=A): %0.4f" % paliveCabinA)

## P(Cabin=D|Alive) = Total(Cabin=D & Alive) / Total(Alive)
## P(Cabin=D|Alive) = P(Alive|Cabin=D) * P(Cabin=D) / P(Alive)
## P(Alive|Cabin=D) = P(Cabin=D|Alive) * P(Alive) / P(Cabin=D)
pcabinDAlive = len(train_df[(train_df['Cabin'] == 'D') & (train_df['Survived'] == 1)]) / alives
paliveCabinD = pcabinDAlive * palives / pcabinD + ADJUSTMENT
print("Prob(Alive|Cabin=D): %0.4f" % paliveCabinD)

## P(Cabin=G|Alive) = Total(Cabin=G & Alive) / Total(Alive)
## P(Cabin=G|Alive) = P(Alive|Cabin=G) * P(Cabin=G) / P(Alive)
## P(Alive|Cabin=G) = P(Cabin=G|Alive) * P(Alive) / P(Cabin=G)
pcabinGAlive = len(train_df[(train_df['Cabin'] == 'G') & (train_df['Survived'] == 1)]) / alives
paliveCabinG = pcabinGAlive * palives / pcabinG + ADJUSTMENT
print("Prob(Alive|Cabin=G): %0.4f" % paliveCabinG)

## P(Cabin=A|Dead) = Total(Cabin=A & Dead) / Total(Dead)
## P(Cabin=A|Dead) = P(Alive|Cabin=A) * P(Cabin=A) / P(Dead)
## P(Dead|Cabin=A) = P(Cabin=A|Dead) * P(Dead) / P(Cabin=A)
pcabinADead = len(train_df[(train_df['Cabin'] == 'A') & (train_df['Survived'] == 0)]) / dead
pdeadCabinA = pcabinADead * pdead / pcabinA + ADJUSTMENT
print("Prob(Dead|Cabin=A): %0.4f" % pdeadCabinA)

## P(Cabin=D|Dead) = Total(Cabin=D & Dead) / Total(Dead)
## P(Cabin=D|Dead) = P(Dead|cabin=D) * P(Cabin=D) / P(Dead)
## P(Dead|Cabin=D) = P(Cabin=D|Alive) * P(Dead) / P(Cabin=D)
pcabinDDead = len(train_df[(train_df['Cabin'] == 'D') & (train_df['Survived'] == 0)]) / dead
pdeadCabinD = pcabinDDead * pdead / pcabinD + ADJUSTMENT
print("Prob(Dead|Cabin=D): %0.4f" % pdeadCabinD)

## P(Cabin=G|Dead) = Total(Cabin=G & Dead) / Total(Dead)
## P(Cabin=G|Dead) = P(Dead|Cabin=G) * P(Cabin=G) / P(Dead)
## P(Dead|Cabin=G) = P(Cabin=G|Dead) * P(Dead) / P(Cabin=G)
pcabinGDead = len(train_df[(train_df['Cabin'] == 'G') & (train_df['Survived'] == 0)]) / dead
pdeadCabinG = pcabinGDead * pdead / pcabinG + ADJUSTMENT
print("Prob(Dead|Cabin=G): %0.4f" % pdeadCabinG)

## P(Size=1|Alive) = Total(Size=1 & Alive) / Total(Alive)
## P(Size=1|Alive) = P(Alive|Size=1) * PSize=1) / P(Alive)
## P(Alive|Size = 1) = P(Size=1|Alive) * P(Alive) / P(Size=1)
psize1Alive = len(train_df[(train_df['Size'] == 1) & (train_df['Survived'] == 1)]) / alives
paliveSize1 = psize1Alive * palives / psize1 + ADJUSTMENT
print("Prob(Alive|Size=1): %0.4f" % paliveSize1)

## P(Size=2|Alive) = Total(Size=2 & Alive) / Total(Alive)
## P(Size=2|Alive) = P(Alive|Size=2) * PSize=2) / P(Alive)
## P(Alive|Size=2) = P(Size=2|Alive) * P(Alive) / P(Size=2)
psize2Alive = len(train_df[(train_df['Size'] == 2) & (train_df['Survived'] == 1)]) / alives
paliveSize2 = psize2Alive * palives / psize2 + ADJUSTMENT
print("Prob(Alive|Size=2): %0.4f" % paliveSize2)

## P(Size=3|Alive) = Total(Size=3 & Alive) / Total(Alive)
## P(Size=3|Alive) = P(Alive|Size=3) * PSize=3) / P(Alive)
## P(Alive|Size=3) = P(Size=3|Alive) * P(Alive) / P(Size=3)
psize3Alive = len(train_df[(train_df['Size'] == 3) & (train_df['Survived'] == 1)]) / alives
paliveSize3 = psize3Alive * palives / psize3 + ADJUSTMENT
print("Prob(Alive|Size=3): %0.4f" % paliveSize3)

## P(Size=5|Alive) = Total(Size=5 & Alive) / Total(Alive)
## P(Size=5|Alive) = P(Alive|Size=5) * PSize=5) / P(Alive)
## P(Alive|Size=5) = P(Size=5|Alive) * P(Alive) / P(Size=5)
psize5Alive = len(train_df[(train_df['Size'] == 5) & (train_df['Survived'] == 1)]) / alives
paliveSize5 = psize5Alive * palives / psize5 + ADJUSTMENT
print("Prob(Alive|Size=5): %0.4f" % paliveSize5)

## P(Size=1|Dead) = Total(Size=1 & Dead) / Total(Dead)
## P(Size=1|Dead) = P(Dead|Size=1) * PSize=1) / P(Dead)
## P(Alive|Size = 1) = P(Size=1|Dead) * P(Dead) / P(Size=1)
psize1Dead = len(train_df[(train_df['Size'] == 1) & (train_df['Survived'] == 0)]) / dead
pdeadSize1 = psize1Dead * pdead / psize1 + ADJUSTMENT
print("Prob(Dead|Size=1): %0.4f" % pdeadSize1)

## P(Size=2|Dead) = Total(Size=2 & Dead) / Total(Dead)
## P(Size=2|Dead) = P(Dead|Size=2) * PSize=2) / P(Dead)
## P(Dead|Size=2) = P(Size=2|Dead) * P(Dead) / P(Size=2)
psize2Dead = len(train_df[(train_df['Size'] == 2) & (train_df['Survived'] == 0)]) / dead
pdeadSize2 = psize2Dead * pdead / psize2 + ADJUSTMENT
print("Prob(Dead|Size=2): %0.4f" % pdeadSize2)

## P(Size=3|Dead) = Total(Size=3 & Dead) / Total(Dead)
## P(Size=3|Dead) = P(Dead|Size=3) * PSize=3) / P(Dead)
## P(Dead|Size=3) = P(Size=3|Dead) * P(Dead) / P(Size=3)
psize3Dead = len(train_df[(train_df['Size'] == 3) & (train_df['Survived'] == 0)]) / dead
pdeadSize3 = psize3Alive * pdead / psize3 + ADJUSTMENT
print("Prob(Dead|Size=3): %0.4f" % pdeadSize3)

## P(Size=5|Dead) = Total(Size=5 & Dead) / Total(Dead)
## P(Size=5|Dead) = P(Dead|Size=5) * PSize=5) / P(AliveDead)
## P(Dead|Size=5) = P(Size=5|Dead) * P(Dead) / P(Size=5)
psize5Dead = len(train_df[(train_df['Size'] == 5) & (train_df['Survived'] == 0)]) / dead
pdeadSize5 = psize5Dead * pdead / psize5 + ADJUSTMENT
print("Prob(Dead|Size=5): %0.4f" % pdeadSize5)

## P(Title=Mr|Alive) = Total(Tital=Mr & Alive) / Total(Alive)
## P(Title=Mr|Alive) = P(Alive|Title=Mr) * P(Title=Mr) / P(Alive)
## P(Alive|Title=Mr) = P(Title=Mr|Alive) * P(Alive) / P(Title=Mr)
ptitleMrAlive = len(train_df[(train_df['Title'] == 'Mr') & (train_df['Survived'] == 1)]) / alives
paliveTitleMr = ptitleMrAlive * palives / ptitleMr + ADJUSTMENT
print("Prob(Alive|Title=Mr): %0.4f" % paliveTitleMr)

## P(Title=Mrs|Alive) = Total(Tital=Mrs & Alive) / Total(Alive)
## P(Title=Mrs|Alive) = P(Alive|Title=Mrs) * P(Title=Mrs) / P(Alive)
## P(Alive|Title=Mrs) = P(Title=Mrs|Alive) * P(Alive) / P(Title=Mrs)
ptitleMrsAlive = len(train_df[(train_df['Title'] == 'Mrs') & (train_df['Survived'] == 1)]) / alives
paliveTitleMrs = ptitleMrsAlive * palives / ptitleMrs + ADJUSTMENT
print("Prob(Alive|Title=Mrs): %0.4f" % paliveTitleMrs)

## P(Title=Miss|Alive) = Total(Tital=Miss & Alive) / Total(Alive)
## P(Title=Miss|Alive) = P(Alive|Title=Miss) * P(Title=Miss) / P(Alive)
## P(Alive|Title=Miss) = P(Title=Miss|Alive) * P(Alive) / P(Title=Miss)
ptitleMissAlive = len(train_df[(train_df['Title'] == 'Miss') & (train_df['Survived'] == 1)]) / alives
paliveTitleMiss = ptitleMissAlive * palives / ptitleMiss + ADJUSTMENT
print("Prob(Alive|Title=Miss): %0.4f" % paliveTitleMiss)

## P(Title=GramPa|Alive) = Total(Tital=GramPa & Alive) / Total(Alive)
## P(Title=GramPa|Alive) = P(Alive|Title=GramPa) * P(Title=GramPa) / P(Alive)
## P(Alive|Title=GramPa) = P(Title=GramPa|Alive) * P(Alive) / P(Title=Mrs)
ptitleGramPaAlive = len(train_df[(train_df['Title'] == 'GramPa') & (train_df['Survived'] == 1)]) / alives
paliveTitleGramPa = ptitleGramPaAlive * palives / ptitleGramPa + ADJUSTMENT
print("Prob(Alive|Title=GramPa): %0.4f" % paliveTitleGramPa)

## P(Title=Master|Alive) = Total(Tital=Master & Alive) / Total(Alive)
## P(Title=Master|Alive) = P(Alive|Title=Master) * P(Title=Master) / P(Alive)
## P(Alive|Title=Master) = P(Title=Master|Alive) * P(Alive) / P(Title=Master)
ptitleMasterAlive = len(train_df[(train_df['Title'] == 'Master') & (train_df['Survived'] == 1)]) / alives
paliveTitleMaster = ptitleMasterAlive * palives / ptitleMaster + ADJUSTMENT
print("Prob(Alive|Title=Master): %0.4f" % paliveTitleMaster)

## P(Title=Girl|Alive) = Total(Tital=Girl & Alive) / Total(Alive)
## P(Title=Girl|Alive) = P(Alive|Title=Girl) * P(Title=Mrs) / P(Alive)
## P(Alive|Title=Girl) = P(Title=Girl|Alive) * P(Alive) / P(Title=Girl)
ptitleGirlAlive = len(train_df[(train_df['Title'] == 'Girl') & (train_df['Survived'] == 1)]) / alives
paliveTitleGirl = ptitleGirlAlive * palives / ptitleGirl + ADJUSTMENT
print("Prob(Alive|Title=Girl): %0.4f" % paliveTitleGirl)

## P(Title=GramMa|Alive) = Total(Tital=GramMa & Alive) / Total(Alive)
## P(Title=GramMa|Alive) = P(Alive|Title=GramMa) * P(Title=GramMa) / P(Alive)
## P(Alive|Title=GramMa) = P(Title=GramMa|Alive) * P(Alive) / P(Title=GramMa)
ptitleGramMaAlive = len(train_df[(train_df['Title'] == 'GramMa') & (train_df['Survived'] == 1)]) / alives
paliveTitleGramMa = ptitleGramMaAlive * palives / ptitleGramMa + ADJUSTMENT
print("Prob(Alive|Title=GramMa): %0.4f" % paliveTitleGramMa)

## P(Title=Baron|Alive) = Total(Tital=Baron & Alive) / Total(Alive)
## P(Title=Baron|Alive) = P(Alive|Title=Baron) * P(Title=Baron) / P(Alive)
## P(Alive|Title=Baron) = P(Title=Baron|Alive) * P(Alive) / P(Title=Baron)
ptitleBaronAlive = len(train_df[(train_df['Title'] == 'Baron') & (train_df['Survived'] == 1)]) / alives
paliveTitleBaron = ptitleBaronAlive * palives / ptitleBaron + ADJUSTMENT
print("Prob(Alive|Title=Baron): %0.4f" % paliveTitleBaron)

## P(Title=Clergy|Alive) = Total(Tital=Clergy & Alive) / Total(Alive)
## P(Title=Clergy|Alive) = P(Alive|Title=Clergy) * P(Title=Clergy) / P(Alive)
## P(Alive|Title=Clergy) = P(Title=Clergy|Alive) * P(Alive) / P(Title=Clergy)
ptitleClergyAlive = len(train_df[(train_df['Title'] == 'Clergy') & (train_df['Survived'] == 1)]) / alives
paliveTitleClergy = ptitleClergyAlive * palives / ptitleClergy + ADJUSTMENT
print("Prob(Alive|Title=Clergy): %0.4f" % paliveTitleClergy)

## P(Title=Boy|Alive) = Total(Tital=Boy & Alive) / Total(Alive)
## P(Title=Boy|Alive) = P(Alive|Title=Boy) * P(Title=Boy) / P(Alive)
## P(Alive|Title=Boy) = P(Title=Boy|Alive) * P(Alive) / P(Title=Boy)
ptitleBoyAlive = len(train_df[(train_df['Title'] == 'Boy') & (train_df['Survived'] == 1)]) / alives
paliveTitleBoy = ptitleBoyAlive * palives / ptitleBoy + ADJUSTMENT
print("Prob(Alive|Title=Boy): %0.4f" % paliveTitleBoy)

## P(Title=Doctor|Alive) = Total(Tital=Doctor & Alive) / Total(Alive)
## P(Title=Doctor|Alive) = P(Alive|Title=Doctor) * P(Title=Doctor) / P(Alive)
## P(Alive|Title=Doctor) = P(Title=Doctor|Alive) * P(Alive) / P(Title=Doctor)
ptitleDoctorAlive = len(train_df[(train_df['Title'] == 'Doctor') & (train_df['Survived'] == 1)]) / alives
paliveTitleDoctor = ptitleDoctorAlive * palives / ptitleDoctor + ADJUSTMENT
print("Prob(Alive|Title=Doctor): %0.4f" % paliveTitleDoctor)

## P(Title=Army|Alive) = Total(Tital=Army & Alive) / Total(Alive)
## P(Title=Army|Alive) = P(Alive|Title=Army) * P(Title=Army) / P(Alive)
## P(Alive|Title=Army) = P(Title=Army|Alive) * P(Alive) / P(Title=Army)
ptitleArmyAlive = len(train_df[(train_df['Title'] == 'Army') & (train_df['Survived'] == 1)]) / alives
paliveTitleArmy = ptitleArmyAlive * palives / ptitleArmy + ADJUSTMENT
print("Prob(Alive|Title=Army): %0.4f" % paliveTitleArmy)

## P(Title=Baronness|Alive) = Total(Tital=Baronness & Alive) / Total(Alive)
## P(Title=Baronness|Alive) = P(Alive|Title=Baronness) * P(Title=Baronness) / P(Alive)
## P(Alive|Title=Baronness) = P(Title=ArmyBaronness|Alive) * P(Alive) / P(Title=Baronness)
ptitleBaronnessAlive = len(train_df[(train_df['Title'] == 'Baronness') & (train_df['Survived'] == 1)]) / alives
paliveTitleBaronness = ptitleBaronnessAlive * palives / ptitleBaronness + ADJUSTMENT
print("Prob(Alive|Title=Baronness): %0.4f" % paliveTitleBaronness)

## P(Title=Nurse|Alive) = Total(Tital=Nurse & Alive) / Total(Alive)
## P(Title=Nurse|Alive) = P(Alive|Title=Nurse) * P(Title=Nurse) / P(Alive)
## P(Alive|Title=Nurse) = P(Title=Nurse|Alive) * P(Alive) / P(Title=Nurse)
ptitleNurseAlive = len(train_df[(train_df['Title'] == 'Nurse') & (train_df['Survived'] == 1)]) / alives
paliveTitleNurse = ptitleNurseAlive * palives / ptitleNurse + ADJUSTMENT
print("Prob(Alive|Title=Nurse): %0.4f" % paliveTitleNurse)

## P(Title=Mr|Dead) = Total(Tital=Mr & Dead) / Total(Dead)
## P(Title=Mr|Dead) = P(Alive|Title=Mr) * P(Title=Mr) / P(Dead)
## P(Dead|Title=Mr) = P(Title=Mr|Dead) * P(Dead) / P(Title=Mr)
ptitleMrDead = len(train_df[(train_df['Title'] == 'Mr') & (train_df['Survived'] == 0)]) / dead
pdeadTitleMr = ptitleMrDead * pdead / ptitleMr + ADJUSTMENT
print("Prob(Dead|Title=Mr): %0.4f" % pdeadTitleMr)

## P(Title=Mrs|Dead) = Total(Tital=Mrs & Dead) / Total(Dead)
## P(Title=Mrs|Dead) = P(Dead|Title=Mrs) * P(Title=Mrs) / P(Dead)
## P(Dead|Title=Mrs) = P(Title=Mrs|Dead) * P(Dead) / P(Title=Mrs)
ptitleMrsDead = len(train_df[(train_df['Title'] == 'Mrs') & (train_df['Survived'] == 0)]) / dead
pdeadTitleMrs = ptitleMrsDead * pdead / ptitleMrs + ADJUSTMENT
print("Prob(Dead|Title=Mrs): %0.4f" % pdeadTitleMrs)

## P(Title=Miss|Dead) = Total(Tital=Miss & Dead) / Total(Dead)
## P(Title=Miss|Dead) = P(Dead|Title=Miss) * P(Title=Miss) / P(Dead)
## P(Dead|Title=Miss) = P(Title=Miss|Dead) * P(Dead) / P(Title=Miss)
ptitleMissDead = len(train_df[(train_df['Title'] == 'Miss') & (train_df['Survived'] == 0)]) / dead
pdeadTitleMiss = ptitleMissDead * pdead / ptitleMiss + ADJUSTMENT
print("Prob(Dead|Title=Miss): %0.4f" % pdeadTitleMiss)

## P(Title=GramPa|Dead) = Total(Tital=GramPa & Dead) / Total(Dead)
## P(Title=GramPa|Dead) = P(Dead|Title=GramPa) * P(Title=GramPa) / P(Dead)
## P(Dead|Title=GramPa) = P(Title=GramPa|Dead) * P(Dead) / P(Title=Mrs)
ptitleGramPaDead = len(train_df[(train_df['Title'] == 'GramPa') & (train_df['Survived'] == 0)]) / dead
pdeadTitleGramPa = ptitleGramPaDead * pdead / ptitleGramPa + ADJUSTMENT
print("Prob(Dead|Title=GramPa): %0.4f" % pdeadTitleGramPa)

## P(Title=Master|Dead) = Total(Tital=Master & Dead) / Total(Dead)
## P(Title=Master|Dead) = P(Dead|Title=Master) * P(Title=Master) / P(Dead)
## P(Dead|Title=Master) = P(Title=Master|Dead) * P(Dead) / P(Title=Master)
ptitleMasterDead = len(train_df[(train_df['Title'] == 'Master') & (train_df['Survived'] == 0)]) / dead
pdeadTitleMaster = ptitleMasterDead * pdead / ptitleMaster + ADJUSTMENT
print("Prob(Dead|Title=Master): %0.4f" % pdeadTitleMaster)

## P(Title=Girl|Dead) = Total(Tital=Girl & Dead) / Total(Dead)
## P(Title=Girl|Dead) = P(Dead|Title=Girl) * P(Title=Mrs) / P(Dead)
## P(Dead|Title=Girl) = P(Title=Girl|Dead) * P(Dead) / P(Title=Girl)
ptitleGirlDead = len(train_df[(train_df['Title'] == 'Girl') & (train_df['Survived'] == 0)]) / dead
pdeadTitleGirl = ptitleGirlDead * pdead / ptitleGirl + ADJUSTMENT
print("Prob(Dead|Title=Girl): %0.4f" % pdeadTitleGirl)

## P(Title=GramMa|Dead) = Total(Tital=GramMa & Dead) / Total(Dead)
## P(Title=GramMa|Dead) = P(Dead|Title=GramMa) * P(Title=GramMa) / P(Dead)
## P(Dead|Title=GramMa) = P(Title=GramMa|Dead) * P(Dead) / P(Title=GramMa)
ptitleGramMaDead = len(train_df[(train_df['Title'] == 'GramMa') & (train_df['Survived'] == 0)]) / dead
pdeadTitleGramMa = ptitleGramMaDead * pdead / ptitleGramMa + ADJUSTMENT
print("Prob(Dead|Title=GramMa): %0.4f" % pdeadTitleGramMa)

## P(Title=Baron|Dead) = Total(Tital=Baron & Dead) / Total(Dead)
## P(Title=Baron|Dead) = P(Dead|Title=Baron) * P(Title=Baron) / P(Dead)
## P(Dead|Title=Baron) = P(Title=Baron|Dead) * P(Dead) / P(Title=Baron)
ptitleBaronDead = len(train_df[(train_df['Title'] == 'Baron') & (train_df['Survived'] == 0)]) / dead
pdeadTitleBaron = ptitleBaronDead * pdead / ptitleBaron + ADJUSTMENT
print("Prob(Dead|Title=Baron): %0.4f" % pdeadTitleBaron)

## P(Title=Clergy|Dead) = Total(Tital=Clergy & Dead) / Total(Dead)
## P(Title=Clergy|Dead) = P(Dead|Title=Clergy) * P(Title=Clergy) / P(Dead)
## P(Dead|Title=Clergy) = P(Title=Clergy|Dead) * P(Dead) / P(Title=Clergy)
ptitleClergyDead = len(train_df[(train_df['Title'] == 'Clergy') & (train_df['Survived'] == 0)]) / dead
pdeadTitleClergy = ptitleClergyDead * pdead / ptitleClergy + ADJUSTMENT
print("Prob(Dead|Title=Clergy): %0.4f" % pdeadTitleClergy)

## P(Title=Boy|Dead) = Total(Tital=Boy & Dead) / Total(Dead)
## P(Title=Boy|Dead) = P(Dead|Title=Boy) * P(Title=Boy) / P(Dead)
## P(Dead|Title=Boy) = P(Title=Boy|Dead) * P(Dead) / P(Title=Boy)
ptitleBoyDead = len(train_df[(train_df['Title'] == 'Boy') & (train_df['Survived'] == 0)]) / dead
pdeadTitleBoy = ptitleBoyDead * pdead / ptitleBoy + ADJUSTMENT
print("Prob(Dead|Title=Boy): %0.4f" % pdeadTitleBoy)

## P(Title=Doctor|Dead) = Total(Tital=Doctor & Dead) / Total(Dead)
## P(Title=Doctor|Dead) = P(Dead|Title=Doctor) * P(Title=Doctor) / P(Dead)
## P(Dead|Title=Doctor) = P(Title=Doctor|Dead) * P(Dead) / P(Title=Doctor)
ptitleDoctorDead = len(train_df[(train_df['Title'] == 'Doctor') & (train_df['Survived'] == 0)]) / dead
pDeadTitleDoctor = ptitleDoctorDead * pdead / ptitleDoctor + ADJUSTMENT
print("Prob(Dead|Title=Doctor): %0.4f" % pDeadTitleDoctor)

## P(Title=Army|Dead) = Total(Tital=Army & Dead) / Total(Dead)
## P(Title=Army|Dead) = P(Dead|Title=Army) * P(Title=Army) / P(Dead)
## P(Dead|Title=Army) = P(Title=Army|Dead) * P(Dead) / P(Title=Army)
ptitleArmyDead = len(train_df[(train_df['Title'] == 'Army') & (train_df['Survived'] == 0)]) / dead
pdeadTitleArmy = ptitleArmyDead * pdead / ptitleArmy + ADJUSTMENT
print("Prob(Dead|Title=Army): %0.4f" % pdeadTitleArmy) 

## P(Title=Baronness|Dead) = Total(Tital=Baronness & Dead) / Total(Dead)
## P(Title=Baronness|Dead) = P(Dead|Title=Baronness) * P(Title=Baronness) / P(Dead)
## P(Dead|Title=Baronness) = P(Title=ArmyBaronness|Dead) * P(Dead) / P(Title=Baronness)
ptitleBaronnessDead = len(train_df[(train_df['Title'] == 'Baronness') & (train_df['Survived'] == 0)]) / dead
pdeadTitleBaronness = ptitleBaronnessDead * pdead / ptitleBaronness + ADJUSTMENT
print("Prob(Dead|Title=Baronness): %0.4f" % pdeadTitleBaronness)

## P(Title=Nurse|Dead) = Total(Tital=Nurse & Dead) / Total(Dead)
## P(Title=Nurse|Dead) = P(Dead|Title=Nurse) * P(Title=Nurse) / P(Dead)
## P(Dead|Title=Nurse) = P(Title=Nurse|Dead) * P(Dead) / P(Title=Nurse)
ptitleNurseDead = len(train_df[(train_df['Title'] == 'Nurse') & (train_df['Survived'] == 0)]) / dead
pdeadTitleNurse = ptitleNurseDead* pdead / ptitleNurse + ADJUSTMENT
print("Prob(Dead|Title=Nurse): %0.4f" % pdeadTitleNurse)


bayes = {}
bayes['A'] = 0.6162
bayes['D'] = 0.3838

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