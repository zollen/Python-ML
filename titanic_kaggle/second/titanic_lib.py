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

def navieBayes(train_df):
    
    ADJUSTMENT = 0.0001
    
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
    age0 = len(train_df[train_df['Age'] == 0])
    age15 = len(train_df[train_df['Age'] == 15])
    age20 = len(train_df[train_df['Age'] == 20])
    age24 = len(train_df[train_df['Age'] == 24])
    age26 = len(train_df[train_df['Age'] == 26])
    age28 = len(train_df[train_df['Age'] == 28])
    age32 = len(train_df[train_df['Age'] == 32])
    age36 = len(train_df[train_df['Age'] == 36])
    age46 = len(train_df[train_df['Age'] == 46])
    fare0 = len(train_df[train_df['Fare'] == 0])
    fare10 = len(train_df[train_df['Fare'] == 10])
    fare20 = len(train_df[train_df['Fare'] == 20])
    fare30 = len(train_df[train_df['Fare'] == 30])
    fare40 = len(train_df[train_df['Fare'] == 40])
    fare80 = len(train_df[train_df['Fare'] == 80])


    ## Prior(ALIVE) = Total(ALIVE) / TOTAL(ALL)
    palives = alives / total
    ## Prior(DEAD) = Total(DEAD) / TOTAL(ALL)
    pdead = dead / total
    ## Prior(Sex=male) = Total(mail) / Total(all)
    pmales = males / total
    ## Prior(Sex=female) = Total(female) / Total(all)
    pfemales = females / total
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
    ## Prior(Age=0) = Total(age9) / total
    page0 = age0 / total
    ## Prior(Age=15) = Total(age15) / total
    page15 = age15 / total
    ## Prior(Age=20) = Total(age20) / total
    page20 = age20 / total
    ## Prior(Age=24) = Total(age24) / total
    page24 = age24 / total
    ## Prior(Age=26) = Total(age26) / total
    page26 = age26 / total
    ## Prior(Age=28.5) = Total(age28.5) / total
    page28 = age28 / total
    ## Prior(Age=32) = Total(age32) / total
    page32 = age32 / total
    ## Prior(Age=36) = Total(age36) / total
    page36 = age36 / total
    ## Prior(Age=46) = Total(age46) / total
    page46 = age46 / total
    ## Prior(Fare=0) = Total(Fare=0) / total
    pfare0 = fare0 / total
    ## Prior(Fare=10) = Total(Fare=10) / total
    pfare10 = fare10 / total
    ## Prior(Fare=20) = Total(Fare=20) / total
    pfare20 = fare20 / total
    ## Prior(Fare=30) = Total(Fare=30) / total
    pfare30 = fare30 / total
    ## Prior(Fare=40) = Total(Fare=40) / total
    pfare40 = fare40 / total
    ## Prior(Fare=80) = Total(Fare=80) / total
    pfare80 = fare80 / total
    
    ## P(Sex=male|Alive) = Total(Sex=male & Alive) / Total(Alive)
    ## P(Sex=male|Alive) = P(Alive|Sex=male) * P(Sex=male) / P(Alive)
    ## P(Alive|Sex=male) = P(Sex=male|Alive) * P(Alive) / P(Sex=male)
    pmaleAlive = len(train_df[(train_df['Sex'] == 'male') & (train_df['Survived'] == 1)]) / alives
    paliveMale = pmaleAlive * palives / pmales + ADJUSTMENT

    ## P(Sex=female|Alive) = Total(Sex=female & Alive) / Total(Alive)
    ## P(Sex=female|Alive) = P(Alive|Sex=female) * P(Sex=female) / P(Alive)
    ## P(Alive|Sex=female) = P(Sex=female|Alive) * P(Alive) / P(Sex=female)
    pfemaleAlive = len(train_df[(train_df['Sex'] == 'female') & (train_df['Survived'] == 1)]) / alives
    paliveFemale = pfemaleAlive * palives / pfemales + ADJUSTMENT

    ## P(Sex=male|Dead) = Total(Sex=male & Dead) / Total(Dead)
    ## P(Sex=male|Dead) = P(Dead|Sex=male) * P(Sex=male) / P(Dead)
    ## P(Dead|Sex=male) = P(Sex=male|Dead) * P(Dead) / P(Sex=male)
    pmaleDead = len(train_df[(train_df['Sex'] == 'male') & (train_df['Survived'] == 0)]) / dead
    pdeadMale = pmaleDead * pdead / pmales + ADJUSTMENT

    ## P(Sex=female|Dead) = Total(Sex=female & Dead) / Total(Dead)
    ## P(Sex=female|Dead) = P(Dead|Sex=female) * P(Sex=female) / P(Dead)
    ## P(Dead|Sex=female) = P(Sex=female|Dead) * P(Dead) / P(Sex=female)
    pfemaleDead = len(train_df[(train_df['Sex'] == 'female') & (train_df['Survived'] == 0)]) / dead
    pdeadFemale = pfemaleDead * pdead / pfemales + ADJUSTMENT

    ## P(Pclass=1|Alive) = Total(Pclass=1 & Alive)/Total(Alive) 
    ## P(Pclass=1|Alive) = P(Alive|Pclass=1)*P(Pclass=1) / P(Alive)
    ## P(Alive|Pclass=1) = P(Pclass=1|Alive) * P(Alive) / P(Pclass=1)
    pclass1Alive = len(train_df[(train_df['Pclass'] == 1) & (train_df['Survived'] == 1)]) / alives
    palivepclass1 = pclass1Alive * palives / ppclass1 + ADJUSTMENT

    ## P(Pclass=2|Alive) = Total(Pclass=2 & Alive)/Total(Alive) 
    ## P(Pclass=2|Alive) = P(Alive|Pclass=1)*P(Pclass=1) / P(Alive)
    ## P(Alive|Pclass=2) = P(Pclass=2|Alive) * P(Alive) / P(Pclass=2)
    pclass2Alive = len(train_df[(train_df['Pclass'] == 2) & (train_df['Survived'] == 1)]) / alives
    palivepclass2 = pclass2Alive * palives / ppclass2 + ADJUSTMENT

    ## P(Pclass=3|Alive) = Total(Pclass=3 & Alive)/Total(Alive) 
    ## P(Pclass=3|Alive) = P(Alive|Pclass=3)*P(Pclass=1) / P(Alive)
    ## P(Alive|Pclass=3) = P(Pclass=3|Alive) * P(Alive) / P(Pclass=3)
    pclass3Alive = len(train_df[(train_df['Pclass'] == 3) & (train_df['Survived'] == 1)]) / alives
    palivepclass3 = pclass3Alive * palives / ppclass3 + ADJUSTMENT

    ## P(Pclass=1|Dead) = Total(Pclass=1 & Dead)/Total(Dead) 
    ## P(Pclass=1|Dead) = P(Dead|Pclass=1)*P(Pclass=1) / P(Dead)
    ## P(Dead|Pclass=1) = P(Pclass=1|Dead) * P(Dead) / P(Pclass=1)
    pclass1Dead = len(train_df[(train_df['Pclass'] == 1) & (train_df['Survived'] == 0)]) / dead
    pdeadpclass1 = pclass1Dead * pdead / ppclass1 + ADJUSTMENT

    ## P(Pclass=2|Dead) = Total(Pclass=2 & Dead)/Total(Dead) 
    ## P(Pclass=2|Dead) = P(Dead|Pclass=1)*P(Pclass=2) / P(Dead)
    ## P(Dead|Pclass=2) = P(Pclass=2|Dead) * P(Dead) / P(Pclass=2)
    pclass2Dead = len(train_df[(train_df['Pclass'] == 2) & (train_df['Survived'] == 0)]) / dead
    pdeadpclass2 = pclass2Dead * pdead / ppclass2 + ADJUSTMENT

    ## P(Pclass=3|Dead) = Total(Pclass=3 & Dead)/Total(Dead) 
    ## P(Pclass=3|Dead) = P(Dead|Pclass=3)*P(Pclass=1) / P(Dead)
    ## P(Dead|Pclass=3) = P(Pclass=3|Dead) * P(Dead) / P(Pclass=3)
    pclass3Dead = len(train_df[(train_df['Pclass'] == 3) & (train_df['Survived'] == 0)]) / dead
    pdeadpclass3 = pclass3Dead * pdead / ppclass3 + ADJUSTMENT

    ## P(Embarked=S|Alive) = Total(Embarked=S & Alive) / Total(Alive)
    ## P(Embarked=S|Alive) = P(Alive|Embarked=S) * P(Embarked=S) / P(Alive)
    ## P(Alive|Embarked=S) = P(Embarked=S|Alive) * P(Alive) / P(Embarked=S)
    pembarkedSAlive = len(train_df[(train_df['Embarked'] == 'S') & (train_df['Survived'] == 1)]) / alives
    paliveEmbarkedS = pembarkedSAlive * palives / pembarkedS + ADJUSTMENT

    ## P(Embarked=C|Alive) = Total(Embarked=C & Alive) / Total(Alive)
    ## P(Embarked=C|Alive) = P(Alive|Embarked=C) * P(Embarked=C) / P(Alive)
    ## P(Alive|Embarked=C) = P(Embarked=C|Alive) * P(Alive) / P(Embarked=C)
    pembarkedCAlive = len(train_df[(train_df['Embarked'] == 'C') & (train_df['Survived'] == 1)]) / alives
    paliveEmbarkedC = pembarkedCAlive * palives / pembarkedC + ADJUSTMENT

    ## P(Embarked=Q|Alive) = Total(Embarked=Q & Alive) / Total(Alive)
    ## P(Embarked=Q|Alive) = P(Alive|Embarked=Q) * P(Embarked=Q) / P(Alive)
    ## P(Alive|Embarked=Q) = P(Embarked=Q|Alive) * P(Alive) / P(Embarked=Q)
    pembarkedQAlive = len(train_df[(train_df['Embarked'] == 'Q') & (train_df['Survived'] == 1)]) / alives
    paliveEmbarkedQ = pembarkedQAlive * palives / pembarkedQ + ADJUSTMENT

    ## P(Embarked=S|Dead) = Total(Embarked=S & Dead) / Total(Dead)
    ## P(Embarked=S|Dead) = P(Alive|Embarked=S) * P(Embarked=S) / P(Dead)
    ## P(Dead|Embarked=S) = P(Embarked=S|Dead) * P(Dead) / P(Embarked=S)
    pembarkedSDead = len(train_df[(train_df['Embarked'] == 'S') & (train_df['Survived'] == 0)]) / dead
    pdeadEmbarkedS = pembarkedSDead * pdead / pembarkedS + ADJUSTMENT

    ## P(Embarked=C|Dead) = Total(Embarked=C & Dead) / Total(Dead)
    ## P(Embarked=C|Dead) = P(Dead|Embarked=C) * P(Embarked=C) / P(Dead)
    ## P(Dead|Embarked=C) = P(Embarked=C|Dead) * P(Dead) / P(Embarked=C)
    pembarkedCDead = len(train_df[(train_df['Embarked'] == 'C') & (train_df['Survived'] == 0)]) / dead
    pdeadEmbarkedC = pembarkedCDead * pdead / pembarkedC + ADJUSTMENT

    ## P(Embarked=Q|Dead) = Total(Embarked=Q & Dead) / Total(Dead)
    ## P(Embarked=Q|Dead) = P(Dead|Embarked=Q) * P(Embarked=Q) / P(Dead)
    ## P(Dead|Embarked=Q) = P(Embarked=Q|Dead) * P(Dead) / P(Embarked=Q)
    pembarkedQDead = len(train_df[(train_df['Embarked'] == 'Q') & (train_df['Survived'] == 0)]) / dead
    pdeadEmbarkedQ = pembarkedQDead * pdead / pembarkedQ + ADJUSTMENT

    ## P(Cabin=A|Alive) = Total(Cabin=A & Alive) / Total(Alive)
    ## P(Cabin=A|Alive) = P(Alive|Cabin=A) * P(Cabin=A) / P(Alive)
    ## P(Alive|Cabin=A) = P(Cabin=A|Alive) * P(Alive) / P(Cabin=A)
    pcabinAAlive = len(train_df[(train_df['Cabin'] == 'A') & (train_df['Survived'] == 1)]) / alives
    paliveCabinA = pcabinAAlive * palives / pcabinA + ADJUSTMENT

    ## P(Cabin=D|Alive) = Total(Cabin=D & Alive) / Total(Alive)
    ## P(Cabin=D|Alive) = P(Alive|Cabin=D) * P(Cabin=D) / P(Alive)
    ## P(Alive|Cabin=D) = P(Cabin=D|Alive) * P(Alive) / P(Cabin=D)
    pcabinDAlive = len(train_df[(train_df['Cabin'] == 'D') & (train_df['Survived'] == 1)]) / alives
    paliveCabinD = pcabinDAlive * palives / pcabinD + ADJUSTMENT

    ## P(Cabin=G|Alive) = Total(Cabin=G & Alive) / Total(Alive)
    ## P(Cabin=G|Alive) = P(Alive|Cabin=G) * P(Cabin=G) / P(Alive)
    ## P(Alive|Cabin=G) = P(Cabin=G|Alive) * P(Alive) / P(Cabin=G)
    pcabinGAlive = len(train_df[(train_df['Cabin'] == 'G') & (train_df['Survived'] == 1)]) / alives
    paliveCabinG = pcabinGAlive * palives / pcabinG + ADJUSTMENT

    ## P(Cabin=A|Dead) = Total(Cabin=A & Dead) / Total(Dead)
    ## P(Cabin=A|Dead) = P(Alive|Cabin=A) * P(Cabin=A) / P(Dead)
    ## P(Dead|Cabin=A) = P(Cabin=A|Dead) * P(Dead) / P(Cabin=A)
    pcabinADead = len(train_df[(train_df['Cabin'] == 'A') & (train_df['Survived'] == 0)]) / dead
    pdeadCabinA = pcabinADead * pdead / pcabinA + ADJUSTMENT

    ## P(Cabin=D|Dead) = Total(Cabin=D & Dead) / Total(Dead)
    ## P(Cabin=D|Dead) = P(Dead|cabin=D) * P(Cabin=D) / P(Dead)
    ## P(Dead|Cabin=D) = P(Cabin=D|Alive) * P(Dead) / P(Cabin=D)
    pcabinDDead = len(train_df[(train_df['Cabin'] == 'D') & (train_df['Survived'] == 0)]) / dead
    pdeadCabinD = pcabinDDead * pdead / pcabinD + ADJUSTMENT

    ## P(Cabin=G|Dead) = Total(Cabin=G & Dead) / Total(Dead)
    ## P(Cabin=G|Dead) = P(Dead|Cabin=G) * P(Cabin=G) / P(Dead)
    ## P(Dead|Cabin=G) = P(Cabin=G|Dead) * P(Dead) / P(Cabin=G)
    pcabinGDead = len(train_df[(train_df['Cabin'] == 'G') & (train_df['Survived'] == 0)]) / dead
    pdeadCabinG = pcabinGDead * pdead / pcabinG + ADJUSTMENT

    ## P(Size=1|Alive) = Total(Size=1 & Alive) / Total(Alive)
    ## P(Size=1|Alive) = P(Alive|Size=1) * PSize=1) / P(Alive)
    ## P(Alive|Size = 1) = P(Size=1|Alive) * P(Alive) / P(Size=1)
    psize1Alive = len(train_df[(train_df['Size'] == 1) & (train_df['Survived'] == 1)]) / alives
    paliveSize1 = psize1Alive * palives / psize1 + ADJUSTMENT

    ## P(Size=2|Alive) = Total(Size=2 & Alive) / Total(Alive)
    ## P(Size=2|Alive) = P(Alive|Size=2) * PSize=2) / P(Alive)
    ## P(Alive|Size=2) = P(Size=2|Alive) * P(Alive) / P(Size=2)
    psize2Alive = len(train_df[(train_df['Size'] == 2) & (train_df['Survived'] == 1)]) / alives
    paliveSize2 = psize2Alive * palives / psize2 + ADJUSTMENT

    ## P(Size=3|Alive) = Total(Size=3 & Alive) / Total(Alive)
    ## P(Size=3|Alive) = P(Alive|Size=3) * PSize=3) / P(Alive)
    ## P(Alive|Size=3) = P(Size=3|Alive) * P(Alive) / P(Size=3)
    psize3Alive = len(train_df[(train_df['Size'] == 3) & (train_df['Survived'] == 1)]) / alives
    paliveSize3 = psize3Alive * palives / psize3 + ADJUSTMENT

    ## P(Size=5|Alive) = Total(Size=5 & Alive) / Total(Alive)
    ## P(Size=5|Alive) = P(Alive|Size=5) * PSize=5) / P(Alive)
    ## P(Alive|Size=5) = P(Size=5|Alive) * P(Alive) / P(Size=5)
    psize5Alive = len(train_df[(train_df['Size'] == 5) & (train_df['Survived'] == 1)]) / alives
    paliveSize5 = psize5Alive * palives / psize5 + ADJUSTMENT

    ## P(Size=1|Dead) = Total(Size=1 & Dead) / Total(Dead)
    ## P(Size=1|Dead) = P(Dead|Size=1) * PSize=1) / P(Dead)
    ## P(Alive|Size = 1) = P(Size=1|Dead) * P(Dead) / P(Size=1)
    psize1Dead = len(train_df[(train_df['Size'] == 1) & (train_df['Survived'] == 0)]) / dead
    pdeadSize1 = psize1Dead * pdead / psize1 + ADJUSTMENT

    ## P(Size=2|Dead) = Total(Size=2 & Dead) / Total(Dead)
    ## P(Size=2|Dead) = P(Dead|Size=2) * PSize=2) / P(Dead)
    ## P(Dead|Size=2) = P(Size=2|Dead) * P(Dead) / P(Size=2)
    psize2Dead = len(train_df[(train_df['Size'] == 2) & (train_df['Survived'] == 0)]) / dead
    pdeadSize2 = psize2Dead * pdead / psize2 + ADJUSTMENT

    ## P(Size=3|Dead) = Total(Size=3 & Dead) / Total(Dead)
    ## P(Size=3|Dead) = P(Dead|Size=3) * PSize=3) / P(Dead)
    ## P(Dead|Size=3) = P(Size=3|Dead) * P(Dead) / P(Size=3)
    psize3Dead = len(train_df[(train_df['Size'] == 3) & (train_df['Survived'] == 0)]) / dead
    pdeadSize3 = psize3Alive * pdead / psize3 + ADJUSTMENT

    ## P(Size=5|Dead) = Total(Size=5 & Dead) / Total(Dead)
    ## P(Size=5|Dead) = P(Dead|Size=5) * PSize=5) / P(AliveDead)
    ## P(Dead|Size=5) = P(Size=5|Dead) * P(Dead) / P(Size=5)
    psize5Dead = len(train_df[(train_df['Size'] == 5) & (train_df['Survived'] == 0)]) / dead
    pdeadSize5 = psize5Dead * pdead / psize5 + ADJUSTMENT

    ## P(Title=Mr|Alive) = Total(Title=Mr & Alive) / Total(Alive)
    ## P(Title=Mr|Alive) = P(Alive|Title=Mr) * P(Title=Mr) / P(Alive)
    ## P(Alive|Title=Mr) = P(Title=Mr|Alive) * P(Alive) / P(Title=Mr)
    ptitleMrAlive = len(train_df[(train_df['Title'] == 'Mr') & (train_df['Survived'] == 1)]) / alives
    paliveTitleMr = ptitleMrAlive * palives / ptitleMr + ADJUSTMENT

    ## P(Title=Mrs|Alive) = Total(Title=Mrs & Alive) / Total(Alive)
    ## P(Title=Mrs|Alive) = P(Alive|Title=Mrs) * P(Title=Mrs) / P(Alive)
    ## P(Alive|Title=Mrs) = P(Title=Mrs|Alive) * P(Alive) / P(Title=Mrs)
    ptitleMrsAlive = len(train_df[(train_df['Title'] == 'Mrs') & (train_df['Survived'] == 1)]) / alives
    paliveTitleMrs = ptitleMrsAlive * palives / ptitleMrs + ADJUSTMENT

    ## P(Title=Miss|Alive) = Total(Title=Miss & Alive) / Total(Alive)
    ## P(Title=Miss|Alive) = P(Alive|Title=Miss) * P(Title=Miss) / P(Alive)
    ## P(Alive|Title=Miss) = P(Title=Miss|Alive) * P(Alive) / P(Title=Miss)
    ptitleMissAlive = len(train_df[(train_df['Title'] == 'Miss') & (train_df['Survived'] == 1)]) / alives
    paliveTitleMiss = ptitleMissAlive * palives / ptitleMiss + ADJUSTMENT

    ## P(Title=GramPa|Alive) = Total(Title=GramPa & Alive) / Total(Alive)
    ## P(Title=GramPa|Alive) = P(Alive|Title=GramPa) * P(Title=GramPa) / P(Alive)
    ## P(Alive|Title=GramPa) = P(Title=GramPa|Alive) * P(Alive) / P(Title=Mrs)
    ptitleGramPaAlive = len(train_df[(train_df['Title'] == 'GramPa') & (train_df['Survived'] == 1)]) / alives
    paliveTitleGramPa = ptitleGramPaAlive * palives / ptitleGramPa + ADJUSTMENT

    ## P(Title=Master|Alive) = Total(Title=Master & Alive) / Total(Alive)
    ## P(Title=Master|Alive) = P(Alive|Title=Master) * P(Title=Master) / P(Alive)
    ## P(Alive|Title=Master) = P(Title=Master|Alive) * P(Alive) / P(Title=Master)
    ptitleMasterAlive = len(train_df[(train_df['Title'] == 'Master') & (train_df['Survived'] == 1)]) / alives
    paliveTitleMaster = ptitleMasterAlive * palives / ptitleMaster + ADJUSTMENT

    ## P(Title=Girl|Alive) = Total(Title=Girl & Alive) / Total(Alive)
    ## P(Title=Girl|Alive) = P(Alive|Title=Girl) * P(Title=Mrs) / P(Alive)
    ## P(Alive|Title=Girl) = P(Title=Girl|Alive) * P(Alive) / P(Title=Girl)
    ptitleGirlAlive = len(train_df[(train_df['Title'] == 'Girl') & (train_df['Survived'] == 1)]) / alives
    paliveTitleGirl = ptitleGirlAlive * palives / ptitleGirl + ADJUSTMENT

    ## P(Title=GramMa|Alive) = Total(Title=GramMa & Alive) / Total(Alive)
    ## P(Title=GramMa|Alive) = P(Alive|Title=GramMa) * P(Title=GramMa) / P(Alive)
    ## P(Alive|Title=GramMa) = P(Title=GramMa|Alive) * P(Alive) / P(Title=GramMa)
    ptitleGramMaAlive = len(train_df[(train_df['Title'] == 'GramMa') & (train_df['Survived'] == 1)]) / alives
    paliveTitleGramMa = ptitleGramMaAlive * palives / ptitleGramMa + ADJUSTMENT

    ## P(Title=Baron|Alive) = Total(Title=Baron & Alive) / Total(Alive)
    ## P(Title=Baron|Alive) = P(Alive|Title=Baron) * P(Title=Baron) / P(Alive)
    ## P(Alive|Title=Baron) = P(Title=Baron|Alive) * P(Alive) / P(Title=Baron)
    ptitleBaronAlive = len(train_df[(train_df['Title'] == 'Baron') & (train_df['Survived'] == 1)]) / alives
    paliveTitleBaron = ptitleBaronAlive * palives / ptitleBaron + ADJUSTMENT

    ## P(Title=Clergy|Alive) = Total(Title=Clergy & Alive) / Total(Alive)
    ## P(Title=Clergy|Alive) = P(Alive|Title=Clergy) * P(Title=Clergy) / P(Alive)
    ## P(Alive|Title=Clergy) = P(Title=Clergy|Alive) * P(Alive) / P(Title=Clergy)
    ptitleClergyAlive = len(train_df[(train_df['Title'] == 'Clergy') & (train_df['Survived'] == 1)]) / alives
    paliveTitleClergy = ptitleClergyAlive * palives / ptitleClergy + ADJUSTMENT

    ## P(Title=Boy|Alive) = Total(Title=Boy & Alive) / Total(Alive)
    ## P(Title=Boy|Alive) = P(Alive|Title=Boy) * P(Title=Boy) / P(Alive)
    ## P(Alive|Title=Boy) = P(Title=Boy|Alive) * P(Alive) / P(Title=Boy)
    ptitleBoyAlive = len(train_df[(train_df['Title'] == 'Boy') & (train_df['Survived'] == 1)]) / alives
    paliveTitleBoy = ptitleBoyAlive * palives / ptitleBoy + ADJUSTMENT

    ## P(Title=Doctor|Alive) = Total(Title=Doctor & Alive) / Total(Alive)
    ## P(Title=Doctor|Alive) = P(Alive|Title=Doctor) * P(Title=Doctor) / P(Alive)
    ## P(Alive|Title=Doctor) = P(Title=Doctor|Alive) * P(Alive) / P(Title=Doctor)
    ptitleDoctorAlive = len(train_df[(train_df['Title'] == 'Doctor') & (train_df['Survived'] == 1)]) / alives
    paliveTitleDoctor = ptitleDoctorAlive * palives / ptitleDoctor + ADJUSTMENT

    ## P(Title=Army|Alive) = Total(Title=Army & Alive) / Total(Alive)
    ## P(Title=Army|Alive) = P(Alive|Title=Army) * P(Title=Army) / P(Alive)
    ## P(Alive|Title=Army) = P(Title=Army|Alive) * P(Alive) / P(Title=Army)
    ptitleArmyAlive = len(train_df[(train_df['Title'] == 'Army') & (train_df['Survived'] == 1)]) / alives
    paliveTitleArmy = ptitleArmyAlive * palives / ptitleArmy + ADJUSTMENT

    ## P(Title=Baronness|Alive) = Total(Title=Baronness & Alive) / Total(Alive)
    ## P(Title=Baronness|Alive) = P(Alive|Title=Baronness) * P(Title=Baronness) / P(Alive)
    ## P(Alive|Title=Baronness) = P(Title=ArmyBaronness|Alive) * P(Alive) / P(Title=Baronness)
    ptitleBaronnessAlive = len(train_df[(train_df['Title'] == 'Baronness') & (train_df['Survived'] == 1)]) / alives
    paliveTitleBaronness = ptitleBaronnessAlive * palives / ptitleBaronness + ADJUSTMENT

    ## P(Title=Nurse|Alive) = Total(Title=Nurse & Alive) / Total(Alive)
    ## P(Title=Nurse|Alive) = P(Alive|Title=Nurse) * P(Title=Nurse) / P(Alive)
    ## P(Alive|Title=Nurse) = P(Title=Nurse|Alive) * P(Alive) / P(Title=Nurse)
    ptitleNurseAlive = len(train_df[(train_df['Title'] == 'Nurse') & (train_df['Survived'] == 1)]) / alives
    paliveTitleNurse = ptitleNurseAlive * palives / ptitleNurse + ADJUSTMENT

    ## P(Title=Mr|Dead) = Total(Title=Mr & Dead) / Total(Dead)
    ## P(Title=Mr|Dead) = P(Alive|Title=Mr) * P(Title=Mr) / P(Dead)
    ## P(Dead|Title=Mr) = P(Title=Mr|Dead) * P(Dead) / P(Title=Mr)
    ptitleMrDead = len(train_df[(train_df['Title'] == 'Mr') & (train_df['Survived'] == 0)]) / dead
    pdeadTitleMr = ptitleMrDead * pdead / ptitleMr + ADJUSTMENT

    ## P(Title=Mrs|Dead) = Total(Title=Mrs & Dead) / Total(Dead)
    ## P(Title=Mrs|Dead) = P(Dead|Title=Mrs) * P(Title=Mrs) / P(Dead)
    ## P(Dead|Title=Mrs) = P(Title=Mrs|Dead) * P(Dead) / P(Title=Mrs)
    ptitleMrsDead = len(train_df[(train_df['Title'] == 'Mrs') & (train_df['Survived'] == 0)]) / dead
    pdeadTitleMrs = ptitleMrsDead * pdead / ptitleMrs + ADJUSTMENT

    ## P(Title=Miss|Dead) = Total(Title=Miss & Dead) / Total(Dead)
    ## P(Title=Miss|Dead) = P(Dead|Title=Miss) * P(Title=Miss) / P(Dead)
    ## P(Dead|Title=Miss) = P(Title=Miss|Dead) * P(Dead) / P(Title=Miss)
    ptitleMissDead = len(train_df[(train_df['Title'] == 'Miss') & (train_df['Survived'] == 0)]) / dead
    pdeadTitleMiss = ptitleMissDead * pdead / ptitleMiss + ADJUSTMENT

    ## P(Title=GramPa|Dead) = Total(Title=GramPa & Dead) / Total(Dead)
    ## P(Title=GramPa|Dead) = P(Dead|Title=GramPa) * P(Title=GramPa) / P(Dead)
    ## P(Dead|Title=GramPa) = P(Title=GramPa|Dead) * P(Dead) / P(Title=Mrs)
    ptitleGramPaDead = len(train_df[(train_df['Title'] == 'GramPa') & (train_df['Survived'] == 0)]) / dead
    pdeadTitleGramPa = ptitleGramPaDead * pdead / ptitleGramPa + ADJUSTMENT

    ## P(Title=Master|Dead) = Total(Title=Master & Dead) / Total(Dead)
    ## P(Title=Master|Dead) = P(Dead|Title=Master) * P(Title=Master) / P(Dead)
    ## P(Dead|Title=Master) = P(Title=Master|Dead) * P(Dead) / P(Title=Master)
    ptitleMasterDead = len(train_df[(train_df['Title'] == 'Master') & (train_df['Survived'] == 0)]) / dead
    pdeadTitleMaster = ptitleMasterDead * pdead / ptitleMaster + ADJUSTMENT

    ## P(Title=Girl|Dead) = Total(Title=Girl & Dead) / Total(Dead)
    ## P(Title=Girl|Dead) = P(Dead|Title=Girl) * P(Title=Mrs) / P(Dead)
    ## P(Dead|Title=Girl) = P(Title=Girl|Dead) * P(Dead) / P(Title=Girl)
    ptitleGirlDead = len(train_df[(train_df['Title'] == 'Girl') & (train_df['Survived'] == 0)]) / dead
    pdeadTitleGirl = ptitleGirlDead * pdead / ptitleGirl + ADJUSTMENT

    ## P(Title=GramMa|Dead) = Total(Title=GramMa & Dead) / Total(Dead)
    ## P(Title=GramMa|Dead) = P(Dead|Title=GramMa) * P(Title=GramMa) / P(Dead)
    ## P(Dead|Title=GramMa) = P(Title=GramMa|Dead) * P(Dead) / P(Title=GramMa)
    ptitleGramMaDead = len(train_df[(train_df['Title'] == 'GramMa') & (train_df['Survived'] == 0)]) / dead
    pdeadTitleGramMa = ptitleGramMaDead * pdead / ptitleGramMa + ADJUSTMENT

    ## P(Title=Baron|Dead) = Total(Title=Baron & Dead) / Total(Dead)
    ## P(Title=Baron|Dead) = P(Dead|Title=Baron) * P(Title=Baron) / P(Dead)
    ## P(Dead|Title=Baron) = P(Title=Baron|Dead) * P(Dead) / P(Title=Baron)
    ptitleBaronDead = len(train_df[(train_df['Title'] == 'Baron') & (train_df['Survived'] == 0)]) / dead
    pdeadTitleBaron = ptitleBaronDead * pdead / ptitleBaron + ADJUSTMENT

    ## P(Title=Clergy|Dead) = Total(Title=Clergy & Dead) / Total(Dead)
    ## P(Title=Clergy|Dead) = P(Dead|Title=Clergy) * P(Title=Clergy) / P(Dead)
    ## P(Dead|Title=Clergy) = P(Title=Clergy|Dead) * P(Dead) / P(Title=Clergy)
    ptitleClergyDead = len(train_df[(train_df['Title'] == 'Clergy') & (train_df['Survived'] == 0)]) / dead
    pdeadTitleClergy = ptitleClergyDead * pdead / ptitleClergy + ADJUSTMENT

    ## P(Title=Boy|Dead) = Total(Title=Boy & Dead) / Total(Dead)
    ## P(Title=Boy|Dead) = P(Dead|Title=Boy) * P(Title=Boy) / P(Dead)
    ## P(Dead|Title=Boy) = P(Title=Boy|Dead) * P(Dead) / P(Title=Boy)
    ptitleBoyDead = len(train_df[(train_df['Title'] == 'Boy') & (train_df['Survived'] == 0)]) / dead
    pdeadTitleBoy = ptitleBoyDead * pdead / ptitleBoy + ADJUSTMENT

    ## P(Title=Doctor|Dead) = Total(Title=Doctor & Dead) / Total(Dead)
    ## P(Title=Doctor|Dead) = P(Dead|Title=Doctor) * P(Title=Doctor) / P(Dead)
    ## P(Dead|Title=Doctor) = P(Title=Doctor|Dead) * P(Dead) / P(Title=Doctor)
    ptitleDoctorDead = len(train_df[(train_df['Title'] == 'Doctor') & (train_df['Survived'] == 0)]) / dead
    pDeadTitleDoctor = ptitleDoctorDead * pdead / ptitleDoctor + ADJUSTMENT

    ## P(Title=Army|Dead) = Total(Title=Army & Dead) / Total(Dead)
    ## P(Title=Army|Dead) = P(Dead|Title=Army) * P(Title=Army) / P(Dead)
    ## P(Dead|Title=Army) = P(Title=Army|Dead) * P(Dead) / P(Title=Army)
    ptitleArmyDead = len(train_df[(train_df['Title'] == 'Army') & (train_df['Survived'] == 0)]) / dead
    pdeadTitleArmy = ptitleArmyDead * pdead / ptitleArmy + ADJUSTMENT

    ## P(Title=Baronness|Dead) = Total(Title=Baronness & Dead) / Total(Dead)
    ## P(Title=Baronness|Dead) = P(Dead|Title=Baronness) * P(Title=Baronness) / P(Dead)
    ## P(Dead|Title=Baronness) = P(Title=ArmyBaronness|Dead) * P(Dead) / P(Title=Baronness)
    ptitleBaronnessDead = len(train_df[(train_df['Title'] == 'Baronness') & (train_df['Survived'] == 0)]) / dead
    pdeadTitleBaronness = ptitleBaronnessDead * pdead / ptitleBaronness + ADJUSTMENT

    ## P(Title=Nurse|Dead) = Total(Title=Nurse & Dead) / Total(Dead)
    ## P(Title=Nurse|Dead) = P(Dead|Title=Nurse) * P(Title=Nurse) / P(Dead)
    ## P(Dead|Title=Nurse) = P(Title=Nurse|Dead) * P(Dead) / P(Title=Nurse)
    ptitleNurseDead = len(train_df[(train_df['Title'] == 'Nurse') & (train_df['Survived'] == 0)]) / dead
    pdeadTitleNurse = ptitleNurseDead * pdead / ptitleNurse + ADJUSTMENT

    ## P(Age=0|Alive) = Total(Age=0 & Alive) / Total(Alive)
    ## P(Age=0|Alive) = P(Alive|Age=0) * P(Age=0) / P(Alive)
    ## P(Alive|Age=0) = P(Age=0|Alive) * P(Alive) / P(Age=0)
    page0Alive = len(train_df[(train_df['Age'] == 0) & (train_df['Survived'] == 1)]) / alives
    paliveAge0 = page0Alive * palives / page0 + ADJUSTMENT

    ## P(Age=15|Alive) = Total(Age=15 & Alive) / Total(Alive)
    ## P(Age=15|Alive) = P(Alive|Age=15) * P(Age=15) / P(Alive)
    ## P(Alive|Age=15) = P(Age=15|Alive) * P(Alive) / P(Age=15)
    page15Alive = len(train_df[(train_df['Age'] == 15) & (train_df['Survived'] == 1)]) / alives
    paliveAge15 = page15Alive * palives / page15 + ADJUSTMENT

    ## P(Age=20|Alive) = Total(Age=20 & Alive) / Total(Alive)
    ## P(Age=20|Alive) = P(Alive|Age=20) * P(Age=20) / P(Alive)
    ## P(Alive|Age=20) = P(Age=20|Alive) * P(Alive) / P(Age=20)
    page20Alive = len(train_df[(train_df['Age'] == 20) & (train_df['Survived'] == 1)]) / alives
    paliveAge20 = page20Alive * palives / page20 + ADJUSTMENT

    ## P(Age=24|Alive) = Total(Age=24 & Alive) / Total(Alive)
    ## P(Age=24|Alive) = P(Alive|Age=24) * P(Age=24) / P(Alive)
    ## P(Alive|Age=24) = P(Age=24|Alive) * P(Alive) / P(Age=24)
    page24Alive = len(train_df[(train_df['Age'] == 24) & (train_df['Survived'] == 1)]) / alives
    paliveAge24 = page24Alive * palives / page24 + ADJUSTMENT

    ## P(Age=26|Alive) = Total(Age=26 & Alive) / Total(Alive)
    ## P(Age=26|Alive) = P(Alive|Age=26) * P(Age=26) / P(Alive)
    ## P(Alive|Age=26) = P(Age=26|Alive) * P(Alive) / P(Age=26)
    page26Alive = len(train_df[(train_df['Age'] == 26) & (train_df['Survived'] == 1)]) / alives
    paliveAge26 = page26Alive * palives / page26 + ADJUSTMENT

    ## P(Age=28|Alive) = Total(Age=28 & Alive) / Total(Alive)
    ## P(Age=28|Alive) = P(Alive|Age=28) * P(Age=28) / P(Alive)
    ## P(Alive|Age=28) = P(Age=28|Alive) * P(Alive) / P(Age=28)
    page28Alive = len(train_df[(train_df['Age'] == 28) & (train_df['Survived'] == 1)]) / alives
    paliveAge28 = page28Alive * palives / page28 + ADJUSTMENT

    ## P(Age=32|Alive) = Total(Age=32 & Alive) / Total(Alive)
    ## P(Age=32|Alive) = P(Alive|Age=32) * P(Age=32) / P(Alive)
    ## P(Alive|Age=32) = P(Age=32|Alive) * P(Alive) / P(Age=32)
    page32Alive = len(train_df[(train_df['Age'] == 32) & (train_df['Survived'] == 1)]) / alives
    paliveAge32 = page32Alive * palives / page32 + ADJUSTMENT

    ## P(Age=36|Alive) = Total(Age=36 & Alive) / Total(Alive)
    ## P(Age=36|Alive) = P(Alive|Age=36) * P(Age=36) / P(Alive)
    ## P(Alive|Age=36) = P(Age=36|Alive) * P(Alive) / P(Age=36)
    page36Alive = len(train_df[(train_df['Age'] == 36) & (train_df['Survived'] == 1)]) / alives
    paliveAge36 = page36Alive * palives / page36 + ADJUSTMENT

    ## P(Age=46|Alive) = Total(Age=46 & Alive) / Total(Alive)
    ## P(Age=46|Alive) = P(Alive|Age=46) * P(Age=46) / P(Alive)
    ## P(Alive|Age=46) = P(Age=46|Alive) * P(Alive) / P(Age=46)
    page46Alive = len(train_df[(train_df['Age'] == 46) & (train_df['Survived'] == 1)]) / alives
    paliveAge46 = page46Alive * palives / page46 + ADJUSTMENT

    ## P(Age=0|Dead) = Total(Age=0 & Dead) / Total(Dead)
    ## P(Age=0|Dead) = P(Dead|Age=0) * P(Age=0) / P(Dead)
    ## P(Dead|Age=0) = P(Age=0|Dead) * P(Dead) / P(Age=0)
    page0Dead = len(train_df[(train_df['Age'] == 0) & (train_df['Survived'] == 0)]) / dead
    pdeadAge0 = page0Dead * pdead / page0 + ADJUSTMENT

    ## P(Age=15|Dead) = Total(Age=15 & Dead) / Total(Dead)
    ## P(Age=15|Dead) = P(Dead|Age=15) * P(Age=15) / P(Dead)
    ## P(Dead|Age=15) = P(Age=15|Dead) * P(Dead) / P(Age=15)
    page15Dead = len(train_df[(train_df['Age'] == 15) & (train_df['Survived'] == 0)]) / dead
    pdeadAge15 = page15Dead * pdead / page15 + ADJUSTMENT

    ## P(Age=20|Dead) = Total(Age=20 & Dead) / Total(Dead)
    ## P(Age=20|Dead) = P(Dead|Age=20) * P(Age=20) / P(Dead)
    ## P(Dead|Age=20) = P(Age=20|Dead) * P(Dead) / P(Age=20)
    page20Dead = len(train_df[(train_df['Age'] == 20) & (train_df['Survived'] == 0)]) / dead
    pdeadAge20 = page20Dead * pdead / page20 + ADJUSTMENT

    ## P(Age=24|Dead) = Total(Age=24 & Dead) / Total(Dead)
    ## P(Age=24|Dead) = P(Dead|Age=24) * P(Age=24) / P(Dead)
    ## P(Dead|Age=24) = P(Age=24|Dead) * P(Dead) / P(Age=24)
    page24Dead = len(train_df[(train_df['Age'] == 24) & (train_df['Survived'] == 0)]) / dead
    pdeadAge24 = page24Dead * pdead / page24 + ADJUSTMENT

    ## P(Age=26|Dead) = Total(Age=26 & Dead) / Total(Dead)
    ## P(Age=26|Dead) = P(Dead|Age=26) * P(Age=26) / P(Dead)
    ## P(Dead|Age=26) = P(Age=26|Dead) * P(Dead) / P(Age=26)
    page26Dead = len(train_df[(train_df['Age'] == 26) & (train_df['Survived'] == 0)]) / dead
    pdeadAge26 = page26Dead * pdead / page26 + ADJUSTMENT

    ## P(Age=28|Dead) = Total(Age=28 & Dead) / Total(Dead)
    ## P(Age=28|Dead) = P(Dead|Age=28) * P(Age=28) / P(Dead)
    ## P(Dead|Age=28) = P(Age=28|Dead) * P(Dead) / P(Age=28)
    page28Dead = len(train_df[(train_df['Age'] == 28) & (train_df['Survived'] == 0)]) / dead
    pdeadAge28 = page28Dead * pdead / page28 + ADJUSTMENT

    ## P(Age=32|Dead) = Total(Age=32 & Dead) / Total(Dead)
    ## P(Age=32|Dead) = P(Dead|Age=32) * P(Age=32) / P(Dead)
    ## P(Dead|Age=32) = P(Age=32|Dead) * P(Dead) / P(Age=32)
    page32Dead = len(train_df[(train_df['Age'] == 32) & (train_df['Survived'] == 0)]) / dead
    pdeadAge32 = page32Dead * pdead / page32 + ADJUSTMENT

    ## P(Age=36|Dead) = Total(Age=36 & Dead) / Total(Dead)
    ## P(Age=36|Dead) = P(Dead|Age=36) * P(Age=36) / P(Dead)
    ## P(Dead|Age=36) = P(Age=36|Dead) * P(Dead) / P(Age=36)
    page36Dead = len(train_df[(train_df['Age'] == 36) & (train_df['Survived'] == 0)]) / dead
    pdeadAge36 = page36Dead * pdead / page36 + ADJUSTMENT

    ## P(Age=46|Dead) = Total(Age=46 & Dead) / Total(Dead)
    ## P(Age=46|Dead) = P(Dead|Age=46) * P(Age=46) / P(Dead)
    ## P(Dead|Age=46) = P(Age=46|Dead) * P(Dead) / P(Age=46)
    page46Dead = len(train_df[(train_df['Age'] == 46) & (train_df['Survived'] == 0)]) / dead
    pdeadAge46 = page46Dead * pdead / page46 + ADJUSTMENT

    ## P(Fare=0|Alive) = Total(Fare=0 & Alive) / Total(Alive)
    ## P(Fare=0|Alive) = P(Alive|Fare=0) * P(Fare=0) / P(Alive)
    ## P(Alive|Fare=0) = P(Fare=0|Alive) * P(Alive) / P(Fare=0)
    pfare0Alive = len(train_df[(train_df['Fare'] == 0) & (train_df['Survived'] == 1)]) / alives
    paliveFare0 = pfare0Alive * palives / pfare0 + ADJUSTMENT

    ## P(Fare=10|Alive) = Total(Fare=10 & Alive) / Total(Alive)
    ## P(Fare=10|Alive) = P(Alive|Fare=10) * P(Fare=10) / P(Alive)
    ## P(Alive|Fare=10) = P(Fare=10|Alive) * P(Alive) / P(Fare=10)
    pfare10Alive = len(train_df[(train_df['Fare'] == 10) & (train_df['Survived'] == 1)]) / alives
    paliveFare10 = pfare10Alive * palives / pfare10 + ADJUSTMENT

    ## P(Fare=20|Alive) = Total(Fare=20 & Alive) / Total(Alive)
    ## P(Fare=20|Alive) = P(Alive|Fare=20) * P(Fare=20) / P(Alive)
    ## P(Alive|Fare=20) = P(Fare=20|Alive) * P(Alive) / P(Fare=20)
    pfare20Alive = len(train_df[(train_df['Fare'] == 20) & (train_df['Survived'] == 1)]) / alives
    paliveFare20 = pfare20Alive * palives / pfare20 + ADJUSTMENT
    
    ## P(Fare=30|Alive) = Total(Fare=30 & Alive) / Total(Alive)
    ## P(Fare=30|Alive) = P(Alive|Fare=30) * P(Fare=30) / P(Alive)
    ## P(Alive|Fare=30) = P(Fare=30|Alive) * P(Alive) / P(Fare=30)
    pfare30Alive = len(train_df[(train_df['Fare'] == 30) & (train_df['Survived'] == 1)]) / alives
    paliveFare30 = pfare30Alive * palives / pfare30 + ADJUSTMENT

    ## P(Fare=40|Alive) = Total(Fare=40 & Alive) / Total(Alive)
    ## P(Fare=40|Alive) = P(Alive|Fare=40) * P(Fare=40) / P(Alive)
    ## P(Alive|Fare=40) = P(Fare=40|Alive) * P(Alive) / P(Fare=40)
    pfare40Alive = len(train_df[(train_df['Fare'] == 40) & (train_df['Survived'] == 1)]) / alives
    paliveFare40 = pfare40Alive * palives / pfare40 + ADJUSTMENT

    ## P(Fare=80|Alive) = Total(Fare=80 & Alive) / Total(Alive)
    ## P(Fare=80|Alive) = P(Alive|Fare=80) * P(Fare=80) / P(Alive)
    ## P(Alive|Fare=80) = P(Fare=80|Alive) * P(Alive) / P(Fare=80)
    pfare80Alive = len(train_df[(train_df['Fare'] == 80) & (train_df['Survived'] == 1)]) / alives
    paliveFare80 = pfare80Alive * palives / pfare80 + ADJUSTMENT

    ## P(Fare=0|Dead) = Total(Fare=0 & Dead) / Total(Dead)
    ## P(Fare=0|Dead) = P(Dead|Fare=0) * P(Fare=0) / P(Dead)
    ## P(Dead|Fare=0) = P(Fare=0|Dead) * P(Dead) / P(Fare=0)
    pfare0Dead = len(train_df[(train_df['Fare'] == 0) & (train_df['Survived'] == 0)]) / dead
    pdeadFare0 = pfare0Dead * pdead / pfare0 + ADJUSTMENT

    ## P(Fare=10|Dead) = Total(Fare=10 & Dead) / Total(Dead)
    ## P(Fare=10|Dead) = P(Dead|Fare=10) * P(Fare=10) / P(Dead)
    ## P(Dead|Fare=10) = P(Fare=10|Dead) * P(Dead) / P(Fare=10)
    pfare10Dead = len(train_df[(train_df['Fare'] == 10) & (train_df['Survived'] == 0)]) / dead
    pdeadFare10 = pfare10Dead * pdead / pfare10 + ADJUSTMENT

    ## P(Fare=20|Dead) = Total(Fare=20 & Dead) / Total(Dead)
    ## P(Fare=20|Dead) = P(Dead|Fare=20) * P(Fare=20) / P(Dead)
    ## P(Dead|Fare=20) = P(Fare=20|Dead) * P(Dead) / P(Fare=20)
    pfare20Dead= len(train_df[(train_df['Fare'] == 20) & (train_df['Survived'] == 0)]) / dead
    pdeadFare20 = pfare20Dead * pdead / pfare20 + ADJUSTMENT
    
    ## P(Fare=30|Dead) = Total(Fare=30 & Dead) / Total(Dead)
    ## P(Fare=30|Dead) = P(Dead|Fare=30) * P(Fare=30) / P(Dead)
    ## P(Dead|Fare=30) = P(Fare=30|Dead) * P(Dead) / P(Fare=30)
    pfare30Dead = len(train_df[(train_df['Fare'] == 30) & (train_df['Survived'] == 0)]) / dead
    pdeadFare30 = pfare30Dead * pdead / pfare30 + ADJUSTMENT

    ## P(Fare=40|Alive) = Total(Fare=40 & Alive) / Total(Alive)
    ## P(Fare=40|Alive) = P(Alive|Fare=40) * P(Fare=40) / P(Alive)
    ## P(Alive|Fare=40) = P(Fare=40|Alive) * P(Alive) / P(Fare=40)
    pfare40Dead = len(train_df[(train_df['Fare'] == 40) & (train_df['Survived'] == 0)]) / dead
    pdeadFare40 = pfare40Dead * pdead / pfare40 + ADJUSTMENT

    ## P(Fare=80|Dead) = Total(Fare=80 & Dead) / Total(Dead)
    ## P(Fare=80|Dead) = P(Dead|Fare=80) * P(Fare=80) / P(Dead)
    ## P(Dead|Fare=80) = P(Fare=80|Dead) * P(Dead) / P(Fare=80)
    pfare80Dead = len(train_df[(train_df['Fare'] == 80) & (train_df['Survived'] == 0)]) / dead
    pdeadFare80 = pfare80Dead * pdead / pfare80 + ADJUSTMENT

    bayes['A'] = palives
    bayes['D'] = pdead
    
    bayes['A|Sex=male'] = paliveMale
    bayes['A|Sex=female'] = paliveFemale
    bayes['D|Sex=male'] = pdeadMale
    bayes['D|Sex=female'] = pdeadFemale

    bayes['A|Pclass=1'] = palivepclass1
    bayes['A|Pclass=2'] = palivepclass2
    bayes['A|Pclass=3'] = palivepclass3
    bayes['D|Pclass=1'] = pdeadpclass1
    bayes['D|Pclass=2'] = pdeadpclass2
    bayes['D|Pclass=3'] = pdeadpclass3

    bayes['A|Embarked=S'] = paliveEmbarkedS
    bayes['A|Embarked=C'] = paliveEmbarkedC
    bayes['A|Embarked=Q'] = paliveEmbarkedQ
    bayes['D|Embarked=S'] = pdeadEmbarkedS
    bayes['D|Embarked=C'] = pdeadEmbarkedC
    bayes['D|Embarked=Q'] = pdeadEmbarkedQ

    bayes['A|Cabin=A'] = paliveCabinA
    bayes['A|Cabin=D'] = paliveCabinD
    bayes['A|Cabin=G'] = paliveCabinG
    bayes['D|Cabin=A'] = pdeadCabinA
    bayes['D|Cabin=D'] = pdeadCabinD
    bayes['D|Cabin=G'] = pdeadCabinG

    bayes['A|Size=1'] = paliveSize1
    bayes['A|Size=2'] = paliveSize2
    bayes['A|Size=3'] = paliveSize3
    bayes['A|Size=5'] = paliveSize5
    bayes['D|Size=1'] = pdeadSize1
    bayes['D|Size=2'] = pdeadSize2
    bayes['D|Size=3'] = pdeadSize3
    bayes['D|Size=5'] = pdeadSize5

    bayes['A|Title=Mr'] = paliveTitleMr
    bayes['A|Title=Mrs'] = paliveTitleMrs
    bayes['A|Title=Miss'] = paliveTitleMiss
    bayes['A|Title=GramPa'] = paliveTitleGramPa
    bayes['A|Title=Master'] = paliveTitleMaster
    bayes['A|Title=Girl'] = paliveTitleGirl
    bayes['A|Title=GramMa'] = paliveTitleGramMa
    bayes['A|Title=Baron'] = paliveTitleBaron
    bayes['A|Title=Clergy'] = paliveTitleClergy
    bayes['A|Title=Boy'] = paliveTitleBoy
    bayes['A|Title=Doctor'] = paliveTitleDoctor
    bayes['A|Title=Army'] = paliveTitleArmy
    bayes['A|Title=Baronness'] = paliveTitleBaronness
    bayes['A|Title=Nurse'] = paliveTitleNurse

    bayes['D|Title=Mr'] = pdeadTitleMr
    bayes['D|Title=Mrs'] = pdeadTitleMrs
    bayes['D|Title=Miss'] = pdeadTitleMiss
    bayes['D|Title=GramPa'] = pdeadTitleGramPa
    bayes['D|Title=Master'] = pdeadTitleMaster
    bayes['D|Title=Girl'] = pdeadTitleGirl
    bayes['D|Title=GramMa'] = pdeadTitleGramMa
    bayes['D|Title=Baron'] = pdeadTitleBaron
    bayes['D|Title=Clergy'] = pdeadTitleClergy
    bayes['D|Title=Boy'] = pdeadTitleBoy
    bayes['D|Title=Doctor'] = pDeadTitleDoctor
    bayes['D|Title=Army'] = pdeadTitleArmy
    bayes['D|Title=Baronness'] = pdeadTitleBaronness
    bayes['D|Title=Nurse'] = pdeadTitleNurse

    bayes['A|Age=0'] = paliveAge0
    bayes['A|Age=15'] = paliveAge15
    bayes['A|Age=20'] = paliveAge20
    bayes['A|Age=24'] = paliveAge24
    bayes['A|Age=26'] = paliveAge26
    bayes['A|Age=28'] = paliveAge28
    bayes['A|Age=32'] = paliveAge32
    bayes['A|Age=36'] = paliveAge36
    bayes['A|Age=46'] = paliveAge46
    bayes['D|Age=0'] = pdeadAge0
    bayes['D|Age=15'] = pdeadAge15
    bayes['D|Age=20'] = pdeadAge20
    bayes['D|Age=24'] = pdeadAge24
    bayes['D|Age=26'] = pdeadAge26
    bayes['D|Age=28'] = pdeadAge28
    bayes['D|Age=32'] = pdeadAge32
    bayes['D|Age=36'] = pdeadAge36
    bayes['D|Age=46'] = pdeadAge46

    bayes['A|Fare=0'] = paliveFare0
    bayes['A|Fare=10'] = paliveFare10
    bayes['A|Fare=20'] = paliveFare20
    bayes['A|Fare=30'] = paliveFare30
    bayes['A|Fare=40'] = paliveFare40
    bayes['A|Fare=80'] = paliveFare80
    bayes['D|Fare=0'] = pdeadFare0
    bayes['D|Fare=10'] = pdeadFare10
    bayes['D|Fare=20'] = pdeadFare20
    bayes['D|Fare=30'] = pdeadFare30
    bayes['D|Fare=40'] = pdeadFare40
    bayes['D|Fare=80'] = pdeadFare80
    
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
