'''
Created on Sep. 24, 2020

@author: zollen
'''
import os
import operator
import random
import math
from deap import base
from deap import creator
from deap import tools
from pathlib import Path
import numpy as np
import pandas as pd
import pprint
import seaborn as sb
import warnings
import titanic_kaggle.lib.titanic_lib as tb
from sklearn.preprocessing import Binarizer
from sklearn.metrics import accuracy_score

warnings.filterwarnings('ignore')

    
    

pd.set_option('max_columns', None)
pd.set_option('max_rows', None)
np.random.seed(0)
sb.set_style('whitegrid')
pp = pprint.PrettyPrinter(indent=3) 

label_column = [ 'Survived']
identity_columns = [ 'PassengerId', 'Ticket' ]
numeric_columns = [ 'Age', 'Fare' ]
categorical_columns = [ 'Sex', 'Title', 'Pclass', 'Embarked', 'Cabin' ]
all_features_columns = numeric_columns + categorical_columns 

    
def fillAge(src_df, dest_df):
    
    ages = src_df.groupby(['Title', 'Sex', 'SibSp', 'Parch'])['Age'].median()

    for index, value in ages.items():
            dest_df.loc[(dest_df['Age'].isna() == True) &
                 (dest_df['Title'] == index[0]) & 
                 (dest_df['Sex'] == index[1]) &
                 (dest_df['SibSp'] == index[2]) &
                 (dest_df['Parch'] == index[3]), 'Age'] = value
                 
    ages = src_df.groupby(['Title', 'Sex', 'SibSp'])['Age'].median()

    for index, value in ages.items():
        dest_df.loc[(dest_df['Age'].isna() == True) &
                 (dest_df['Title'] == index[0]) & 
                 (dest_df['Sex'] == index[1]) &
                 (dest_df['SibSp'] == index[2]), 'Age'] = value
               
    ages = src_df.groupby(['Title', 'Sex'])['Age'].median()

    for index, value in ages.items():
        dest_df.loc[(dest_df['Age'].isna() == True) &
                 (dest_df['Title'] == index[0]) & 
                 (dest_df['Sex'] == index[1]), 'Age'] = value

## pd.qcut() based boundaries yields better result
def binAge(df):
    df.loc[df['Age'] < 14, 'Age'] = 7
    df.loc[(df['Age'] < 19) & (df['Age'] >= 14), 'Age'] = 17
    df.loc[(df['Age'] < 22) & (df['Age'] >= 19), 'Age'] = 20
    df.loc[(df['Age'] < 25) & (df['Age'] >= 22), 'Age'] = 24
    df.loc[(df['Age'] < 28) & (df['Age'] >= 25), 'Age'] = 27
    df.loc[(df['Age'] < 31.8) & (df['Age'] >= 28), 'Age'] = 30
    df.loc[(df['Age'] < 36) & (df['Age'] >= 31.8), 'Age'] = 34
    df.loc[(df['Age'] < 41) & (df['Age'] >= 36), 'Age'] = 38
    df.loc[(df['Age'] < 50) & (df['Age'] >= 41), 'Age'] = 46
    df.loc[(df['Age'] < 80) & (df['Age'] >= 50), 'Age'] = 70
    df.loc[df['Age'] >= 80, 'Age'] = 80

## pd.qcut() based boundaries yields better result    
def binFare(df):
    df.loc[df['Fare'] < 7.229, 'Fare'] = 4
    df.loc[(df['Fare'] < 7.75) & (df['Fare'] >= 7.229), 'Fare'] = 6
    df.loc[(df['Fare'] < 7.896) & (df['Fare'] >= 7.75), 'Fare'] = 7
    df.loc[(df['Fare'] < 8.05) & (df['Fare'] >= 7.896), 'Fare'] = 8
    df.loc[(df['Fare'] < 10.5) & (df['Fare'] >= 8.05), 'Fare'] = 9
    df.loc[(df['Fare'] < 13) & (df['Fare'] >= 10.5), 'Fare'] = 11
    df.loc[(df['Fare'] < 15.85) & (df['Fare'] >= 13), 'Fare'] = 14
    df.loc[(df['Fare'] < 24) & (df['Fare'] >= 15.85), 'Fare'] = 20
    df.loc[(df['Fare'] < 26.55) & (df['Fare'] >= 24), 'Fare'] = 25
    df.loc[(df['Fare'] < 33.308) & (df['Fare'] >= 26.55), 'Fare'] = 30
    df.loc[(df['Fare'] < 55.9) & (df['Fare'] >= 33.308), 'Fare'] = 45
    df.loc[(df['Fare'] < 83.158) & (df['Fare'] >= 55.9), 'Fare'] = 75
    df.loc[(df['Fare'] < 1000) & (df['Fare'] >= 83.158), 'Fare'] = 100

def binTicket(df):
    df.loc[df['Ticket'] < 7.889] = 3
    df.loc[(df['Ticket'] < 9.257) & (df['Ticket'] >= 7.889), 'Ticket'] = 8
    df.loc[(df['Ticket'] < 9.775) & (df['Ticket'] >= 9.257), 'Ticket'] = 9
    df.loc[(df['Ticket'] < 10.263) & (df['Ticket'] >= 9.775), 'Ticket'] = 10
    df.loc[(df['Ticket'] < 11.627) & (df['Ticket'] >= 10.263), 'Ticket'] = 11
    df.loc[(df['Ticket'] < 12.379) & (df['Ticket'] >= 11.627), 'Ticket'] = 12
    df.loc[(df['Ticket'] < 12.746) & (df['Ticket'] >= 12.379), 'Ticket'] = 13
    df.loc[(df['Ticket'] < 12.763) & (df['Ticket'] >= 12.746), 'Ticket'] = 14
    df.loc[(df['Ticket'] < 12.822) & (df['Ticket'] >= 12.763), 'Ticket'] = 15
    df.loc[df['Ticket'] >= 12.822, 'Ticket'] = 16
        
def fillCabin(src_df, dest_df):
    
    df = dest_df.copy()
    
    binFare(df)
    binAge(df)
      
    cabins = src_df.groupby(['Title', 'Fare', 'Pclass', 'SibSp', 'Parch', 'Embarked', 'Cabin'])['Cabin'].count()
    
    for index, _ in cabins.items():
        df.loc[(df['Cabin'].isna() == True) &
                 (df['Title'] == index[0]) & 
                 (df['Fare'] == index[1]) & 
                 (df['Pclass'] == index[2]) &
                 (df['SibSp'] == index[3]) &
                 (df['Parch'] == index[4]) & 
                 (df['Embarked'] == index[5]), 'Cabin'] = cabins[index[0], index[1], index[2], index[3], index[4], index[5]].idxmax()
           
    cabins = src_df.groupby(['Title', 'Fare', 'Pclass', 'SibSp', 'Parch', 'Cabin'])['Cabin'].count()

    for index, _ in cabins.items():
        df.loc[(df['Cabin'].isna() == True) &
                 (df['Title'] == index[0]) & 
                 (df['Fare'] == index[1]) & 
                 (df['Pclass'] == index[2]) &
                 (df['SibSp'] == index[3]) &
                 (df['Parch'] == index[4]), 'Cabin'] = cabins[index[0], index[1], index[2], index[3], index[4]].idxmax()
                   
    cabins = src_df.groupby(['Title', 'Fare', 'Pclass', 'SibSp', 'Cabin'])['Cabin'].count()

    for index, _ in cabins.items():
        df.loc[(df['Cabin'].isna() == True) &
                 (df['Title'] == index[0]) & 
                 (df['Fare'] == index[1]) & 
                 (df['Pclass'] == index[2]) &
                 (df['SibSp'] == index[3]), 'Cabin'] = cabins[index[0], index[1], index[2], index[3]].idxmax()
                  
    cabins = src_df.groupby(['Title', 'Fare', 'Pclass', 'Cabin'])['Cabin'].count()

    for index, _ in cabins.items():
        df.loc[(df['Cabin'].isna() == True) &
                 (df['Title'] == index[0]) & 
                 (df['Fare'] == index[1]) & 
                 (df['Pclass'] == index[2]), 'Cabin'] = cabins[index[0], index[1], index[2]].idxmax()
                   
    cabins = src_df.groupby(['Title', 'Fare', 'Cabin'])['Cabin'].count()

    for index, _ in cabins.items():
        df.loc[(df['Cabin'].isna() == True) &
                 (df['Title'] == index[0]) & 
                 (df['Fare'] == index[1]), 'Cabin'] = cabins[index[0], index[1]].idxmax()
                   
    cabins = src_df.groupby(['Title', 'Cabin'])['Cabin'].count()

    for index, _ in cabins.items():
        df.loc[(df['Cabin'].isna() == True) &
                 (df['Title'] == index[0]), 'Cabin'] = cabins[index[0]].idxmax()
                 
    df.loc[(df['Cabin'].isna() == True), 'Cabin' ] = 'X'
             
    dest_df['Cabin'] = df['Cabin']
    
PROJECT_DIR=str(Path(__file__).parent.parent)  
train_df = pd.read_csv(os.path.join(PROJECT_DIR, 'data/train.csv'))
test_df = pd.read_csv(os.path.join(PROJECT_DIR, 'data/test.csv'))

train_df.loc[train_df['Cabin'] == 'T', 'Cabin'] = 'A'
train_df.loc[train_df['Embarked'].isna() == True, 'Embarked'] = 'S'
train_df.loc[train_df['Fare'].isna() == True, 'Fare'] = 7.25
test_df.loc[test_df['Embarked'].isna() == True, 'Embarked'] = 'S' 
test_df.loc[test_df['Fare'].isna() == True, 'Fare'] = 7.25


tb.reeigneeringTitle(train_df)
tb.reeigneeringTitle(test_df)

train_df['Room']  = train_df['Cabin'].apply(tb.captureRoom)
test_df['Room']  = test_df['Cabin'].apply(tb.captureRoom)

train_df['Cabin'] = train_df['Cabin'].apply(tb.captureCabin) 
test_df['Cabin'] = test_df['Cabin'].apply(tb.captureCabin) 

train_df['Title'] = train_df['Title'].map(tb.titles)
test_df['Title'] = test_df['Title'].map(tb.titles)

train_df['Sex'] = train_df['Sex'].map(tb.sexes)
test_df['Sex'] = test_df['Sex'].map(tb.sexes)

train_df['Embarked'] = train_df['Embarked'].map(tb.embarkeds)
test_df['Embarked'] = test_df['Embarked'].map(tb.embarkeds)

train_df['Ticket'] = train_df['Ticket'].apply(tb.captureTicketId)
test_df['Ticket'] = test_df['Ticket'].apply(tb.captureTicketId)

train_df['Ticket'] = np.log(train_df['Ticket'])
test_df['Ticket'] = np.log(test_df['Ticket'])


all_df = pd.concat( [ train_df, test_df ], ignore_index = True )

fillAge(all_df, train_df)
fillAge(all_df, test_df)
fillAge(all_df, all_df)

train_df['Age'] = train_df['Age'].astype('int32')
test_df['Age'] = test_df['Age'].astype('int32')

binFare(all_df)
binAge(all_df)

fillCabin(all_df, train_df)
fillCabin(all_df, test_df)
fillCabin(all_df, all_df)

train_df['Cabin'] = train_df['Cabin'].map(tb.cabins)
test_df['Cabin'] = test_df['Cabin'].map(tb.cabins)
all_df['Cabin'] = all_df['Cabin'].map(tb.cabins)

tb.reeigneeringFamilySize(train_df)
tb.reeigneeringFamilySize(test_df)
tb.reeigneeringFamilySize(all_df)

tb.typecast(train_df)
tb.typecast(test_df)


ttrain_df = train_df.copy()
ttest_df = test_df.copy()
binFare(ttrain_df)
binAge(ttrain_df)
binFare(ttest_df)
binAge(ttest_df)
binTicket(ttrain_df)
binTicket(ttest_df)

                 
tbl = {
    "Title": np.union1d(ttrain_df['Title'].unique(), ttest_df['Title'].unique()),
    "Age": np.union1d(ttrain_df['Age'].unique(), ttest_df['Age'].unique()),
    "Sex": ttrain_df['Sex'].unique(),
    "Pclass": ttrain_df['Pclass'].unique(),
    "Cabin": np.union1d(ttrain_df['Cabin'].unique(), ttest_df['Cabin'].unique()),
    "Size": np.union1d(ttrain_df['Size'].unique(), ttest_df['Size'].unique()),
    "Fare": np.union1d(ttrain_df['Fare'].unique(), ttest_df['Fare'].unique()),
    "Embarked": ttrain_df['Embarked'].unique(),
    "Ticket": np.union1d(ttrain_df['Ticket'].unique(), ttest_df['Ticket'].unique())
    }

pp.pprint(tbl)


tb.navieBayes(ttrain_df, tbl)

columns = [ 'Title', 'Age', 'Sex', 'Pclass', 'Cabin', 'Size', 'Fare', 'Embarked', "Ticket" ]
coeffs = { "Title": 1.0, "Age": 1.0, "Sex": 1.0, "Pclass": 1.0, 
          "Cabin": 1.0, "Size": 1.0, "Fare": 1.0, "Embarked": 1.0, 
          "Ticket": 1.0 }

tb.reeigneeringSurvProb(ttrain_df, coeffs, columns)
tb.reeigneeringSurvProb(ttest_df, coeffs, columns )

train_df['Chance'] = ttrain_df['Chance']
test_df['Chance'] = ttest_df['Chance']

train_df['Cabin'] = train_df['Cabin'] * 1000 + train_df['Room']
test_df['Cabin'] = test_df['Cabin'] * 1000 + test_df['Room']

ttrain_df['Bayes'] = Binarizer(threshold=0.5).fit_transform(
    np.expand_dims(train_df['Chance'].values, 1))
pp.pprint("Accuracy %0.4f" % accuracy_score(ttrain_df['Survived'], ttrain_df['Bayes']))

tttest_df = ttrain_df.copy()
def evaluate(individual):
    calculate(tttest_df, dict(zip(columns, individual)), columns)
    return accuracy_score(tttest_df['Survived'], tttest_df['Chance']), 

def calculate(dest_df, coeffs, columns):
    func = tb.survivability(True, coeffs, columns)
    dest_df['Chance'] = dest_df.apply(func, axis = 1)

def generate(size, pmin, pmax, smin, smax):
    part = creator.Particle(random.uniform(pmin, pmax) for _ in range(size)) 
    part.speed = [random.uniform(smin, smax) for _ in range(size)]
    part.smin = smin
    part.smax = smax
    return part
    
def update(part, best, phi1, phi2):
    u1 = (random.uniform(0, phi1) for _ in range(len(part)))
    u2 = (random.uniform(0, phi2) for _ in range(len(part)))
    v_u1 = map(operator.mul, u1, map(operator.sub, part.best, part))
    v_u2 = map(operator.mul, u2, map(operator.sub, best, part))
    part.speed = list(map(operator.add, part.speed, map(operator.add, v_u1, v_u2)))
    for i, speed in enumerate(part.speed):
        if abs(speed) < part.smin:
            part.speed[i] = math.copysign(part.smin, speed)
        elif abs(speed) > part.smax:
            part.speed[i] = math.copysign(part.smax, speed)
    part[:] = list(map(operator.add, part, part.speed))

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Particle", list, fitness=creator.FitnessMax, speed=list, 
    smin=None, smax=None, best=None)

toolbox = base.Toolbox()
toolbox.register("particle", generate, size=9, pmin=0.1, pmax=2, smin=-0.2, smax=0.2)
toolbox.register("population", tools.initRepeat, list, toolbox.particle)
toolbox.register("update", update, phi1=0.2, phi2=0.2)
toolbox.register("evaluate", evaluate)

def main():
    pop = toolbox.population(n=2000)

    GEN = 150
    best = None

    for g in range(GEN):
        for part in pop:
            part.fitness.values = toolbox.evaluate(part)
            if not part.best or part.best.fitness < part.fitness:
                part.best = creator.Particle(part)
                part.best.fitness.values = part.fitness.values
            if not best or best.fitness < part.fitness:
                best = creator.Particle(part)
                best.fitness.values = part.fitness.values
            
        for part in pop:
            toolbox.update(part, best)

    print("=== BEST ===")
    pp.pprint(dict(zip(columns, best)))
    print("BEST Accuracy: %0.4f" % evaluate(best))
    
    return pop, best

if __name__ == "__main__":
    main()