'''
Created on Aug. 22, 2020

@author: zollen
'''
import os
from pathlib import Path
import pandas as pd
import numpy as np
import seaborn as sb
from sklearn import preprocessing
from matplotlib import pyplot as plt
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import warnings


warnings.filterwarnings('ignore')

pd.set_option('max_columns', None)
pd.set_option('max_rows', None)
pd.set_option('max_colwidth', 15)
pd.set_option('precision', 2)
np.random.seed(87)
sb.set_style('whitegrid')

label_column = [ 'survived' ]
numeric_columns = [ 'age', 'fare' ]
categorical_columns = [ 'sex', 'n_siblings_spouses', 'parch', 'class', 'deck', 'embark_town', 'alone' ]
all_features_columns = numeric_columns + categorical_columns

PROJECT_DIR=str(Path(__file__).parent.parent)  
train_df = pd.read_csv(os.path.join(PROJECT_DIR, 'data/train.csv'))
test_df = pd.read_csv(os.path.join(PROJECT_DIR , 'data/eval.csv'))

print(train_df.info())
print(train_df.isnull().sum())
print(train_df.describe())

if False:
    sb.heatmap(train_df.corr(), annot=True, linewidth=0.5, cmap="Oranges")

for name in  categorical_columns:
    print("Total(valid [%s] records): %d" % (name, len(train_df[train_df[name] != 'unknown'])))
    print("Total('unknown' [%s] records): %d" % (name, len(train_df[train_df[name] == 'unknown'])))
    
"""
Since there is only one record with 'unknown' embark_town. We will deal with that record first.
"""
print(train_df[train_df['embark_town'] == 'unknown'])    


"""
Let's compare the column 'embark_town' with 'fare', 'class', 'deck'
The following visual analysis indicates that there is a very good chance this alone 
38 years old lady who paid 80 dollars of fare, First class, Deck 8 with no companion
probably came from either Southampton or Cherbourg
"""
if False:
    fig, (a1, a2, a3) = plt.subplots(1, 3)

    fig.set_size_inches(14 , 10)

    sb.swarmplot(x = "survived", y = "fare", hue = "embark_town", alpha = 0.9, data = train_df, ax = a1)
    a1.set_title('fare - embark_town')
    sb.countplot(x = "class", hue = "embark_town", data = train_df, ax = a2);
    a2.set_title('class - embark_town')
    sb.countplot(x = "deck", hue = "embark_town", data = train_df, ax = a3);
    a3.set_title('deck - embark_town')

    plt.show()

## *Never* use chained indexes (i.e. train_df[a][b][c][d] = 1..etc), each level of 
## indexing might return a copy instead of an actual view.    

print("Let's put embark_town 'Cherbourg' to this rich lady")
train_df.loc[train_df['embark_town'] == 'unknown', 'embark_town'] = 'Cherbourg'
print(train_df.loc[48])


"""
Let's compare the column 'deck' with other features 'fare', 'class'
"""

if False:
    fig, (a1, a2, a3) = plt.subplots(1, 3)

    fig.set_size_inches(14 , 10)

    g1 = sb.swarmplot(x = "class", y = "fare", hue = "deck", alpha = 0.9, data = train_df, ax = a1)
    g1.set(yticks=range(0, 600, 25))
    a1.set_title('fare - deck')
    g2 = sb.countplot(x = "class", hue = "deck", data = train_df, ax = a2);
    g2.set(yticks=range(0, 350, 25))
    a2.set_title('class - deck')
    g3 = sb.countplot(x = "embark_town", hue = "deck", data = train_df, ax = a3);
    g3.set(yticks=range(0, 500, 25))
    a3.set_title('embark_town - deck')

    plt.show()


"""
Let's seperate the valid samples from the unknown deck samples
"""

label_pred_column = 'deck' 
numeric_pred_columns = [ 'age', 'fare', 'survived' ]
categorical_pred_columns = [ 'sex', 'n_siblings_spouses', 'parch', 'class', 'embark_town', 'alone' ]
all_pred_columns = numeric_pred_columns + categorical_pred_columns

model = ExtraTreesClassifier()

def process(model, fit, df, fileName):
    good_df = df[df[label_pred_column] != 'unknown']
    bad_df = df[df[label_pred_column] == 'unknown']

    ggood_df = good_df.copy()
    bbad_df = bad_df.copy()

    for name in categorical_pred_columns:
        encoder = preprocessing.LabelEncoder()   
        keys = df[name].unique()
   
        if len(keys) == 2:
            encoder = preprocessing.LabelBinarizer()

        encoder.fit(keys)

        ggood_df[name] = np.squeeze(encoder.transform(ggood_df[name].values))    
        bbad_df[name] = np.squeeze(encoder.transform(bbad_df[name].values))
    
    """
    Let's put the ggood_df into a ExteremeTree classifier
    """
    if fit == True:
        model.fit(ggood_df[all_pred_columns], ggood_df[label_pred_column])  
        print("================= GOOD DATA =====================")
        preds = model.predict(ggood_df[all_pred_columns])
        print("Accuracy: %0.2f" % accuracy_score(ggood_df[label_pred_column], preds))
        print(confusion_matrix(ggood_df[label_pred_column], preds))

    preds2 = model.predict(bbad_df[all_pred_columns])
    bad_df.loc[:, label_pred_column] = preds2

    testme_df = pd.concat([ good_df, bad_df ])
    testme_df.to_csv(os.path.join(PROJECT_DIR, 'data/' + fileName))


model = ExtraTreesClassifier()
process(model, True, train_df, "train_processed.csv")
process(model, False, test_df, "eval_processed.csv")