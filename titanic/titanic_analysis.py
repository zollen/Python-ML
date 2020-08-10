'''
Created on Aug. 1, 2020

@author: zollen
'''

import os
from pathlib import Path
import seaborn as sb
import pandas as pd
from sklearn import preprocessing
import statsmodels.api as sm
from matplotlib import pyplot as plt

pd.set_option('max_columns', None)
pd.set_option('max_rows', None)
sb.set_style('whitegrid')

label_column = [ 'survived']
numeric_columns = [ 'age', 'fare' ]
categorical_columns = [ 'sex', 'class', 'deck', 'alone', 'n_siblings_spouses', 'parch', 'embark_town' ]
all_features_columns = numeric_columns + categorical_columns

PROJECT_DIR=str(Path(__file__).parent.parent)
df = pd.read_csv(os.path.join(PROJECT_DIR, 'data/train.csv'))


print(df.head())
print("=============== STATS ===================")
print(df.describe())
print("============== COLLERATION ==============")
print(df.corr(method='pearson'))
print("============== SKEW =====================")
print(df.skew())


ddf = df.copy()

for name in categorical_columns:
    encoder = preprocessing.LabelEncoder()
    encoder.fit(ddf[name].unique())
    ddf[name] = encoder.transform(ddf[name].values)
    
model=sm.Logit(ddf[label_column], ddf[all_features_columns])
result=model.fit()
print(result.summary2())


for name in categorical_columns:
    print("=============== Total: ", name, "================")
    print(df.groupby(name).size())

sb.pairplot(df, hue = 'survived', diag_kind = "kde", kind = "scatter", palette = "bright")

fig, (a1, a2, a3, a4, a5) = plt.subplots(1, 5)

fig.set_size_inches(14 , 10)

sb.swarmplot(x = "survived", y = "age", hue = "sex", data = df, ax = a1)
a1.set_title("survived - sex ")

sb.barplot(x = "sex", y = "survived", hue = "embark_town", data = df, ax = a2)
a2.set_title("sex - embark_town")

#sb.pointplot(x = "sex", y = "survived", hue = "class", data = df, ax = a3)
sb.pointplot(x = "deck", y = "survived", hue = "deck", data = df, ax = a3)
a3.set_title('deck - survived')

sb.barplot(x = "survived", y = "fare", hue = "class", data = df, ax = a4)
a4.set_title("survived - class")

#sb.factorplot("survived", col = "parch", col_wrap = 3, data = df, kind = "count")

#sb.factorplot("survived", col = "embark_town", col_wrap = 3, data = df, kind = "count")

sb.countplot(x = "survived", data = df, palette = "Reds", ax = a5);

a4.set_title("survived - class")
a5.set_title("total")

plt.show()


