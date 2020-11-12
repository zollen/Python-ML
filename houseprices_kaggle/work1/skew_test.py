'''
Created on Nov. 11, 2020

@author: zollen
'''
import os
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import norm, skew, boxcox_normmax
from scipy.special import boxcox1p
import matplotlib.pyplot as plt
import seaborn as sb
import warnings

warnings.filterwarnings('ignore')

sb.set_style("whitegrid")

PROJECT_DIR=str(Path(__file__).parent.parent)  
train_df = pd.read_csv(os.path.join(PROJECT_DIR, 'data/train.csv'))
test_df = pd.read_csv(os.path.join(PROJECT_DIR, 'data/test.csv'))

all_df = pd.concat([train_df, test_df])

print("===================== SKEW AND LOG TEST ======================")
train_df['LogMiscVal'] = train_df['MiscVal'].apply(lambda x : np.log1p(x))
train_df['BoxMiscVal'] = boxcox1p(train_df['MiscVal'], boxcox_normmax(train_df['MiscVal'] + 1))
print("Skew(train_df['MiscVal']):                                   %0.4f" % skew(train_df['MiscVal'].values))
print("Skew(Log1p(train_df'MiscVal'])):                             %0.4f" % skew(train_df['LogMiscVal'].values))
print("Skew(BoxCox(train_df['MiscVal'], boxcox_normmax(train_df))): %0.4f" % skew(train_df['BoxMiscVal'].values))
print()

all_df['LogMiscVal'] = all_df['MiscVal'].apply(lambda x : np.log1p(x))
all_df['BoxMiscVal'] = boxcox1p(all_df['MiscVal'], boxcox_normmax(all_df['MiscVal'] + 1))
print("Skew(all_df['MiscVal']):                                     %0.4f" % skew(all_df['MiscVal'].values))
print("Skew(Log1p(all_df['MiscVal'])):                              %0.4f" % skew(all_df['LogMiscVal'].values))
print("Skew(BoxCox(all_df['MiscVal'])):                             %0.4f" % skew(all_df['BoxMiscVal'].values))

print()
train_df['AllMiscVal'] = boxcox1p(train_df['MiscVal'], boxcox_normmax(all_df['MiscVal'] + 1))
print("Skew(BoxCox(train_df['MiscVal'], boxcox_normmax(all_df))):   %0.4f" % skew(train_df['AllMiscVal'].values))


fig, (a1, a2, a3) = plt.subplots(1, 3)

fig.set_size_inches(12 , 4)

a1.set_title("MiscVal")
sb.distplot(train_df['MiscVal'], fit = norm, ax = a1)
a2.set_title("Log1p(MiscVal)")
sb.distplot(train_df['LogMiscVal'], fit = norm, ax = a2)
a3.set_title("BoxCox(MiscVal)")
sb.distplot(train_df['BoxMiscVal'], fit = norm, ax = a3)

plt.show()