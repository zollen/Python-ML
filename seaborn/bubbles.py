'''
Created on Nov. 29, 2020

@author: zollen
'''

import os
from pathlib import Path
import pandas as pd
import seaborn as sb
from matplotlib import pyplot as plt
import warnings

warnings.filterwarnings('ignore')
pd.set_option('max_columns', None)
pd.set_option('max_rows', None)
pd.set_option('display.width', 1000)
sb.set_style('whitegrid')

PROJECT_DIR=str(Path(__file__).parent.parent)  
df = pd.read_csv(os.path.join(PROJECT_DIR, 'data/winequality-white.csv'), sep=';')

sb.relplot(x="alcohol", 
            y="sulphates",
            hue="quality", 
            size="quality",
            alpha=0.5, 
            data=df)

plt.show()