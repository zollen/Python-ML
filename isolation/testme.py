'''
Created on Jul. 20, 2021

@author: zollen
@url: https://towardsdatascience.com/isolation-forest-the-anomaly-detection-algorithm-any-data-scientist-should-know-1a99622eec2d
@desc: “Isolation Forest” is a brilliant algorithm for anomaly detection born in 2009. 
        It has since become very popular: it is also implemented in Scikit-learn.
        
        What makes anomaly detection so hard is that it is an unsupervised problem. 
        In other words, we usually don’t have labels telling us which instances are 
        actually “anomalies”. Or rather, even if we had labels, it would be very hard 
        to frame anomaly detection as a supervised problem. In fact:
            * anomalies are rare;
            * anomalies are novel;
            * anomalies are different from each other.
            
        The core idea is that it should be very easy to “isolate” anomalies based on 
        the caracteristics that make them unique. if we fit a decision tree on all the 
        observations, outliers should be found closer to the root of the tree than 
        “normal” instances.
        
'''

import pandas as pd
from sklearn.ensemble import IsolationForest
import seaborn as sb
from matplotlib import pyplot as plt
import warnings

warnings.filterwarnings('ignore')
sb.set_style("whitegrid")

df = pd.DataFrame({
        'x': [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 2.0],
        'y': [2.1, 2.2, 3.0, 2.6, 2.2, 2.8, 3.7]
    }, index = ['A', 'B', 'C', 'D', 'E', 'F', 'G'])


'''
The plot shows that 'G': (2.0, 3.7) is the outlier
'''
sb.jointplot(x = 'x',y = 'y', data = df, color='red')


iforest = IsolationForest(n_estimators = 100).fit(df)
scores = iforest.score_samples(df)
'''
'G': (2.0, 3.7) has the lowest value
'''
print(pd.Series(data=scores, index=df.index).sort_values(ascending=True))


plt.show()