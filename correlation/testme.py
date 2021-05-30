'''
Created on May 29, 2021

@author: zollen
@url: https://towardsdev.com/how-to-not-misunderstand-correlation-75ce9b0289e
'''

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sb
import warnings

warnings.filterwarnings('ignore')

sb.set_style('whitegrid')
sb.set_context('talk')

'''
It is very important to get the scatterplots right. Before you move on to regression 
and further steps, you have to do your best to understand the relationship between the 
variables of interest. Especially, finding out whether the relationship is linear or not 
by looking at scatterplots is crucial to understand the correlation coefficient.
'''


olymp = pd.read_csv('../data/athlete_events.csv', 
                    usecols=['Sex', 'Age', 'Height', 'Weight', 
                    'Year', 'Sport', 'NOC']).dropna()

print(olymp.head())
print(olymp.info())

fig, ax = plt.subplots(2, 1, figsize=(10,6))

# Jitter the height
height_jitter = olymp['Height'] + np.random.normal(0, 2, size=len(olymp))
# Jitter the weight
weight_jitter = olymp['Weight'] + np.random.normal(0, 1, size=len(olymp))

# Create the scatterplot
ax[0].plot(height_jitter, 
        weight_jitter, 
        marker='.', # plot as dots
        linestyle='',
        alpha=0.02,
        markersize=3) # remove line

# Labeling
ax[0].set(title='Height vs. Weight of Olympic Athletes',
      xlabel='Height (cm)', ylabel='Weight (kg)')

# Zoom in
ax[0].axis([150, 210, 30, 125])


sb.set(rc={'figure.figsize':(10, 6)})
sb.regplot(x="Height", y="Weight", data=olymp[['Height', 'Weight']], 
           marker='.', fit_reg = False, x_jitter = 0.2, y_jitter = 0.2, 
           scatter_kws = {'alpha' : 1/3}, ax=ax[1])




'''
Pearson’s Correlation Coefficient Calcuation

        Σ [ (x(i) - mean(x)) * (y(i) - mean(y)) ]
r = ---------------------------------------------------
     sqrt( Σ(x(i) - mean(x))^2 * Σ(y(i) - mean(y))^2 )
     
The formula is covariance of x and y over the product of their standard deviations.
Pearson’s coefficient only captures linear relationships
'''


df_corr = olymp[['Weight', 'Height', 'Age']].corr()

print(df_corr)

'''
The result is a correlation matrix that shows the correlation coefficients of individual 
pairs of three variables. By interpreting the results, we can see that height and weight 
are highly correlated with a coefficient of 0.8. However, the relationships between age 
and weight as well as age and height are weak (0.21, 0.14 respectively).

However, you should never, ever conclude about the relationship between variables by 
just looking at the correlation coefficient. Make sure that your assumptions are correct 
by looking at them visually with the help of scatterplots or in some cases, 
using boxplots or violin plots.

Coefficient close to 0 does not mean ‘no relationship’
After you compute the correlation matrix, make sure to visually check your assumptions 
about the coefficients before you move on. One of the common pitfalls of correlation is 
that if you compare two variables that do not have a linear relationship, you will get 
a coefficient very close to 0.
'''
plt.show()