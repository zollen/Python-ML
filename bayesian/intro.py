'''
Created on Jun. 21, 2021

@author: zollen
@url: https://www.youtube.com/watch?v=SP-sAAYvGT8
@title: Bayesian Analysis with PyMC3
'''

import pymc3 as pm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.linear_model import LinearRegression
import warnings

warnings.filterwarnings('ignore')
sb.set_style('whitegrid')


'''
m - slope
b - intercept
σ - standard deviation of the noise

y(true) = mx + b
y = y(true) + N(0, σ)
'''


true_slope = 5
true_intercept = 10
true_sigma = 1

num_points = 10

if __name__ == "__main__":

    x_vals = np.linspace(0, 1, num_points)
    true_y_vals = true_slope * x_vals + true_intercept
    y_vals = true_y_vals + np.random.normal(scale=true_sigma, size=num_points)
    
    true_params = {'slope': true_slope, 'intercept': true_intercept, 'sigma': true_sigma}
    
    if False:
        plt.figure(figsize=(7,7))
        p1 = plt.scatter(x_vals, y_vals)
        p2, = plt.plot(x_vals, true_y_vals, color='r')
        plt.xlabel('x', fontsize=20)
        plt.ylabel('y', fontsize=20)
        plt.legend((p1, p2), ('samples', 'true line'), fontsize=18)
    
    '''
    Let's fit with a linear model and see if it does better!
    '''
    clf = LinearRegression()
    clf.fit(x_vals.reshape(-1,1), y_vals)
    preds = clf.predict(x_vals.reshape(-1,1))
    resids = preds - y_vals
    
    print('True Model:')
    print('y_true = %s*x + %s'%(true_slope, true_intercept))
    print('True sigma: %s\n'%true_params['sigma'])
    
    print('LinearRegression Estimated Model:')
    print('y_hat = %s*x + %s'%(clf.coef_[0], clf.intercept_))
    print('Sd Residuals: %s'%(resids.std()))
    
    mle_estimates = {'slope': clf.coef_[0], 'intercept': clf.intercept_, 'sigma': resids.std()}
    
    '''
    Priors
    ======
    m => N(0, 20)
    b => N(0, 20)
    σ => Exp(1)
    
    Likelihood
    ==========
    y | m, b σ => N(mx + b, σ)
    
    Posterior
    =========
    m, b, σ | y => ?
    ∝ - proportional
    P(m, b, σ | y) ∝ P(y | m, b, σ) x P(m) x P(b) x P(σ)
    '''
    
    with pm.Model() as model:
        #priors
        sigma = pm.Exponential("sigma", lam=1.0)
        intercept = pm.Normal("intercept", mu=0, sigma=20)
        slope = pm.Normal("slope", mu=0, sigma=20)
    
        #Likelihood
        likelihood = pm.Normal("y", mu=slope*x_vals + intercept, sigma=sigma, observed=y_vals)
    
        #posterior - draw 1000 samples from the posterior, 4 times independently
        trace = pm.sample(1000, cores=4)
        
    print('Bayesian Estimated Model:')
    print('y_hat = %s*x + %s'%(trace.get_values('slope').mean(), 
                               trace.get_values('intercept').mean()))
    print('Sd Residuals: %s'%(trace.get_values('sigma').mean()))

    if False:
        pm.traceplot(trace)
        plt.tight_layout() 
    
    if True:    
        for var in ['slope', 'intercept', 'sigma']:
            plt.figure(figsize=(10, 10))
            vals = trace.get_values(var)
            mean, lower, upper = round(vals.mean(), 2), round(vals.mean()-vals.std(), 2), round(vals.mean()+vals.std(), 2)
            sb.distplot(vals)
            posterior_est = plt.axvline(mean, color='b')
            mle_est = plt.axvline(mle_estimates[var], color='b', linestyle='dotted')
            plt.axvline(lower, color='r', linestyle='--')
            plt.axvline(upper, color='r', linestyle='--')
            plt.title('%s [True = %s]\nPosterior Mean: %s\nBound: (%s, %s)'%(var,true_params[var],mean,lower,upper), fontsize=20)
            true_val = plt.axvline(true_params[var], color='k')
        
            plt.legend((true_val, mle_est, posterior_est), ('true', 'MLE', 'Posterior Mean'), fontsize=18)

plt.show()
