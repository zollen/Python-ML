'''
Created on Jun. 15, 2021

@author: zollen
@url: https://www.youtube.com/watch?v=mu-l-K8-8jA
@url: https://github.com/ritvikmath/YouTubeVideoCode/blob/main/Bayesian%20Time%20Series.ipynb
@desc: Bayesian Analysis for Time Series data
'''


import pymc3 as pm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from statsmodels.tsa.arima_model import ARIMA
import warnings

warnings.filterwarnings('ignore')
sb.set_style('whitegrid')


true_phi_1 = -0.2
true_phi_2 = 0.5
true_sigma = 0.1

if __name__ == "__main__":  
    
    xvals = [np.random.normal(0, true_sigma), np.random.normal(0, true_sigma)]
    
    for _ in range(50):
        xvals.append(true_phi_1*xvals[-1] + true_phi_2*xvals[-2] + np.random.normal(0, true_sigma))
    
    
    xvals = np.array(xvals[2:])
        
    
    model = ARIMA(xvals, order=(2,0,0)).fit(maxiter=200)
    print(model.summary())
    
    
    '''
    ==============================================================================
                     coef    std err          z      P>|z|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const          0.0027      0.018      0.152      0.880      -0.033       0.038
    ar.L1.y       -0.1538      0.125     -1.233      0.217      -0.398       0.091
    ar.L2.y        0.4722      0.127      3.723      0.000       0.224       0.721
    The esimated coef of:
        L1 is -0.15, kind of close to the true phi1 -0.2
        L2 is 0.4722, pretty close to the true phi2 0.5
    
    These are just point estimates, the confident interval gives us some ideas the range 
    of the values. We want to get the full distribution of values for these 3 parameters.
    This is why we move to bayesian analysis
    '''
    
    if False:
        # compare between true values and fitted values
        plt.figure(figsize=(10,4))
        plt.plot(xvals)
        plt.plot(model.fittedvalues)
        plt.legend(['True Values', 'Fitted Values'])
    
    if False:  
        # let's forecast next 5 data points
        plt.figure(figsize=(10,4))
        plt.plot(xvals)
        forecast = model.forecast(5)
        plt.plot(range(len(xvals), len(xvals) + 5), forecast[0], color='g')
        plt.fill_between(range(len(xvals), len(xvals) + 5), forecast[2][:,0], forecast[2][:,1], color='g', alpha=0.25)
    
    '''
    Baylesian Analysis with PyMC3
    =============================
    Since we have 3 parameters, we have 3 priors
    Because I don't know the answer beforehand, so I am going to keep it open and 
    initialize a large range: mean=0, std=20
    
    Priors:
    Ø1  => N(0, 20)   (normal dist(mean=0, std=20)
    Ø2  => N(0, 20    (normal dist(mean=0, std=20)
    σ   => Exp(1)     (stddev)
    
    Likelihood:
    x(t)|Ø1, Ø2, σ, x(t-1), x(t-2) => N(Ø1 * x(t-1) + Ø2 * x(t-2), σ)
    
    Posterior 
    Given the entire time series, what is the distriubtion of these parameters: Ø1, Ø2, σ?
    Ø1, Ø2, σ|x => ?
    '''
     
    with pm.Model() as bayes_model:
        #priors
        # let's define the search area of mu=0, sigma=20
        phi = pm.Normal("phi", mu=0, sigma=20, shape=2) # shape=2 two phi values to return to phi
        sigma = pm.Exponential("sigma", lam=1)
        
        #likelihood
        likelihood = pm.AR("x", phi, sigma, observed=xvals) # autoregressive 
        
        #posterior
        trace = pm.sample(1000, cores=2) # gives me 1000 samples and 2 independent runs
        
        pm.traceplot(trace)
        plt.tight_layout()
        
        '''
        Bayesian Estimated Model:
        Phi[0]:  -0.2582255925574683
        Phi[1]:  0.22959905724892118
        Sd Residuals:  0.1142909435615185
        It is pretty close to coeffs in the ARIMA summary
        '''
        
        phi1_vals = trace.get_values('phi')[:,0]
        phi2_vals = trace.get_values('phi')[:,1]
        sigma_vals = trace.get_values('sigma')
        
        print('Bayesian Estimated Model:')
        print('Phi#1: ', phi1_vals.mean())
        print('Phi#2: ', phi2_vals.mean())
        print('Sd Residuals: ', sigma_vals.mean())
        
        '''
        Forecast Next Value
        '''
        num_samples = 10000
        forecasted_vals = []
        num_periods = 5
        
        for _ in range(num_samples):
            curr_vals = list(xvals.copy())
            
            phi1_val = np.random.choice(phi1_vals)
            phi2_val = np.random.choice(phi2_vals)
            sigma_val = np.random.choice(sigma_vals)
            
            for _ in range(num_periods):
                curr_vals.append(curr_vals[-1]*phi1_val + curr_vals[-2]*phi2_val + np.random.normal(0, sigma_val))
            forecasted_vals.append(curr_vals[-num_periods:]) 
        forecasted_vals = np.array(forecasted_vals)
        
        forecast = model.forecast(5)
        
        for i in range(num_periods):
            plt.figure(figsize=(10,4))
            vals = forecasted_vals[:,i]
            mu, dev = round(vals.mean(), 3), round(vals.std(), 3)
            sb.distplot(vals)
            p1 = plt.axvline(forecast[0][i], color='k')
            p2 = plt.axvline(vals.mean(), color='b')
            plt.legend((p1,p2), ('MLE', 'Posterior Mean'), fontsize=20)
            plt.title('Forecasted t+%s\nPosterior Mean: %s\nMLE: %s\nSD Bayes: %s\nSD MLE: %s'%((i+1), mu, round(forecast[0][i],3), dev, round(forecast[1][i],3)), fontsize=20)
            

    
plt.show()