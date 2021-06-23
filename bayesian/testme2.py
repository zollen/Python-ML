'''
Created on Jun. 22, 2021

@author: zollen
@url: https://colab.research.google.com/drive/1pM8DqiMO1QjvZ0Y_LzJbJ_jIryeU5O7o#scrollTo=FF7UohbyyKeM
'''
import pymc3 as pm
import arviz as az
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import warnings

warnings.filterwarnings('ignore')
sb.set_style('whitegrid')

'''
p(μ|x) = ( p(x|μ) * p(μ) ) / p(x)
p(μ|x)∝ p(x|μ) * p(μ)

p(μ|x1,...,xN) ∝ p(x1|μ) * p(x2|μ) * ... * p(xN|μ) * p(μ2)

N      - number of data
∑ x(i) - sum of all current data
20     - standard deviation

                                ∑ x(i)                   1
p(μ|x1,...,xN) ∝ Normal (------------------ , ------------------ )
                           N + (1 / (20)^2)     N + (1 / (20)^2)
'''                     

if __name__ == "__main__":
    
    '''
    μ => N(0, 20)
    y => N(μ, 1)
    '''
    μ = 4
    σ_μ = 20                     
    y = μ + np.random.randn(100)
    
    # simple model for μ and σ
    with pm.Model() as model:
        
        # Prior - search space of mu, σ
        dist_μ = pm.Normal('mu', mu=0, sd=σ_μ)
        dist_σ = pm.Exponential('sigma', lam=1/5)
        
        likelihood = pm.Normal('y', mu=dist_μ, sd=dist_σ, observed=y)
        
        # sampling 1000 random guessed_μ and σ, then later calulate the mean
        trace = pm.sample(1000)
        
        print('[mu]: ', trace.get_values('mu').mean())
        print('[signa]: ', trace.get_values('sigma').mean())
        
        pm.traceplot(trace)
        plt.tight_layout() 

plt.show()
