'''
Created on Jun. 22, 2021

@author: zollen
'''
import pymc3 as pm
import matplotlib.pyplot as plt
import seaborn as sb
import warnings

warnings.filterwarnings('ignore')
sb.set_style('whitegrid')

n = 100
h = 61
alpha = 2
beta = 2

if __name__ == "__main__":
    with pm.Model() as model:

        p = pm.Beta("beta", alpha=alpha, beta=beta)
        
        likelihood = pm.Binomial("y", n = n, p = p, observed = h)

        trace = pm.sample(1000, cores=1)
        
        print('Beta: ', trace.get_values('beta').mean())
        
        pm.traceplot(trace)
        
plt.show()