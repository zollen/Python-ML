'''
Created on Jun. 15, 2021

@author: zollen
@title: Kalman Filter for Time Series Data
@url: https://medium.com/dataman-in-ai/kalman-filter-explained-4d65b47916bf

(Full Explaination)
@url: https://www.kalmanfilter.net/background.html
'''

import yfinance as yf
from pykalman import KalmanFilter
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')
sb.set_style('whitegrid')

'''
1. The location of X(t) depends on X(t-1)
2. The location of X(t) is not observable
3. The observable Y(t) comes from the unobservable X(t) with gaussian noises.
4. The Kalman Filter is able to recover the X(t), given a sequence of nosiy Y(t).

Kalman Filter:
    It has one term X(t-1) and the error term. The location of the flying 
    ball at time t is X(t), which depends on the prior location X(t-1), multiplying the
    transaction matrix A(t), plus a random Gaussian noise qt. The error term qt has its 
    covariance matrix Q(t). 
The location of the reflection is Y(t)
    It has the state X(t), multiplying the observation C(t), plus a white 
    noise rt. The observation matrix tell us the next observation(or called measurement)
    we should expect given the predicted state of X(t). The covariance mtarix of the 
    error rt is R(t)
----------------------------------------------------

A(t) - Transaction Matrix
N(0, Q(t)) - Random Gaussian Noise Q(t)
C(t) - Observation Matrix
N(0, R(t)) - Random Gaussian Noise R(t)

X(t) = A(t) * X(t-1) + N(0, Q(t))
Y(t) = C(t) * X(t) + N(0, R(t))

Properties of Kalman Filter
---------------------------
1. A Kalman filter is called an optimal estimator. Optimal in what sense? The Kalman 
    filter minimizes the mean square error of the estimated parameters. So it is the 
    best unbiased estimator.
2. It is recursive so that Xt+1 can be calculated only with Xt. and does not require 
    the presence of all past data points X0, X1, …, Xt. This is an important merit for 
    real-time processing.
3. The error terms in Equation (1) and (2) are both Gaussian distributions, so the 
error term in the predicted values also follow the Gaussian distribution.
4. There is no need to provide labeled target data to “train” a model.

How does Kalman Filter works?
-----------------------------
1. Get the initial state value X0. Since we do not know anything about it, we just 
    provide any value for X0. You will see how the Kalman Filter converges to the true 
    value. Get the initial values for the transition matrix A0, the observational matrix 
    C0, and the covariance matrix of the error Qt and the Rt.
2. Predict: The Kalman Filter estimate the current state Xt using the transition 
    matrix At for time step t.
3. Update: The Kalman Filter obtains the new observation Yt as a new input. Then it 
    estimates the covariance matrix Qt and Rt for time step t.
4. Repeat Step (2) and (3) for next time step. In each step, it only needs the 
    statistics of the previous time t-1 but not the entire history.

'''


data = yf.download("SPY", start="2017-01-01", end="2021-12-31")
print(data.head())

# Construct a Kalman filter
kf = KalmanFilter(transition_matrices = [1],    # (F) The value for At. It is a random walk so is set to 1.0
                  observation_matrices = [1],   # (H) The value for Ht.
                  initial_state_mean = 0,       # Any initial value. It will converge to the true state value.
                  initial_state_covariance = 1, # Sigma value for the Qt in Equation (1) the Gaussian distribution
                  observation_covariance=1,     # (R) Sigma value for the Rt in Equation (2) the Gaussian distribution
                  transition_covariance=.01)    # (Q) A small turbulence in the random walk parameter 1.0
# Get the Kalman smoothing
state_means, _ = kf.filter(data['Adj Close'].values)

'''
Do you notice we do not set up a training dataset to train the model? That’s correct. 
The Kalman Filter does not work that way. The purpose of training a model is to get 
the parameters At. The Kalman Filter gets a parameter value for each new time step t.
'''

# Call it KF_mean
data['KF_mean'] = np.array(state_means)
print(data.head())

if True:
    data[['Adj Close','KF_mean']].plot()
    plt.title('Kalman Filter estimates for SPY')
    plt.legend(['SPY','Kalman Estimate'])
    plt.xlabel('Day')
    plt.ylabel('Price')
    
    
    
    
    
    
    
plt.show()