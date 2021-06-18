'''
Created on Jun. 18, 2021

@author: zollen
@title: Multidimensional Kalman Filter
@url: https://www.kalmanfilter.net/multiSummary.html
@url: https://www.kalmanfilter.net/multiExamples.html
'''

'''
KalMan Filter Multidemensional Matrix Equations
-------------------------------------------------
Python-ML/time_series/pykalman/KalmanFilterMultiDiagram.png

K - Kalman Gain
P - Estim_Uncertainty
H - observation matrix
F - transition matrix
u - input variable
G - control matrix
Q - process noise uncertainity covariance matrix
R - measure uncertainity covariance matrix
w - process noise vector
v - meaurement noise vector
x - true system state (usually not available)
c - estimated system state
v - random noice vector
z - measurement vector

1. State Update/Filtering Equestion
H - observation matrix
X_estim(n,n) = X_estim(n,n-1) + K(Z_measure(n) - H * X_estim(n,n-1))

2. Kalman Gain (K)/Weight Equation
           Estim_Uncertainty(n,n-1) * transpose(H)
K = ---------------------------------------------------
      H * Estim_Uncertainty(n,n-1) * transpose(H) + Rn
      
3. Estimate uncertainty Update
Estim_Uncertainty(n,n) = (1 - K * H) * Estim_Uncertainty(n,n-1) * transpose(1 - K * H) + K * R * transpose(K)

4. State Extraplation/Prediction/Transition
X_estim(n+1,n) = F * X_estim(n,n-1) + G * u(n)

5. Uncertainity Extraplation
Estim_Uncertainty(n+1,n) = F * Estim_Uncertainty(n,n) * transpose(F) + Q


Auxiliary Equations
-------------------
Measurement Equation
z(n) = H * x(n) + v(n)

E(x) = μx = expectation of the random variable x - means of the random variable

Measurement covariance Equation
R(n) = E(v(n) * transpose(v(n)))

Process covarinance Equation
Q(n) = E(w(n) * transpose(w(n)))

Estimate uncertainity covariance Equation
P(n) = E(e(n) * transpose(e(n))) = E((x(n) - c(n,n-1)) * transpose(x(n) - c(n-n-1)))

'''

import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt

sb.set_style('whitegrid')

'''
Problem #2 - Vechicle location estimation
In this example, we will estimate the altitude of the rocket. The rocket is equipped 
with an onboard altimeter that provides altitude measurements. In addition to 
altimeter, the rocket is equipped with accelerometer that measures the rocket 
acceleration.

The accelerometer serves as a control input to Kalman Filter.

We assume a constant acceleration dynamics.

Accelerometers doesn't sense gravity, thus we need to reduce gravitational acceleration 
constant g from each accelerometer measurement.

For instance, an accelerometer at resting on a table would measure 1g upwards, while 
accelerometers in free fall will measure zero.

The accelerometer measurement is at time step 1g is:
        x3 - actual acceleration of the object (the second derivative of the object position)
        g  - gravitational acceleration constant; g = -9.8 m/s^2
        e  - an accelerometer measurement error
        
        a(n) = x3 - g + e
        
Thus, the state extrapolation equation can be simplified to:
    u(n) - control variable
    w(n) - process noise
    F    - state transition matrix
    G    - control matrix
    
    X_estim(n+1,n) = F * X_estim(n,n) + G * u(n) + w(n)
    
In this example, we have a control variable u, which is based on the accelerometer 
measurement.

The system state x(n) is defined by:
    x1(n)  - rocket altitude at time n
    x2(n)  - rocket velocity at time n

    x(n) = x1(n)
           x2(n)

We can express the state extrapolation equation as following:

    X1_estim(n+1,n) = 1  Δt  *  X1_estim(n,n)  +  0.5 * Δt^2  *  (a(n) + g)
    X2_estim(n+1,n)   0   1     X2_estim(n,n)         Δt
    
In the above equation:

    F = 1  Δt
        0   1
        
    G = 0.5Δt^2
           Δt
           
    u(n) = ( a(n) + g )
'''
