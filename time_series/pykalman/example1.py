'''
Created on Jun. 17, 2021

@author: zollen
@url: https://www.kalmanfilter.net/kalman1d.html


Five Kalman Filter quations
----------------------------

1. State Update/Filtering Equestion
X_estim(n,n) = X_estim(n,n-1) + K(Z_measure(n) - X_estim(n,n-1))

2. State Extraplation/Prediction/Transition
X_estim(n+1,n) = X_estim(n,n) + Δt * Derivative(X_estim(n,n))
Derivative(X_estim(n+1,n)) = Derivative(X_estim(n,n)) <- Assume constant dynamics model

3. Kalman Gain (K)/Weight Equation
                       Estim_Uncertainty(n,n-1)
K(n) = ---------------------------------------------------------
        Estim_Uncertainty(n,n-1) + Measurment_Uncertainity(n,n) 
        
4. Covariance Update/Corrector Equation
Estim_Uncertainty(n,n) = (1 - K) * Estim_Uncertainty(n,n-1)

5. Covariance Extrapolation/Predictor Covariance Equation
Estim_Uncertainty(n+1,n) = Estim_Uncertainty(n,n) <- Assume constant dynamic model



Overview Diagram
//Python-ML/time_series/pykalman/DetailedKalmanFilterAlgorithm.png


In the State Extrapolation Equation and the Covariance Extrapolation Equation depends on 
    the system dynamics.
The table above demonstrates the special form of the Kalman Filter equations tailored 
    for the specific case. The general form of the equation will be presented later in 
    a matrix notation. Right now, our goal is to understand the concept of the Kalman Filter.

'''

'''
Problem #1
ESTIMATING THE TEMPERATURE OF THE HEATING LIQUID 

1. We think that we have an accurate model, thus we set the process noise variance (q) to 
    0.0001.
2. The measurement error (standard deviation) is 0.01C.
3. The measurements are taken every 5 seconds.
4. The system dynamics is constant.
5. Pay attention, although the real system dynamics is not constant (since the liquid is 
    heating), we are going to treat the system as a system with constant dynamics (the 
    temperature doesn't change).
6. The true liquid temperature at the measurement points is: 50.479C, 51.025C, 51.5C, 
    52.003C, 52.494C, 53.002C, 53.499C, 54.006C, 54.498C, and 54.991C.
7. The set of measurements is: 50.45C, 50.967C, 51.6C, 52.106C, 52.492C, 52.819C, 
    53.433C, 54.007C, 54.523C, and 54.99C.
'''

'''
We don't know what is the temperature of the liquid in a tank is and our *guess* is 10C.
Our guess is very imprecise, we *guess* our initial estimate error ( σ ) to 100. 
The Estimate Uncertainty of the initialization is the error variance (σ^2):
'''

import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt

sb.set_style('whitegrid')

true_values = [50.479, 51.025, 51.5, 52.003, 52.494, 53.002, 53.499, 54.006, 54.498, 54.991 ]
measurements = [50.45, 50.967, 51.6, 52.106, 52.492, 52.819, 53.433, 54.007, 54.523, 54.99  ]

r = 0.01      # measurement error
q = 0.15      # process noise variance
    

#initialization: guess (will be removed later)
estims = [ 10.0 ]
estims_uncertainty = [ 100.0 * 100.0  + q ]

for rnd in range(0, 10):
    # state update
    K = estims_uncertainty[-1] / (estims_uncertainty[-1] + r)
    next_estims = estims[-1] + K * (measurements[rnd] - estims[-1])
    next_estims_uncertainty = (1 - K) * estims_uncertainty[-1]
    
    # predict
    estims.append(next_estims)  # constant dynamic model
    estims_uncertainty.append(next_estims_uncertainty + q)
    
# remove initial guesses
estims = estims[1:]
estims_uncertainty = estims_uncertainty[1:]
    

np.set_printoptions(formatter={"float_kind": lambda x: "%0.4f" % x})
print(np.array(estims))   
print(np.array(estims_uncertainty))



x = range(1, 11)
plt.plot(x, true_values, marker='o')
plt.plot(x, measurements, marker='x')
plt.plot(x, estims, marker='+', color='r')
plt.title('The Liquid Temperature')
plt.legend(['True Temperatures', 'Measurements', 'Estimates'])
plt.xlabel('Measurement Number')
plt.ylabel('Temperature (C)')
plt.ylim(50, 55)
plt.show()    
