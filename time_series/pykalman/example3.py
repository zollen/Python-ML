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
Problem #2 - ROCKET ALTITUDE ESTIMATION
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
    
    X_estim(n+1,n)  =   F    *   X_estim(n,n)  +      G       *     u(n)
    ----------------------------------------------------------------------- 
    X1_estim(n+1,n) = 1  Δt  *  X1_estim(n,n)  +  0.5 * Δt^2  *  (a(n) + g)
    X2_estim(n+1,n)   0   1     X2_estim(n,n)         Δt
    
In the above equation:

    F = 1  Δt
        0   1
        
    G = 0.5Δt^2
           Δt
           
    u(n) = ( a(n) + g )
    

The covariance extraploation equation:
    F - state transition matrix
    Q - process noise matrix
    P(n+1:n) = F * P(n:n) * transpose(T) + Q

The estimate uncertinity in a matrix form is:

    P = p11  p12
        p21  p22 
        
The main diagonal of the matrix are the variances of the estimation:
    p11 is the variance of the altitude estimation
    p22 is the variance of the velocity estimation
    p12,p21 are the off-diagonal entries are covariances
    

The process noise matrix

We will assume the discrete noise model - the noise is different at each time period, 
but it is constant between time periods.

The process noise matrix for the constant acceleration model looks as following:

    Q = q11  q12
        q21  q22
        
We've already derived the Q matrix for the constant acceleration motion model. 
https://www.kalmanfilter.net/covextrap.html#withQ
https://www.kalmanfilter.net/background2.html#exp

The Q matrix for our example is: 

    Δt - time between successive measurements
    e^2 - random variance in accelerometer measurement

    Q = 0.25 * Δt^4  0.5 * Δt^3    *    e^2
         0.5 * Δt^3        Δt^2
         
In our pervious example, we used system's random variance in acceleration σ^2 as a 
multiplier of the process noise matrix. But here, we have accelerometer that measures 
the system random acceleration. Accelerometer error v is much lower than system's 
random acceleration, therefore we use ϵ2 as a multiplier of the process noise matrix.

It makes our estimation uncertainty much lower!

Now we can write down the covariance extrapolation equation for our example:    

    P(n+1:n)                  =   F    *         P(n:n)          * transpose(F) +                     Q
    -------------------------------------------------------------------------------------------------------------------
    p(1:1,n+1:n) p(1:2,n+1:n) = 1  Δt  *  p(1:1,n:n) p(1:2,n:n)  *  1  0        +  0.25 * Δt^4  0.5 * Δt^3    *    e^2
    p(2:1,n+1:n) p(2:2,n+1:n)   0   1     p(2:1,n:n) p(2:2,n:n)    Δt  1            0.5 * Δt^3        Δt^2



The measurement equation

    z(n) - measurement vector
    x(n) - true system state(hidden state)
    v(n) - random noise vector
    H    - observation matrix
    
    z(n) = H * x(n) + v(n)
    
    The measurement provides us only altitude of the rocket:
    
    z(n) = H * x(n)
    
    [x(n), mesaured] = H * x1(n)
                           x2(n)
                           
                           
The measurement uncertainty

    The measurement covariance matrix is:
    R(n) = [ σ(m)^2 ]
    
    The subscript m is for meausrement uncertainty
    For the sake of the example simplicity, we will assume a constant measurement
    uncertainty:
    R(1) = R(2)...R(n-1) = R(n) = R
    

The Kulman Gain
    K(n) - Kalman Gain
    H    - observation matrix
    R(n) - measurement uncertainty
    
                    P(n:n-1) * transpose(H) 
    K(n) = ------------------------------------------
             H * P(n:n-1) * transpose(H) + R(n) 
             
             
    K(1:1,n) = p(x1:x1,n:n-1) p(x1:x2,n:n-1)  *  1  * | 1  0  *  p(x1:x1,n:n-1) p(x1:x2,n:n-1) *  1  +  σ(m)^2 |^(-1)
    K(2:1,n)   p(x2:x1,n:n-1) p(x2:x2,n:n-1)     0    |          p(x2:x1,n:n-1) p(x2:x2,n:n-1)    0            |


The state update equation
    K(n)  - Kalman Gain
    z(n)  - measurement
    H     - observation matrix
    
    X_estim(n:n) = X_estim(n:n-1) + K(n) * ( z(n) - H * X_estim(n:n-1) )
    
    
The covariance update equation
    P(n:n) = (I - K(n) * H) * P(n:n-1) * transpose(I - K(n) * H) + K(n) * R(n) * transpose(K(n))



=====================================================================================
Let us assume a rocket boosting vertically with constant acceleration. The rocker 
equipped with altimeter that provides altitude measurements and accelerometer that 
serves as a control input.

The measurements period: Δt = 0.25s
The rocket acceleration: x2 = 30ms^2
The altimeter measurement error standard deviation: σ(m) = 20m
The accelerometer measurement error standard deviation: e = 0.1ms^2

The state transition matrix F would be:

        F = 1 Δt = 1 0.25
            0  1   0    1
            
The control matrix G would be:

        G = 0.5 * Δt^2  =  0.0313
                  Δt         0.25
                  
The proces noiose matrix Q would be:

        Q = 0.25 * Δt^4   0.5 * Δt^3  * σ(a)^2 = 0.25 * (0.25)^4  0.5 * (0.25)^3  *  (0.1)^2
             0.5 * Δt^2         Δt^2              0.5 * (0.25)^3          0.25^2    
             
The measurement uncertainty R would be:

        R(n) = R = [ σ(m)^2 ] = 400

'''
np.set_printoptions(formatter={"float_kind": lambda x: "%0.4f" % x})

z_alt = np.array([ -32.4, -11.1, 18, 22.9, 19.5, 28.5, 46.5, 68.9, 48.2, 56.1, 90.5, 104.9,
          140.9, 148, 187.6, 209.2, 244.6, 276.4, 323.5, 357.3, 357.4, 398.3, 446.7,
          465.1, 529.4, 570.4, 636.8, 693.3, 707.3, 748.5 ])
z_accl = np.array([ 39.72, 40.02, 39.97, 39.81, 39.75, 39.6, 39.77, 39.83, 39.73, 39.87, 
          39.81, 39.92, 39.78, 39.98, 39.76, 39.86, 39.61, 39.86, 39.74, 39.87, 
          39.63, 39.67, 39.96, 39.8, 39.89, 39.85, 39.9, 39.81, 39.81, 39.68 ])


F = np.array([ [ 1, 0.25 ], 
               [ 0,    1 ] ])
G = np.array([ 0.5 * (0.25**2), 0.25 ])
Q = np.array([ [ 0.25 * (0.25)**4, 0.5 * (0.25)**3 ], 
               [  0.5 * (0.25)**3,         0.25**2 ] ]) * (0.1**2)
P = np.array( [ [ 500,   0 ],
                [   0, 500 ] ])

H = np.array([ 1, 0 ] )
I = np.identity(2)
R =  400 

'''
initialization
===================================
We don't know the rocket location; we will set initial position and velocity to 0.
'''

x0 = np.array([0, 0])

'''
We also don't know what the rocket acceleration is, however we can assume that 
acceleration is greater than zero, let's assume:
'''
g = 9.8

'''
Since our initial state vector is a guess, we will set very high estimate uncertainty. 
High estimate uncertainty resulting high Kalman Gain giving a high weight for 
measurement.
'''
P0 = np.array([ [ 500,   0 ], 
                [   0, 500 ] ])

'''
Prediction
Now, we can predict the next state based on the initialization values.
'''
estims = [ np.matmul(F, x0) + G * (1 + g) ]
estims_uncertainty = [ np.matmul(np.matmul(F, P0), np.transpose(F)) + Q ]

for rnd in range(0, len(z_alt)):
    # state update
    K = np.matmul(estims_uncertainty[-1], np.matrix.transpose(H)) /  \
            (
                np.matmul(
                    np.matmul(H, estims_uncertainty[-1]),
                    np.matrix.transpose(H)
                    ) + R
            )
    
    # Δt = 0.25, Δt * accel = velocity
    next_estims = estims[-1] + K * ( [z_alt[rnd], (z_accl[rnd] * 0.25) ] - estims[-1])  
   
    next_estims_uncertainty = np.matmul(
                                np.matmul(
                                    (I - K.reshape(-1, 1) * H),
                                    estims_uncertainty[-1]
                                    ),
                                np.transpose(I - K.reshape(-1, 1) * H)
                                ) + K.reshape(-1, 1) * R * K
                                
    # predict
    estims.append( 
            np.matmul(F, next_estims) + G * (z_accl[rnd] + g)
        )
    
    estims_uncertainty.append(
            np.matmul(np.matmul(F, next_estims_uncertainty), np.transpose(F)) + Q
        )
    
estims = estims[1:]
estims_uncertainty = estims_uncertainty[1:]

estims_x = [ x[0] for x in estims ]

z_x = range(0, len(z_alt))
plt.plot(z_x, z_alt, marker='o')
plt.plot(z_x, estims_x, marker='x', color='r')
plt.title('Rocket Altitude')
plt.legend(['Measurements', 'Estimates'])

plt.xlabel('Time (s)')
plt.ylabel('Altitude (m)')
plt.show()  
    