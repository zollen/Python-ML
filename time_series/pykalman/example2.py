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
Problem #1 - Vechicle location estimation
In this example, we would like to estimate the location of the vehicle on the XY plane.

The vehicle has an onboard location sensor that reports X and Y coordinates of the 
system.

We assume a constant acceleration dynamics.

Thus, the state extrapolation equation can be simplified to:
    X_estim(n+1,n) = F * X_estim(n,n)
    
The system state x(n) is defined by:
    x1 = position(x)         y1 = position(y)
    x2 = velocity(x)         y2 = velocity(y)
    x3 = accelation(x)       y3 = accelation(y)
    x(n) = [ x1(n), x2(n), x3(n), y1(n), y2(n), y3(n) ]
System of Equations    
    x1(n+1,n) = x1(n,n) + x2(n,n) * Δt + 0.5 * x3(n,n) * Δt^2
    x2(n+1,n) = x2(n,n) + x3(n,n) * Δt
    x3(n+1,n) = x3(n,n)
    y1(n+1,n) = y1(n,n) + y2(n,n) * Δt + 0.5 * y3(n,n) * Δt^2
    y2(n+1,n) = y2(n,n) + y3(n,n) * Δt
    y3(n+1,n) = y3(n,n)
    z1(n+1,n) = z1(n,n) + z2(n,n) * Δt + 0.5 * z3(n,n) * Δt^2
    z2(n+1,n) = z2(n,n) + z3(n,n) * Δt
    z3(n+1,n) = z3(n,n)
Matrix Form    
    x1(n+1,n)    1  Δt  0.5Δt^2    0  0       0       x1(n,n)
    x2(n+1,n)    0   1       Δt    0  0       0       x2(n,n)
    x3(n+1,n)    0   0        1    0  0       0       x3(n,n)
    y1(n+1,n) =  0   0        0    1 Δt 0.5Δt^2   *   y1(n,n)   
    y2(n+1,n)    0   0        0    0  1      Δt       y2(n,n)
    y3(n+1,n)    0   0        0    0  0       1       y3(n,n)



The process noise matrix
    
w - provided by the problem - random vaiance in accelaration
Q(n) = E(w(n) * transpose(w(n)))
      0.25Δt^4  0.5Δt^3  0.5Δt^2           0        0        0
       0.5Δt^3      Δt^2       Δt          0        0        0 
       0.5Δt^2        Δt        1          0        0        0
 Q =         0         0        0   0.25Δt^4  0.5Δt^3  0.5Δt^2  *  (σ^2)
             0         0        0    0.5Δt^3     Δt^2       Δt
             0         0        0    0.5Δt^2       Δt        1
        

The measurement equation

z(n) = H * x(n) + v(n)

                    x1(n)
                    x2(n)
  z_x(n)            x3(n)
  z_y(n) =  H *     y1(n)
                    y2(n)
                    y3(n)
        
    H = 1 0 0 0 0 0            
        0 0 0 1 0 0
    
    
The measurement uncertainity
                
       R(n) = σ[x,x]^2  σ[y,x]^2  
              σ[x,y]^2  σ[y,y]^2
    Assume x uncertainity would not affect y uncertainity
       R(n) = σ[x,x]^2         0
                     0  σ[y,y]^2 
                     

The Kalman Gain
           Estim_Uncertainty(n,n-1) * transpose(H)
K = ---------------------------------------------------
      H * Estim_Uncertainty(n,n-1) * transpose(H) + Rn
      
    Assume x uncertainity would not affect y uncertainity

K(1:1,n) K(1:2,n)   p(x1:x1,n:n-1)  p(x1:x2,n:n-1)  p(x1:x3,n:n-1)              0               0               0     1 0     |                   p(x1:x1,n:n-1)  p(x1:x2,n:n-1)  p(x1:x3,n:n-1)              0               0               0     1 0                         |^(-1)
K(2:2,n) K(2:2,n)   p(x2:x1,n:n-1)  p(x2:x2,n:n-1)  p(x2:x3,n:n-1)              0               0               0     0 0     |                   p(x2:x1,n:n-1)  p(x2:x2,n:n-1)  p(x2:x3,n:n-1)              0               0               0     0 0                         |
K(3:1,n) K(3:2,n) = p(x3:x1,n:n-1)  p(x3:x2,n:n-1)  p(x3:x3,n:n-1)              0               0               0  *  0 0  *  |   1 0 0 0 0 0  *  p(x3:x1,n:n-1)  p(x3:x2,n:n-1)  p(x3:x3,n:n-1)              0               0               0  *  0 0  +  σ[x,x]^2         0  |
K(4:1,n) K(4:2,n)                0               0              0  p(y1:y1,n:n-1)  p(y1:y2,n:n-1)  p(y1:y3,n:n-1)     0 1     |   0 0 0 1 0 0                  0               0              0  p(y1:y1,n:n-1)  p(y1:y2,n:n-1)  p(y1:y3,n:n-1)     0 1            0  σ[y,y]^2  |
K(5:1,n) K(5:2,n)                0               0              0  p(y2:y1,n:n-1)  p(y2:y2,n:n-1)  p(y2:y3,n:n-1)     0 0     |                                0               0              0  p(y2:y1,n:n-1)  p(y2:y2,n:n-1)  p(y2:y3,n:n-1)     0 0                         |
K(6:1,n) K(6:2,n)                0               0              0  p(y3:y1,n:n-1)  p(y3:y2,n:n-1)  p(y3:y3,n:n-1)     0 0     |                                0               0              0  p(y3:y1,n:n-1)  p(y3:y2,n:n-1)  p(y3:y3,n:n-1)     0 0                         |


The State Update Equation
Estim_Uncertainty(n,n) = (1 - K * H) * Estim_Uncertainty(n,n-1) * transpose(1 - K * H) + K * R * transpose(K)
'''  
np.set_printoptions(formatter={"float_kind": lambda x: "%0.4f" % x})
dt = 1     # delta(t): one seconds
ra = 0.15  # random acceleration standard deviation σ^2
                   
F = np.array([
        [1, dt, 0.5*(dt**2),  0,  0,           0],
        [0,  1,          dt,  0,  0,           0],
        [0,  0,           1,  0,  0,           0],
        [0,  0,           0,  1, dt, 0.5*(dt**2)],
        [0,  0,           0,  0,   1,         dt],
        [0,  0,           0,  0,   0,          1]      
        ])

Q = np.array([
        [0.25*(dt**4), 0.5*(dt**3), 0.5*(dt**2),            0,           0,           0],
        [ 0.5*(dt**3),        dt*2,          dt,            0,           0,           0],
        [ 0.5*(dt**2),          dt,           1,            0,           0,           0],
        [           0,           0,           0, 0.25*(dt**4), 0.5*(dt**3), 0.5*(dt**2)],
        [           0,           0,           0,  0.5*(dt**3),        dt*2,          dt],
        [           0,           0,           0,  0.5*(dt**2),          dt,           1]
    
    ]) * (0.15**2)

    
R = np.array([
        [ ra**2,    0 ],
        [     0, ra**2]
    ])

H = np.array([
        [ 1, 0, 0, 0, 0, 0],
        [ 0, 0, 0, 1, 0, 0]
    ])

I = np.identity(6);

z_x = np.array([ -393.66, -375.93, -351.04, -328.96, -299.35, -273.36, -245.89, -222.58, 
        -198.03, -174.17, -146.32, -123.72, -103.47, -78.23, -52.63, -23.34, 25.96,
        49.72, 76.94, 95.38, 119.83, 144.01, 161.84, 180.56, 201.42, 222.62, 239.4,
        252.51, 266.26, 271.75, 277.4, 294.12, 301.23, 291.8, 299.89 ]).reshape(-1, 1)
z_y = np.array([ 300.4, 301.78, 295.1, 305.19, 301.06, 302.05, 300, 303.57, 296.33, 297.65,
        297.41, 299.61, 299.6, 302.39, 295.04, 300.09, 294.72, 298.61, 294.64, 284.88,
        272.82, 264.93, 251.46, 241.27, 222.98, 203.73, 184.1, 166.12, 138.71, 119.71,
        100.41, 79.76, 50.62, 32.99, 2.14 ]).reshape(-1, 1)
        
z_xy = np.concatenate((z_x, z_y), axis=1)


# initialization to zero

estims = [ 
    np.array([
        0,
        0,
        0,
        0,
        0,
        0
    ]) 
]

P0 = np.array([
        [ 500,   0,    0,   0,    0,     0 ],
        [   0, 500,    0,   0,    0,     0 ],
        [   0,   0,  500,   0,    0,     0 ],
        [   0,   0,    0, 500,    0,     0 ],
        [   0,   0,    0,   0,  500,     0 ],
        [   0,   0,    0,   0,    0,   500 ]
    ])


estims_uncertainty = [ np.matmul(F, np.matmul(P0, np.transpose(F))) + Q ]



for rnd in range(1, len(z_xy)):
    # state update
    K = np.matmul(
            np.matmul(estims_uncertainty[-1], np.matrix.transpose(H)), 
            np.linalg.inv(
                np.matmul(
                    np.matmul(H, estims_uncertainty[-1]),
                    np.matrix.transpose(H)
                    ) + R
                )
            )
    
    next_estims = estims[-1] + np.matmul(K, z_xy[rnd] - np.matmul(H, estims[-1]))
    next_estims_uncertainty = np.matmul(
                                np.matmul(
                                    (I - np.matmul(K, H)),
                                    estims_uncertainty[-1]
                                    ),
                                np.transpose(I - np.matmul(K, H))
                                ) + np.matmul(K, np.matmul(R, np.transpose(K)))
    
    # predict
    estims.append(next_estims)  # constant dynamic model
    estims_uncertainty.append(
            np.matmul(F, np.matmul(
                            next_estims_uncertainty,
                            np.transpose(F)
                        )
                    ) + Q
        )

estims = estims[1:]
estims_uncertainty = estims_uncertainty[1:]


estims_x = [ x[0] for x in estims ]
extims_y = [ x[3] for x in estims ]


plt.plot(z_x, z_y, marker='o')
plt.plot(estims_x, extims_y, marker='x', color='r')
plt.title('Vechile Position')
plt.legend(['Measurements', 'Estimates'])

plt.xlabel('X (m)')
plt.ylabel('Y (m)')
plt.show()    
