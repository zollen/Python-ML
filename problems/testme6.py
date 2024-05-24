'''
Created on May 21, 2024

@author: STEPHEN
@url: https://www.sfu.ca/math-coursenotes/Math%20157%20Course%20Notes/sec_Optimization.html
@desc:


You are making cylindrical containers to contain a given volume. Suppose that the top and bottom are 
made of a material that is N times as expensive (cost per unit area) as the material used for the 
lateral side of the cylinder.

Find (in terms of N) the ratio of height to base radius of the cylinder that minimizes the cost of 
making the containers.

A = 2 pi rh + 2 pi r^2
C = c(2 pi rh) + N c(2 pi r^2)

V = pi r^2 h
h = V / (pi r^2)

C(r)  = c ( 2 pi r (V / (pi r^2)) ) + N c ( 2 pi r^2 )
C(r)  = c ( 2 V / r ) + 2 N c pi r^2

C'(r) = - 2 c V / r^2 + 4 N c pi r = 0 
      4 N c pi r = 2 c V / r^2
      2 N pi r^3 = V
      r = ∛( V / ( 2 N pi ) ) 
      

C"(r) = 4 c V / r^3 + 4 N c pi is positive when r is positive, hence this is a global minimum


C'(r) = 0
C'(r): r = ∛( V / ( 2 N pi ) ) 
       h = V / (pi r^2)
              
   h^3     V^3 / ( pi^3 r^6 )      V^2 / ( p^3 r^6 )        V^2            ( 2 N pi )      2 N V^2
   --- = --------------------- = -------------------- = ------------- * -------------- = -------------
   r^3      V / ( 2 N pi )           1 / ( 2 N pi )      ( pi^3 r^6 )            1          pi^2 r^6
       


'''