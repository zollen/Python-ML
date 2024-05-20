'''
Created on May 20, 2024

@author: STEPHEN
@url: https://www.sfu.ca/math-coursenotes/Math%20157%20Course%20Notes/sec_Optimization.html
@desc:

Find the largest rectangle (that is, the rectangle with largest area) that fits inside the graph of the 
parabola y = x^2 below the line y = a (a is an unspecified constant value), with the top side of the 
rectangle on the horizontal line y = a;
 
a - constant, b - variable

y = a - top
A(x)  = (a - x^2) * 2x 
A(x)  = a2x - 2x^3 
A'(x) - a2 - 6x^2 = 0
        a2 = 6x^2
        a (1/3) = x^2
        x = sqrt(a/3)

Maximum area:
A(sqrt(a/3)) = (a - (sqrt(a/3)^2) * 2 * sqrt(a/3) = 2(a - a/3)*sqrt(a/3)
 
'''