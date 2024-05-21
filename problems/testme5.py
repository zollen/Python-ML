'''
Created on May 21, 2024

@author: STEPHEN
@url: https://www.sfu.ca/math-coursenotes/Math%20157%20Course%20Notes/sec_Optimization.html
@desc:


If you fit the largest possible cone inside a sphere, what fraction of the volume of the sphere is 
occupied by the cone? (Here by “cone” we mean a right circular cone, i.e., a cone for which the base 
is perpendicular to the axis of symmetry, and for which the cross-section cut perpendicular to the 
axis of symmetry at any point is a circle.)

Cone
====
V = pi * r^2 * h /3

Sphere
======
V = 4/3 * pi * r^3


R = radius of the sphere
r = radius of the base of the cone
h = height of the cone

(h - R)^2 + r^2 = R^2
V(h) = pi * (R^2 - (h - R)^2) * h / 3
V(h) = pi * h / 3 * (R^2 - (h^2 - 2hR + R^2))
V{h) = pi * h / 3 * (R^2 - h^2 + 2hR - R^2)
V(h) = pi * h / 3 * (2hR - h^2)
V(h) = pi * h^2 * 2 / 3 * R - pi / 3 * h^3   
V'(h) = - pi h^2 + (4/3) pi hR = 0
V'(h) = 4/3 pi hR = pi h^2
V'(h) = 4/3 R = h
Therefore, h = 0 or h = 4/3 R

h = 4/3 R
r^2 = (R^2 - (4/3 R - R)^2) = (R^2 - 1/9R^2) = 8/9 R^2
V(cone) = pi * 8/9R^2 * 4/9R = pi * 32/81 * R^3 
V(sphere) = 4/3 * pi * R^3

V(cone)/V(sphere) = (pi * 32/81 * R^3)  / (4/3 * pi * R^3)
V(cone)/V(sphere) = 32/81 / 4/3 = 32/81 * 3 /4 = 8 / 27 = 30% roughly



'''
