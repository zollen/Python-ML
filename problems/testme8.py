'''
Created on May 21, 2024

@author: STEPHEN
@url: https://www.sfu.ca/math-coursenotes/Math%20157%20Course%20Notes/sec_Optimization.html
@desc:


Suppose you want to reach a point A that is located across the sand from a nearby road (see 
diagram below). Suppose that the road is straight, and b is the distance from A to the closest point C 
on the road. Let v be your speed on the road, and let w, which is less than v, be your speed on the 
sand. Right now you are at the point D, which is a distance a from C. At what point B should you turn 
off the road and head across the sand in order to minimize your travel time to A?

speed_v > speed_w
T(x) = (a - x)/v +  (sqrt(x^2 + b^2))/w
T'(X) = - 1/v + x / (w * sqrt(x^2 + b^2)) = 0
      = w * sqrt(x^2 + b^2) = vx
      = w^2 (x^2 + b^2) = v^2x^2
      = w^2 * b^2 = (v^2 - w^2)x^2
    x = wb / (sqrt(v^2 - w^2))
    
T"(x) = b^2 / ((x^2 + b^2)^(3/2) * w) 
T"(x) > 0, it is always positive, so it is a global minimum

T(0)= a / v + b / w
T(a) = sqrt(a^2 + b^2) / w

T(0) > T(a), so the minimum occurs when x = a


'''