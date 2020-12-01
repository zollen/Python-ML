'''
Created on Dec. 1, 2020

@author: zollen
'''

'''
Least Square Regression
-----------------------
y = a + b x1 + c x2 + d x3 + ... 
a = y-intercept
b, c, d = slope
Least square attempts to minimize b, c and d, and then find a, b, c and d 



Ridge Regression (good if most parameters are useful to the find y)
----------------
Ridge Regression = Least Square Regression + ridge penalty
ridge penalty = lambda * b^2
When mininizing the ridge regression, the slopes of all useless paramters will close to 0, 
but never 0



Lasso Regression (good if most parameters are useless)
------------------------------------------------------
Lasso Regression = Least Square Regression + lasso penalty
lasso penalty = lambda * | b |
When minimizing the lasso regression, the slopes of all useless paramters will become 0



Eleastic Net Regression (when there are tons of parameters and not sure which is useful or not)
-----------------------------------------------------------------------------------------------
Eleastic Net Regression = Least Square Regression + ridge penalty + lasso penalty
Now we have the best of both world

ridge penalty and lasso penalty each has their own lambda value









'''