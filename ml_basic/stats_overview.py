'''
Created on Aug. 12, 2020

@author: zollen
'''
"""

AUC-ROC
======= 
Area under the curve - Receiver Operating Characteristic
https://www.tutorialspoint.com/machine_learning_with_python/images/area_under_curve.jpg
It is a tradeoff between true-positive and false-positive under various decision thresholds.
AUC-ROC close to 1 means it is a good estimator

F1-Score
========
F1 = 2 * [ (precision * recall) / (precision + recall) ]
F1 score is having equal relative contribution of precision and recall.
F1 shows the overall performance of the predictibility of your estimator

R2^2 or R2 square
=================
x = [ x1, x2, x3, .... xn ]
R2^2 = ∑ [ ( (xi - mean(x))^2 - (xi - your_model_prediction(xi))^2 ) / (xi - mean(x))^2 ]
R2^2 is a relative ratio between: the sum of all discrepancies between the mean,
                             and the sum of all discrepancies of your model predictions
R2^2 shows how much variations your model has covered
R2^2 is always between 0 and 1, higher the better

Log Loss (Logarithmic Loss)
===========================
It is also called Logistic regression loss or cross-entropy loss.
It shows the amount of uncertainty of our prediction based on how much it varies from 
    the actual labels

# converting all predicted labels if exactly 0 or 1 = then => 0.000000000001 or 0.9999999999999
# log(0) or log(1 - 1) is negative infinite!
p = numpy.clip(predicted_labels, 1e-15, 1 - 1e-15) 
If Actual label = 1:
    LogLoss = -log(p)
else:
    LogLoss = -log(1 - p)
    
Mean Absoulte Error (MAE)
=========================
MAE = 1 / n * ( ∑ | actual labels - predicted labels | )

Mean Square Error (MSE)
=======================
MSE = 1 / n * ( ∑ ( actual labels - predicted labels )^2 )

MSE would have a larger value than MAE if the discrepancies are large (because of the square effect)

"""