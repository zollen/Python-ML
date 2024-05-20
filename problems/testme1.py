'''
Created on May 20, 2024

@author: STEPHEN
@url: https://www.sfu.ca/math-coursenotes/Math%20157%20Course%20Notes/sec_Optimization.html
@desc:
You want to sell a certain number of items in order to maximize your profit. Market research tells you 
that if you set the price at $1.50, you will be able to sell 5000 items, and for every 10 cents you 
lower the price below $1.50 you will be able to sell another 1000 items. Suppose that your fixed 
costs ( “start-up costs” ) total $2000, and the per item cost of production ( “marginal cost” ) 
is $0.50.

Find the price to set per item and the number of items sold in order to maximize profit, and also 
determine the maximum profit you can get.

x - delta of price
y - number of items


P(x) = (5000 + 10000x) * (1.5 - x) - 2000 - (5000 + 10000x) * 0.5
P(x) = (7500 - 5000x + 15000x - 10000x^2) - 2000 - 2500 - 5000x
P(x) = 3000 + 5000x - 10000x^2
P'(x) = 5000 - 20000x
      5000 = 20000x
      x = 5000/20000
      x = 0.25
      
1.5 - 0.25 = 1.25
P is maximum when price is $1.25

'''
