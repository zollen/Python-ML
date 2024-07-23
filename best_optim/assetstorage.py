'''
Created on Jul 16, 2024

@author: STEPHEN
@url: https://towardsdatascience.com/how-to-solve-an-asset-storage-problem-with-mathematical-programming-3b96b7cc22d1
@desc The assortment problem
A general version of the assortment problem involves selecting a subset of items from a larger set to 
maximize a certain objective, often revenue or profit, under constraints such as shelf space or budget. 
It’s a common problem in retail and operations management, involving optimization techniques and often 
consumer choice modeling. The specific assortment problem we are dealing with in this article is also 
known as the 2D rectangle packing problem, which frequently appears in logistics and production contexts.

@data
Quantity      Heights       Widths
4              3              4 
10             1              1
8              1              2
5              2              3
2              4              2
3              2              6
2              4             10


'''
import numpy as np

from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


class AssortmentProblem(ElementwiseProblem):
    
    def __init__(self, value):

        self.value = value
        self.MAX_VALUE = 9
        
        '''
        MAX: 10 POP: 2000, EVAL: 100000  5x9=45
        [[1, 3, 4],[1, 1, 1],[1, 1, 2],[1, 2, 3],[1, 4, 3],[1, 5, 2]]
        
        MAX: 7  POP: 1800, EVAL: 100000  9x6=54
        [[1, 3, 4],[1, 1, 1],[1, 1, 2],[1, 2, 3],[1, 4, 3],[1, 5, 2],[1, 3, 2]
        
        MAX: 11 POP: 1800, EVAL: 100000  7x10=70
        [[1, 3, 4],[1, 1, 1],[1, 1, 2],[1, 2, 3],[1, 4, 3],[1, 5, 2],[1, 3, 2],[1, 3, 4]]
         
        MAX:  9 POP: 1800, EVAL: 100000  10x9=90
        [[1, 3, 4],[1, 1, 1],[1, 1, 2],[1, 2, 3],[1, 4, 3],[1, 5, 2],[1, 3, 2],[1, 3, 4],[1, 2, 5]]
        '''
        num_params = value.shape[0] * 3
        xl = np.full(num_params, 0.0)
        xu = np.full(num_params, 1.0)
        
        self.KEY_COLS = np.array(range(value.shape[0] * 2))
        self.X_COLS = np.array(range(value.shape[0]))
        self.Y_COLS = np.delete(self.KEY_COLS, self.X_COLS)                    
        self.R_COLS = np.delete(np.array(range(num_params)), self.KEY_COLS)
        
        super().__init__(n_var=num_params, n_obj=1, n_constr=1, xl=xl, xu=xu)
        
    def _show(self, X, data):
        
        total_X, total_Y = self._dimensions(X)
        X, _, _ = self._preprocessing(X)
       
        num = len(self.X_COLS)
        
        _, ax = plt.subplots()
        
        for i in self.X_COLS:
            coords = (X[i], X[num + i])
        
            if X[num * 2 + i] == 1:
                wid = data[i, 1]    
                hig = data[i, 2]
            else:
                wid = data[i, 2]
                hig = data[i, 1]
            
            ax.add_patch(Rectangle(coords, wid, hig,
                      edgecolor = 'black',
                      facecolor = "Grey",
                      fill = True,
                      alpha = 0.5,
                      lw=2))
            
        ax. set_xlim(0, total_X )
        ax. set_ylim(0, total_Y )
        
        ax.set_xticks(range(int(total_X)+1))
        ax.set_yticks(range(int(total_Y)+1))
        ax.grid()
        ax.set_title(f" Total area {total_X} x {total_Y} = {res.F}")
        
        plt.show()
        
   
    def _preprocessing(self, X, convert=True):
        X = np.copy(X)
        
        if convert == True:
            X[self.KEY_COLS] = np.floor(X[self.KEY_COLS] * self.MAX_VALUE)
            X[self.R_COLS] = np.where(X[self.R_COLS] > 0.5, 1, 0)
            
        X = X.astype(int)
            
        wid = X[self.X_COLS] + (self.value[self.X_COLS, 1] * X[self.R_COLS]) + \
            ((1 - X[self.R_COLS]) * self.value[self.X_COLS, 2])
        hig = X[self.Y_COLS] + (self.value[self.X_COLS, 2] * X[self.R_COLS]) + \
            ((1 - X[self.R_COLS]) * self.value[self.X_COLS, 1])
            
        return X, wid, hig
    
    def _dimensions(self, X, convert=True):
        
        X, wid, hig = self._preprocessing(X, convert)
                  
        return np.max(wid), np.max(hig)
        
    def _each_obj(self, X):
        
        X, wid, hig = self._preprocessing(X)
        
        num = len(self.X_COLS)
        
        for i in self.X_COLS:
            print("[%s] {%s} x:[%s]-[%s]  y:[%s]-[%s]" % (i, X[2 * num + i], X[i], wid[i], X[num + i], hig[i]))
            
        return 
    
    def _check(self, X, convert=True):
        
        X, wid, hig = self._preprocessing(X, convert)
        
        xres = np.ones((self.value.shape[0], self.value.shape[0]), dtype=int)
        idx = 0
        for i in self.X_COLS:
            xres[idx] = np.where(X[i] >= wid, 1, 0)
            idx += 1
        
        yres = np.ones((self.value.shape[0], self.value.shape[0]), dtype=int)
        idx = 0
        for i in self.Y_COLS:
            yres[idx] = np.where(X[i] >= hig, 1, 0)
            idx += 1
                
        for i in self.X_COLS:
            for j in self.X_COLS:
                if i != j:
                    if xres[i, j] + xres[j, i] + yres[i, j] + yres[j, i] <= 0:
                        return -1
        
        return 1
        
    def _evaluate(self, X, out, *args, **kwargs):
        
        wid, hig = self._dimensions(X)
        
        out['G'] = self._check(X) * -1  
        out['F'] = wid * hig 
       
            


samples = np.array([[4, 3, 4], 
                    [10, 1, 1],
                    [8, 1, 2],
                    [5, 2, 3],
                    [2, 4, 2],
                    [3, 2, 6],
                    [2, 4, 10]])

samples = np.array([[1, 3, 4], 
                    [1, 1, 1],
                    [1, 1, 2],
                    [1, 2, 3],
                    [1, 4, 3],
                    [1, 5, 2],
                    [1, 3, 2],
                    [1, 3, 4],
                    [1, 2, 5]])


data = np.zeros((np.sum(samples[:,0]), 3), dtype='int32')
i = 0
for q, h, w in samples:
    for _ in range(q):
        data[i,0] = i
        data[i,1] = h
        data[i,2] = w
        i += 1

problem = AssortmentProblem(data)

algorithm = GA(pop_size=1800, eliminate_duplicates=True)

res = minimize(problem,
               algorithm,
               termination=('n_evals', 100000),
               seed=1,
               verbose=True)
if res.X is not None:
    X, _, _ = problem._preprocessing(res.X)
    print("Best solution found: \nX = %s\nF = %s" % (X, res.F))
    width, height = problem._dimensions(res.X)
    print("Width = %s, Height = %s" % (width, height))
    print("Check: %s" % (problem._check(res.X)))
    problem._each_obj(res.X)
    problem._show(res.X, data)
else:
    print("No solution found!!!")
