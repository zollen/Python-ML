'''
Created on Mar 6, 2024

@author: STEPHEN

Assumption: start 0, k=3 ants

k = np.array([0,1,2,3.....last_ant])
p[k] = possible_start_states()
current = 0

while not empty(k) and current < totaL_number_of nodes:
    
    n[k] = next_moves[p[k], current % 10]
    
    ants[k, p[k], n[k]] = c[p[k], n[k]]  
    
    p[k] = n[k]
    
    k = np.remove(k, where nonzero(ants[:,end_states]))
  
    current++

ants[k, 0, 0] = 999999999

L = 1 / sum (ants[:,~end_states])
ant[:, np.where(ants[k] != 0)] = L[k]

'''


import numpy as np

k = np.array([0, 1, 2])
s = np.array([0, 0, 0])
n = np.array([0, 1, 2, 3, 4, 6, 7, 8, 9, 2, 1, 2, 2])
c = np.array([[11, 12, 13, 14, 15],
              [16, 17, 18, 19, 20],
              [21, 22, 23, 24, 25],
              [26, 27, 28, 29, 30],
              [30, 31, 32, 33, 34]])
ants = np.arange(75).reshape(3, 5, 5)
print(ants)
print(n[0:3])
print(ants[k, s, n[k]])
print("=== first round assignment ===")
print(c[s[k],n[k]])
ants[k, s, n[k]] = c[s[k],n[k]]
print(ants)
print("==== general round assignment ===")
p = n[k] # [0, 1, 2]
n = np.array([2, 3, 1])
ants[k, p[k], n[k]] = c[p[k], n[k]]
print(ants)
print("===== Sum of each ant======")
print(np.sum(ants, axis=(1,2)))  # need improvement exclude rows of end states when sum
print("===========")
print(ants[:,4])
print("===============")
ants = np.zeros((3, 5, 5))
ants[0, 3, 0] = 1
ants[0, 4, 3] = 1
#ants[1, 3, 1] = 1
ants[2, 3, 2] = 1
#ants[1, 4, 1] = 1
print(ants)
print("+++++++++++++++++++++")
print(ants[:,[3,4]])
print("+++++++++++++++++")
print(np.nonzero(ants[:,[3,4]]))
print("========== remove done ant ===========")
k = np.delete(k, np.nonzero(ants[:,[3,4]])[0])
print(k)
print("========== assign 99999999 to non done ==========")
# ants[k, start_states[0], 0]
k = [1, 2]
ants[k, 0, 0] = 99999999
print(ants)
print("============ sum all paths except end_states ==========")
m = np.array([0,1,2,3,4])
end_states = np.array([3,4])
ants = np.arange(75).reshape(3, 5, 5)
print(ants)
print(ants[:,np.delete(m, end_states)])
print("+++")
print(np.sum(ants[:,np.delete(m, end_states)], axis=(1,2)))
print("============ fill in the L ===============")
ants = np.zeros((3, 5, 5))
ants[0, 3, 0] = 1
ants[0, 4, 3] = 1
ants[1, 3, 1] = 1
ants[2, 3, 2] = 1
ants[2, 4, 1] = 1
ants[1, 4, 1] = 1
L = np.array([11,12,13])
k = np.array([0,1,2])
ants[np.where(ants != 0)] = L[np.where(ants != 0)[0]]
print(ants)
print("==============================================")
aa = np.array([0, 1, 2, -1, 5, -1, 6])
vv = np.array([10,11,12,13,14,15,16])
aa[aa == -1] = vv[aa == -1]
print(aa)
print("=======check next_moves[k] contains end_states ===============")
ants = np.array([0, 1, 2, 3, 4, 5, 6])
kk =   np.array([0, 2, 7, 3, 7, 5, 2])
ee = np.array([5,7])

ants = np.delete(ants, np.in1d(kk, ee))
print(ants)
