'''
Created on Jul. 31, 2020

@author: zollen
'''
import matplotlib.pyplot as plt
import seaborn as sb

girls_grades = [ 89, 90, 70, 89, 100, 80, 90, 100, 80, 34 ]
boys_grades = [ 30, 29, 49, 48, 100, 48, 38, 45, 20, 30 ]
grades_range1 = [ 10, 20, 30, 40, 50, 60, 70, 80, 90, 100 ]
grades_range2 = [ 5, 15, 25, 35, 45, 55, 65, 75, 85, 95 ]

sb.set_style("whitegrid")

fig = plt.figure()

ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
ax.scatter(grades_range1, girls_grades, color='r')
ax.scatter(grades_range2, boys_grades, color='b')
ax.set_xlabel('Grade Range')
ax.set_ylabel('Grade Scored')
ax.set_title("scattor plot")

plt.show()