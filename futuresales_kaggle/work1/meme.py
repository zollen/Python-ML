"""
Created on Jul. 23, 2021

@author: zollen
"""

import time
import warnings

import pandas as pd

import futuresales_kaggle.lib.future_lib as ft

warnings.filterwarnings("ignore")


train = pd.read_csv("../data/monthly_train.csv")
test = pd.read_csv("../data/monthly_test2.csv")

"""
start_time = time.time()
train, test = ft.add_sales_proximity(10, 
                        train, test)
end_time = time.time()
print(train.head())

print("DONE ", (end_time - start_time))
"""



def calculate_proximity(vals):
    score = 0
    lgth = len(vals)
    total = lgth ** 2
    for idx, row in zip(range(1, lgth + 1), vals):
        score += row * idx ** 2 / total
    return score


"""
          date_block_num  item_cnt_month  sales_proximity
14242838               0               0             0.00
14242839               1               0             0.00
14242840               2               0             0.00
14242841               3               0             0.00
14242842               4               0             0.00
14242843               5               0             0.00
14242844               6               0             0.00
14242845               7               0             0.00
14242846               8               0             0.00
14242847               9               1             1.00
14242848              10               0             0.81
14242849              11               2             2.64
14242850              12               0             2.11
14242851              13               0             1.64
14242852              14               0             1.23
14242853              15               0             0.88
14242854              16               0             0.59
14242855              17               1             1.36
14242856              18               0             1.00
14242857              19               0             0.72
14242858              20               0             0.51
14242859              21               0             0.36
14242860              22               0             0.25
14242861              23               0             0.16
14242862              24               0             0.09
14242863              25               0             0.04
14242864              26               0             0.01
14242865              27               0             0.00
14242866              28               0             0.00
14242867              29               0             0.00
14242868              30               0             0.00
14242869              31               0             0.00
14242870              32               0             0.00
14242871              33               0             0.00

"""
kk = [
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    1,
    0,
    2,
    0,
    0,
    0,
    0,
    0,
    1,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
]

for i in range(0, len(kk)):
    print(calculate_proximity(kk[0 if i - 9 <= 0 else i - 9 : i + 1]))
