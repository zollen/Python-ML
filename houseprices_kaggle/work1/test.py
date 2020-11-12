'''
Created on Nov. 2, 2020

@author: zollen
'''
import os
from pathlib import Path
import numpy as np
import pandas as pd
import pprint
from sklearn.model_selection import KFold, cross_val_score
from xgboost import XGBClassifier
import warnings
from sklearn.impute import SimpleImputer
from xgboost.sklearn import XGBRegressor



warnings.filterwarnings('ignore')
pd.set_option('max_columns', None)
pd.set_option('max_rows', None)
pd.set_option('display.width', 1000)

pp = pprint.PrettyPrinter(indent=3) 

PROJECT_DIR=str(Path(__file__).parent.parent)  
train_df = pd.read_csv(os.path.join(PROJECT_DIR, 'data/train.csv'))
test_df = pd.read_csv(os.path.join(PROJECT_DIR, 'data/test.csv'))


train_df.drop(columns = ['PoolQC', 'MiscFeature', 'Alley', 'Fence'], inplace = True)
test_df.drop(columns = ['PoolQC', 'MiscFeature', 'Alley', 'Fence'], inplace = True)



all_df = pd.concat([ train_df, test_df ]) 

kk = []
def fillValue(name, fid):
    
    global all_df
    
    col_types = all_df.columns.to_series().groupby(all_df.dtypes)
    categorical_columns = []
    numeric_columns = []
       
    for col in col_types:
        if col[0] == 'object':
            categorical_columns = col[1].unique().tolist()
        else:
            numeric_columns += col[1].unique().tolist()
            
    if name in numeric_columns:
        numeric_columns.remove(name)
        
    if name in categorical_columns:
        categorical_columns.remove(name)
        
    numeric_columns.remove('Id')
    numeric_columns.remove('SalePrice')
    
    all_ddf = all_df.copy()
    
    for colnam in categorical_columns:
        keys = all_df[colnam].unique()
        
        if np.nan in keys:
            keys.remove(np.nan)
            
        vals = [ i  for i in range(0, len(keys))]
        labs = dict(zip(keys, vals))
        all_ddf[colnam] = all_ddf[colnam].map(labs)
            
    
    imputer = SimpleImputer()
    all_ddf[categorical_columns + numeric_columns] = imputer.fit_transform(all_ddf[categorical_columns + numeric_columns])
    all_ddf[categorical_columns + numeric_columns] = all_ddf[categorical_columns + numeric_columns].round(0).astype('int64')
    

    all_ddf = pd.get_dummies(all_ddf, columns = categorical_columns) 
    single_df = all_ddf[all_ddf['Id'] == fid]  
    all_ddf = all_ddf[all_ddf[name].isna() == False]
    

    categorical_columns = list(set(all_ddf.columns).symmetric_difference(numeric_columns + ['Id', 'SalePrice', name]))
    all_columns = numeric_columns + categorical_columns
    
    if all_df[name].dtypes == 'object':
        keys = all_ddf[name].unique().tolist()

        if np.nan in keys:
            keys.remove(np.nan)
        
        vals = [ i  for i in range(0, len(keys))]
        labs = dict(zip(keys, vals))
        rlabs = dict(zip(vals, keys))
        all_ddf[name] = all_ddf[name].map(labs)
        model = XGBClassifier(random_state = 87)
    else:
        model = XGBRegressor(random_state = 87)
        

    model.fit(all_ddf[all_columns], all_ddf[name])
    prediction = model.predict(single_df[all_columns])

    all_df.loc[all_df['Id'] == fid, name] = prediction[0]
     
    if all_df[name].dtypes == 'object':        
        print("%4d[%12s] ===> %s" %(fid, name, rlabs[prediction[0]]))
        all_df.loc[all_df['Id'] == fid, name] = rlabs[prediction[0]]
        kk.append(rlabs[prediction[0]])
    else:
        all_df.loc[all_df['Id'] == fid, name] = prediction[0]
        print("%4d[%12s] ===> %d" %(fid, name, prediction[0]))
        kk.append(np.round(prediction[0], 0))

        

for fid in [8, 13, 15, 17, 25, 32, 43, 44, 51, 65, 67, 77, 85, 96, 101, 105, 112, 114, 117, 121, 127, 132, 134, 137, 148, 150, 153, 154, 161, 167, 170, 171, 178, 181, 187, 192, 204, 208, 209, 215, 219, 222, 235, 238, 245, 250, 270, 288, 289, 294, 308, 309, 311, 320, 329, 331, 336, 343, 347, 348, 352, 357, 361, 362, 365, 367, 370, 371, 376, 385, 393, 394, 405, 406, 413, 422, 427, 448, 453, 458, 459, 460, 466, 471, 485, 491, 497, 517, 519, 530, 538, 539, 540, 542, 546, 560, 561, 565, 570, 581, 594, 611, 612, 613, 617, 624, 627, 642, 646, 661, 667, 669, 673, 680, 683, 686, 688, 691, 707, 710, 715, 721, 722, 727, 735, 746, 747, 752, 758, 771, 784, 786, 790, 792, 795, 812, 817, 818, 823, 829, 841, 846, 852, 854, 856, 857, 860, 866, 869, 880, 883, 894, 901, 905, 909, 912, 918, 926, 928, 929, 930, 940, 942, 945, 954, 962, 968, 976, 981, 984, 989, 997, 998, 1004, 1007, 1018, 1019, 1025, 1031, 1033, 1034, 1036, 1038, 1042, 1046, 1058, 1060, 1065, 1078, 1085, 1087, 1098, 1109, 1111, 1117, 1123, 1125, 1139, 1142, 1144, 1147, 1149, 1154, 1155, 1162, 1165, 1178, 1181, 1191, 1194, 1207, 1214, 1231, 1234, 1245, 1248, 1252, 1254, 1261, 1263, 1269, 1271, 1272, 1273, 1277, 1278, 1287, 1288, 1291, 1301, 1302, 1310, 1313, 1319, 1322, 1343, 1347, 1349, 1355, 1357, 1358, 1359, 1363, 1366, 1369, 1374, 1382, 1384, 1397, 1408, 1418, 1420, 1424, 1425, 1430, 1432, 1442, 1444, 1447, 1467, 1501, 1502, 1506, 1508, 1513, 1520, 1536, 1543, 1559, 1564, 1566, 1568, 1574, 1580, 1585, 1593, 1607, 1613, 1628, 1635, 1638, 1640, 1643, 1644, 1645, 1648, 1649, 1660, 1690, 1691, 1692, 1696, 1699, 1701, 1729, 1732, 1733, 1734, 1735, 1737, 1738, 1740, 1741, 1744, 1747, 1751, 1755, 1758, 1759, 1762, 1769, 1820, 1824, 1834, 1841, 1844, 1847, 1848, 1849, 1862, 1863, 1864, 1873, 1879, 1882, 1884, 1886, 1903, 1911, 1912, 1923, 1937, 1942, 1946, 1948, 1950, 1956, 1958, 1985, 1986, 1989, 1990, 1993, 1997, 2000, 2024, 2030, 2031, 2040, 2042, 2043, 2045, 2050, 2053, 2065, 2075, 2111, 2112, 2123, 2129, 2132, 2138, 2141, 2142, 2143, 2147, 2149, 2156, 2158, 2164, 2165, 2167, 2168, 2171, 2172, 2174, 2175, 2176, 2179, 2187, 2203, 2204, 2205, 2223, 2224, 2235, 2242, 2246, 2251, 2254, 2255, 2258, 2259, 2269, 2280, 2301, 2321, 2326, 2328, 2358, 2362, 2373, 2380, 2388, 2390, 2396, 2397, 2404, 2422, 2424, 2432, 2447, 2460, 2467, 2481, 2484, 2485, 2491, 2492, 2493, 2494, 2499, 2513, 2515, 2522, 2531, 2535, 2536, 2548, 2568, 2569, 2571, 2573, 2584, 2587, 2588, 2595, 2597, 2598, 2603, 2607, 2612, 2615, 2617, 2618, 2621, 2626, 2635, 2637, 2663, 2673, 2674, 2677, 2678, 2681, 2684, 2685, 2701, 2704, 2705, 2707, 2708, 2709, 2710, 2715, 2716, 2725, 2728, 2738, 2739, 2742, 2765, 2808, 2811, 2812, 2813, 2815, 2816, 2819, 2840, 2846, 2848, 2851, 2901, 2902, 2909]:
    fillValue('LotFrontage', fid)
    

print(kk)


'''
Strong Correlation
TotalBsmtSF <- 1stFlrSF
SalePrices <- GrLiveArea, TotalBsmtSF, OverallQual
'''
