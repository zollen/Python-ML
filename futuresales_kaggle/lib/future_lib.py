'''
Created on Jul. 17, 2021

@author: zollen
'''
import pandas as pd
import numpy as np


def add_item_avg_cnt(tokens, src, train, test):
    f1 = src.groupby(['date_block_num', 'item_id']).agg({'item_cnt_day': ['mean']})
    f1.columns = [ 'date_item_avg_cnt' ]
    train = train.merge(f1, on=['date_block_num', 'item_id'], how='left')
    train.fillna(0, inplace = True)
    test = test.merge(f1, on=['date_block_num', 'item_id'], how='left')
    test.fillna(0, inplace = True)
    tokens.append('date_item_avg_cnt')
    del f1
    return train, test
    
def add_date_item_avg_cnt(tokens, src, train, test):
    f1 = src.groupby(['date_block_num', 'shop_id', 'item_id']).agg({'item_cnt_day': ['mean']})
    f1.columns = [ 'date_shop_item_avg_cnt' ]
    train = train.merge(f1, on=['date_block_num', 'shop_id', 'item_id'], how='left')
    train.fillna(0, inplace = True)
    test = test.merge(f1, on=['date_block_num', 'shop_id', 'item_id'], how='left')
    test.fillna(0, inplace = True)
    tokens.append('date_shop_item_avg_cnt')
    del f1
    return train, test

def add_date_itemtype_cnt(tokens, src, train, test):
    f1 = src.groupby(['date_block_num', 'item_type']).agg({'item_cnt_day': ['mean']})
    f1.columns = [ 'date_itemtype_avg_cnt' ]
    train = train.merge(f1, on=['date_block_num', 'item_type'], how='left')
    train.fillna(0, inplace = True)
    test = test.merge(f1, on=['date_block_num', 'item_type'], how='left')
    test.fillna(0, inplace = True)
    tokens.append('date_itemtype_avg_cnt')
    del f1
    return train, test

def add_date_itemcat_cnt(tokens, src, train, test):
    f1 = src.groupby(['date_block_num', 'item_category_id']).agg({'item_cnt_day': ['mean']})
    f1.columns = [ 'date_itemcat_avg_cnt' ]
    train = train.merge(f1, on=['date_block_num', 'item_category_id'], how='left')
    train.fillna(0, inplace = True)
    test = test.merge(f1, on=['date_block_num', 'item_category_id'], how='left')
    test.fillna(0, inplace = True)
    tokens.append('date_itemcat_avg_cnt')
    del f1
    return train, test
   
def add_date_shop_subtype_avg_cnt(tokens, src, train, test):
    f1 = src.groupby(['date_block_num', 'shop_id', 'item_subtype']).agg({'item_cnt_day': ['mean']})
    f1.columns = [ 'date_shop_subtype_avg_cnt' ]
    train = train.merge(f1, on=['date_block_num', 'shop_id', 'item_subtype'], how='left')
    train.fillna(0, inplace = True)
    test = test.merge(f1, on=['date_block_num', 'shop_id', 'item_subtype'], how='left')
    test.fillna(0, inplace = True)    
    tokens.append('date_shop_subtype_avg_cnt')
    del f1
    return train, test

def add_date_name23_avg_cnt(tokens, src, train, test):
    f1 = src.groupby(['date_block_num', 'name2', 'name3']).agg({'item_cnt_day': ['mean']})
    f1.columns = [ 'date_name23_avg_cnt' ]
    train = train.merge(f1, on=['date_block_num', 'name2', 'name3'], how='left')
    train.fillna(0, inplace = True)
    test = test.merge(f1, on=['date_block_num', 'name2', 'name3'], how='left')
    test.fillna(0, inplace = True)    
    tokens.append('date_name23_avg_cnt')
    del f1
    return train, test
    
def add_delta_price(tokens, raw, train, test):
    f1 = raw.groupby(['item_id']).agg({"item_price": ["mean"]})
    f1.columns = ['item_avg']
    f2 = raw.groupby(['date_block_num', 'item_id']).agg({"item_price": ["mean"]})
    f2.columns = ['date_item_avg']
            
    all_dff = pd.concat([raw[raw['date_block_num'] == 33], test])
    f3 = all_dff.groupby('item_id').agg({"item_price": ["mean"]})
    f3.columns = ['date_item_avg']
   
    train = train.merge(f1, on=['item_id'], how='left')
    train = train.merge(f2, on=['date_block_num', 'item_id'], how='left')
    test = test.merge(f1, on=['item_id'], how='left')
    test = test.merge(f3, on=['item_id'], how='left')
    
    train.fillna(0, inplace=True)
    test.fillna(0, inplace=True)
    
    train['delta_price'] = (train['date_item_avg'] - train['item_avg']) / train['item_avg']
    test['delta_price'] = (test['date_item_avg'] - test['item_avg']) / test['item_avg']
    
    train.drop(columns=['date_item_avg', 'item_avg'], inplace = True)
    test.drop(columns=['date_item_avg', 'item_avg'], inplace = True)
    
    tokens.append('delta_price')
    del f1
    del f2
    del f3
    del all_dff
    return train, test
    
def add_delta_revenue(tokens, src, train, test):
   
    src['revenue'] = src['item_cnt_day'] * src['item_price']
    f1 = src.groupby( ["date_block_num", "shop_id"] ).agg({"revenue": ["sum"] })
    f1.columns = ['date_shop_revenue']
    f2 = src.groupby(['shop_id']).agg({ "revenue":["mean"] })
    f2.columns = ['shop_revenue']
    f3 = src[src['date_block_num'] == 33].groupby(["shop_id"]).agg({"revenue": ["sum"] })
    f3.columns = ['date_shop_revenue']
    
    train = train.merge(f1, on=['date_block_num', 'shop_id'], how='left')
    train = train.merge(f2, on=['shop_id'], how='left')
    test = test.merge(f3, on=['shop_id'], how='left')
    test = test.merge(f2, on=['shop_id'], how='left')
    
    train.fillna(0, inplace=True)
    test.fillna(0, inplace=True)
    
    train['delta_reveune'] = (train['date_shop_revenue'] - train['shop_revenue']) / train['shop_revenue']
    test['delta_reveune'] = (test['date_shop_revenue'] - test['shop_revenue']) / test['shop_revenue']
    
    tokens.append('delta_reveune')
    
    train.drop(columns=['date_shop_revenue', 'shop_revenue'], inplace = True)
    test.drop(columns=['date_shop_revenue', 'shop_revenue'], inplace = True)
    del f1
    del f2
    del f3
    return train, test

def add_sales_proximity(window, train, test):
    
    def calculate_proximity(vals):   
        score = 0
        lgth = len(vals)
        total = 11440   # np.sum(range(1^2, 33^2))
        for idx, row in zip(range(1, lgth + 1), vals):
            score += row * idx**2 / total     
        return score
    
    def apply_proximity(df):
        cnts = []
        for idx in range(0, len(df)):
            cnts.append(calculate_proximity(df.values[0:idx+1]))
        return cnts
    
    def populate_proximity(rec):
        populate_proximity.all_data[populate_proximity.indx] = rec.values
        populate_proximity.indx += 1
        
    def apply_groups(grp):
        dates = grp.sort_values('date_block_num')
        dates['sales_proximity'] = dates['item_cnt_month'].rolling(window).apply(calculate_proximity)
        dates.loc[dates['sales_proximity'].isna(), 'sales_proximity'] = apply_proximity(dates.loc[dates['sales_proximity'].isna(), 'item_cnt_month'])
        dates.apply(populate_proximity, axis=1)
        
    best = pd.read_csv('../data/best_prediction.csv')
    test = test.merge(best[['ID', 'item_cnt_month']], on=['ID'], how='left')    
  
    columns = ['date_block_num', 'shop_id', 'item_id', 'item_cnt_month']
    all_dff = pd.concat([train, test])
    all_dff = all_dff[columns]

    populate_proximity.all_data = np.zeros((len(all_dff), len(columns) + 1))
    populate_proximity.indx = 0
    all_dff.groupby(['shop_id', 'item_id']).apply(apply_groups)
     
    columns = columns + ['sales_proximity']
    tmp = pd.DataFrame(populate_proximity.all_data, columns=columns)
    tmp['sales_proximity'].fillna(0, inplace=True)
   
    train = train.merge(
        tmp.loc[tmp['date_block_num'] < 34, ['date_block_num', 'item_id', 'shop_id', 'sales_proximity']], 
            on=['date_block_num', 'item_id', 'shop_id'], how='left')
    
    test = test.merge(
        tmp.loc[tmp['date_block_num'] == 34, ['shop_id', 'item_id', 'sales_proximity']],
        on=['shop_id', 'item_id'], how='left')
    
    del best
    del all_dff
    del tmp
    del populate_proximity.all_data
    
    return train, test

def add_prepredict(tokens, train, test):
    
    guess_coeffs = [ 1.5380592989267657e-10, -9.879036042036577e-13, 
              -3.926352206002156e-12, -1.6920708677121077e-15, 
              3.089576979300765e-12, 7.015640157648066e-12, 
              1.8697303994525835e-12, -9.763763403779746e-13, 
              -1.3049033340668944e-14, -3.341402651609675e-11, 
              7.270162231507526e-13, 4.008777494178755e-15 ]
    
    def precfunc(rec):
        return guess_coeffs[0] +                            \
                guess_coeffs[1] * rec['date_block_num'] +   \
                guess_coeffs[2] * rec['shop_id'] +          \
                guess_coeffs[3] * rec['item_id'] +          \
                guess_coeffs[4] * rec['shop_category'] +    \
                guess_coeffs[5] * rec['shop_city'] +        \
                guess_coeffs[6] * rec['item_category_id'] + \
                guess_coeffs[7] * rec['name2'] +            \
                guess_coeffs[8] * rec['name3'] +            \
                guess_coeffs[9] * rec['item_type'] +        \
                guess_coeffs[10] * rec['item_subtype'] +    \
                guess_coeffs[11] * rec['item_price']
                            
    train['pre_predict'] = train.apply(precfunc, axis = 1).astype('int64').clip(0, 20)
    test['pre_predict'] = test.apply(precfunc, axis = 1).astype('int64').clip(0, 20)
    
    train = train.set_index(['shop_id', 'item_id'])
    test = test.set_index(['shop_id', 'item_id'])
    test.loc[~test.index.isin(train.index), 'pre_predict'] = 0
    train = train.reset_index()
    test = test.reset_index()

    tokens.append('pre_predict')
    
    return train, test
    
    
def add_lag_features(df, trailing_window_size, columns, targets):
    
    df_lagged = df.copy()
   
    for window in range(1, trailing_window_size + 1):
        shifted = df[columns + targets ].groupby(columns).shift(window)
        shifted.columns = [x + "_lag" + str(window) for x in df[targets]]
        df_lagged = pd.concat((df_lagged, shifted), axis=1)
    df_lagged.dropna(inplace=True)
    
    return df_lagged
    
