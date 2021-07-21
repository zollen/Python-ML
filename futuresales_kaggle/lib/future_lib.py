'''
Created on Jul. 17, 2021

@author: zollen
'''
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


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
    
def add_delta_price(tokens, train, test):
    all_dff = pd.concat([train, test])
    
    f1 = all_dff.groupby(['item_id']).agg({"item_price": ["mean"]})
    f1.columns = ['item_avg']
    f2 = all_dff.groupby(['date_block_num', 'item_id']).agg({"item_price": ["mean"]})
    f2.columns = ['date_item_avg']
   
    train = train.merge(f1, on=['item_id'], how='left')
    train = train.merge(f2, on=['date_block_num', 'item_id'], how='left')
    test = test.merge(f1, on=['item_id'], how='left')
    test = test.merge(f2, on=['date_block_num', 'item_id'], how='left')
    
    train.fillna(0, inplace=True)
    test.fillna(0, inplace=True)
    
    train['delta_price'] = (train['date_item_avg'] - train['item_avg']) / train['item_avg']
    test['delta_price'] = (test['date_item_avg'] - test['item_avg']) / test['item_avg']
    
    scaler = MinMaxScaler()
    train['delta_price'] = scaler.fit_transform(train['delta_price'].values.reshape(-1, 1))
    test['delta_price'] = scaler.transform(test['delta_price'].values.reshape(-1, 1))
    
    train.drop(columns=['date_item_avg', 'item_avg'], inplace = True)
    test.drop(columns=['date_item_avg', 'item_avg'], inplace = True)
    
    tokens.append('delta_price')
    del f1
    del f2
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


def add_lag_features(df, trailing_window_size, columns, targets):
    
    df_lagged = df.copy()
   
    for window in range(1, trailing_window_size + 1):
        shifted = df[columns + targets ].groupby(columns).shift(window)
        shifted.columns = [x + "_lag" + str(window) for x in df[targets]]
        df_lagged = pd.concat((df_lagged, shifted), axis=1)
    df_lagged.dropna(inplace=True)
    
    return df_lagged
    
