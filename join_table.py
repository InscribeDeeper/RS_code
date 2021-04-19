from dataset.load_dataset import load_click, load_order, load_sku, load_user
from dataset.join_dataset import join_click_order
from dataset.select_data import select_timerange
from dataset.transform import to_sequence
import pandas as pd




if __name__ == "__main__":
    df_click_order = join_click_order()
    df_sku = load_sku()

    print('begin to merge click table and sku table')
    df_click_order_sku = df_click_order.merge(df_sku, left_on='sku_ID',
                                                right_on = 'sku_ID')
    
    df_click_order_sku = df_click_order_sku.query('user_ID != "-"')    
    print('sort data using userID and request time')
    df_click_order_sku.sort_values(['user_ID', 'request_time'], inplace=True)
    print('write data into click_order_sku table')
    df_click_order_sku.to_csv('./data/click_order_sku.csv', index=None)

    del df_click_order_sku, df_sku


    df_user = load_user()
    print('begin to merge click table and user table')
    df_click_order_user = df_click_order.merge(df_user, left_on='user_ID',
                                                    right_on='user_ID')
    df_click_order_user.sort_values(['sku_ID', 'request_time'], inplace=True)
    print('write data into click_order_user table')
    df_click_order_user.to_csv('./data/click_order_user.csv', index=None)